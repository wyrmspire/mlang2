#!/usr/bin/env python3
"""
Backend Simulation Test

Mirrors what the frontend SimulationView does:
1. Load market data
2. Step through bars
3. Call CNN inference every N bars
4. Manage OCO brackets (entry, SL, TP)
5. Track wins/losses

Run:
    python tests/test_backend_simulation.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict


@dataclass
class OCOBracket:
    """OCO bracket tracking - same as frontend."""
    entry: float
    stop: float
    tp: float
    start_time: float
    direction: str  # 'LONG' or 'SHORT'


@dataclass
class Trade:
    """Completed trade record."""
    entry: float
    exit: float
    direction: str
    outcome: str  # 'WIN' or 'LOSS'
    bars_held: int
    pnl: float


def normalize_ohlcv(ohlcv_array):
    """
    Normalize OHLCV exactly as training/inference does.
    Input: (N, 5) array [open, high, low, close, volume]
    Output: (5, N) normalized array
    """
    x = ohlcv_array.T.copy()
    
    # Normalize OHLC by first close (percent change)
    first_close = x[3, 0]
    if first_close > 0:
        x[0:4] = (x[0:4] - first_close) / first_close * 100
    
    # Normalize volume by max
    max_vol = x[4].max()
    if max_vol > 0:
        x[4] = x[4] / max_vol
    else:
        x[4] = 0
    
    return x  # (5, N)


def run_simulation(
    model_path: str = "models/ifvg_4class_cnn.pth",
    start_date: str = None,
    days: int = 7,
    threshold: float = 0.35,
    stop_atr: float = 2.0,
    tp_atr: float = 4.0,
    lookback: int = 30,
    infer_every: int = 5,
    verbose: bool = True
) -> Dict:
    """
    Run simulation matching frontend logic.
    
    Returns summary of trades.
    """
    from src.data.loader import load_continuous_contract
    from src.models.model_registry_init import IFVG4ClassWrapper
    
    print("=" * 60)
    print("Backend Simulation Test")
    print("=" * 60)
    
    # Load model via registry wrapper
    print(f"\n[1] Loading model: {model_path}")
    model = IFVG4ClassWrapper(model_path=model_path)
    print("    Model loaded OK")
    
    # Load market data
    print("\n[2] Loading market data...")
    df = load_continuous_contract()
    
    if start_date:
        import pandas as pd
        start_dt = pd.Timestamp(start_date)
        if start_dt.tzinfo is None:
            start_dt = start_dt.tz_localize('America/New_York')
        df = df[df['time'] >= start_dt]
    
    # Limit to N days
    if len(df) > days * 390:  # ~390 bars per day
        df = df.head(days * 390)
    
    bars = df[['open', 'high', 'low', 'close', 'volume']].values
    times = df['time'].values
    
    print(f"    Loaded {len(bars)} bars")
    
    # Simulation state
    active_oco: Optional[OCOBracket] = None
    trades: List[Trade] = []
    triggers = 0
    
    print(f"\n[3] Running simulation (threshold={threshold}, stop={stop_atr}ATR, tp={tp_atr}ATR)")
    print("-" * 60)
    
    for idx in range(lookback, len(bars)):
        bar = bars[idx]
        current_open, current_high, current_low, current_close, _ = bar
        
        # Check OCO exit (same as frontend)
        if active_oco is not None:
            outcome = None
            exit_price = 0
            
            if active_oco.direction == 'LONG':
                if current_low <= active_oco.stop:
                    outcome = 'LOSS'
                    exit_price = active_oco.stop
                elif current_high >= active_oco.tp:
                    outcome = 'WIN'
                    exit_price = active_oco.tp
            else:  # SHORT
                if current_high >= active_oco.stop:
                    outcome = 'LOSS'
                    exit_price = active_oco.stop
                elif current_low <= active_oco.tp:
                    outcome = 'WIN'
                    exit_price = active_oco.tp
            
            if outcome:
                is_long = active_oco.direction == 'LONG'
                pnl = (exit_price - active_oco.entry) if is_long else (active_oco.entry - exit_price)
                pnl_dollars = pnl * 50  # MES multiplier
                
                trade = Trade(
                    entry=active_oco.entry,
                    exit=exit_price,
                    direction=active_oco.direction,
                    outcome=outcome,
                    bars_held=1,  # Simplified
                    pnl=pnl_dollars
                )
                trades.append(trade)
                
                if verbose:
                    print(f"  [{outcome}] {active_oco.direction} @ {active_oco.entry:.2f} -> {exit_price:.2f} = ${pnl_dollars:+.2f}")
                
                active_oco = None
        
        # Check for trigger (every N bars, no active trade)
        if active_oco is None and idx % infer_every == 0:
            # Build window of last 30 bars
            window = bars[idx - lookback + 1:idx + 1]
            
            # Calculate ATR
            recent = bars[max(0, idx - 13):idx + 1]
            atr = np.mean(recent[:, 1] - recent[:, 2])  # avg(high - low)
            if atr < 0.5:
                atr = current_close * 0.001
            
            # Normalize and predict
            ohlcv_norm = normalize_ohlcv(window)
            result = model.predict({'ohlcv': ohlcv_norm})
            
            # Check trigger
            if result['triggered'] and max(result['long_win_prob'], result['short_win_prob']) >= threshold:
                triggers += 1
                direction = result['direction']
                
                entry = current_close
                if direction == 'LONG':
                    stop = entry - (stop_atr * atr)
                    tp = entry + (tp_atr * atr)
                else:
                    stop = entry + (stop_atr * atr)
                    tp = entry - (tp_atr * atr)
                
                active_oco = OCOBracket(
                    entry=entry,
                    stop=stop,
                    tp=tp,
                    start_time=float(idx),
                    direction=direction
                )
                
                if verbose:
                    prob = result['long_win_prob'] if direction == 'LONG' else result['short_win_prob']
                    print(f"  [TRIGGER] {direction} @ {entry:.2f} (prob={prob:.2%}, SL={stop:.2f}, TP={tp:.2f})")
    
    # Summary
    print("-" * 60)
    wins = sum(1 for t in trades if t.outcome == 'WIN')
    losses = len(trades) - wins
    total_pnl = sum(t.pnl for t in trades)
    
    print(f"\n[4] Summary")
    print(f"    Triggers: {triggers}")
    print(f"    Trades:   {len(trades)}")
    print(f"    Wins:     {wins}")
    print(f"    Losses:   {losses}")
    print(f"    Win Rate: {wins/max(1, len(trades)):.1%}")
    print(f"    Total PnL: ${total_pnl:+.2f}")
    
    return {
        'triggers': triggers,
        'trades': len(trades),
        'wins': wins,
        'losses': losses,
        'win_rate': wins / max(1, len(trades)),
        'total_pnl': total_pnl,
        'trade_details': [
            {'entry': t.entry, 'exit': t.exit, 'direction': t.direction, 'outcome': t.outcome, 'pnl': t.pnl}
            for t in trades
        ]
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Backend simulation test")
    parser.add_argument("--model", default="models/ifvg_4class_cnn.pth", help="Model path")
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, default=7, help="Number of days")
    parser.add_argument("--threshold", type=float, default=0.35, help="Trigger threshold")
    parser.add_argument("--stop-atr", type=float, default=2.0, help="Stop loss ATR multiple")
    parser.add_argument("--tp-atr", type=float, default=4.0, help="Take profit ATR multiple")
    
    args = parser.parse_args()
    
    results = run_simulation(
        model_path=args.model,
        start_date=args.start,
        days=args.days,
        threshold=args.threshold,
        stop_atr=args.stop_atr,
        tp_atr=args.tp_atr
    )

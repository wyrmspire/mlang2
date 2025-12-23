#!/usr/bin/env python3
"""
ICT Inverted FVG Strategy Backtest Runner

Batch processing approach for efficient backtesting.
Processes data day-by-day selecting IFVG setups.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, Any, Optional, List

from src.data.loader import load_continuous_contract
from src.data.resample import resample_all_timeframes
from src.policy.library.ict_ifvg import ICTIFVGScanner, IFVGSetup
from src.config import POINT_VALUE, TICK_SIZE

# ============================================================================
# CONFIGURATION
# ============================================================================

NY_TZ = ZoneInfo("America/New_York")
RESULTS_DIR = Path("results/ict_ifvg")

# Strategy params
MIN_LIQUIDITY_SCORE = 2
INVERSION_WINDOW_BARS = 6  # 30 min on 5m
RISK_REWARD = 2.0
RISK_PER_TRADE = 300.0

# Trade window
TRADE_WINDOW_START = time(9, 30)
TRADE_WINDOW_END = time(15, 30)

# Data range
START_DATE = "2025-03-18"
WEEKS = 24


# ============================================================================
# HELPERS
# ============================================================================

def make_serializable(obj):
    """Convert numpy/pandas types to JSON-serializable Python types."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj


def get_raw_ohlcv_window(
    df_1m: pd.DataFrame,
    entry_time: pd.Timestamp,
    history_bars: int = 60,
    future_bars: int = 120
) -> List[Dict]:
    """Extract raw OHLCV data around a trade entry for visualization."""
    # Find the entry bar index
    mask = df_1m['time'] <= entry_time
    if not mask.any():
        return []
    
    entry_idx = mask.sum() - 1
    start_idx = max(0, entry_idx - history_bars)
    end_idx = min(len(df_1m), entry_idx + future_bars)
    
    window = df_1m.iloc[start_idx:end_idx]
    
    return [
        {
            "time": row['time'].isoformat(),
            "open": float(row['open']),
            "high": float(row['high']),
            "low": float(row['low']),
            "close": float(row['close']),
            "volume": int(row['volume'])
        }
        for _, row in window.iterrows()
    ]


def simulate_limit_order_outcome(
    df_1m: pd.DataFrame,
    entry_time: pd.Timestamp,
    entry_price: float,
    stop_price: float,
    tp_price: float,
    direction: str,
    max_bars: int = 240  # 4 hours
) -> Dict[str, Any]:
    """
    Simulate a limit order fill and outcome.
    
    Returns outcome dict with:
    - filled: bool
    - outcome: 'WIN', 'LOSS', or 'TIMEOUT'
    - exit_price: float
    - bars_held: int
    - pnl_points: float
    """
    # Get bars after entry time
    future = df_1m[df_1m['time'] > entry_time].head(max_bars)
    
    if future.empty:
        return {"filled": False, "outcome": "NO_DATA", "exit_price": 0, "bars_held": 0, "pnl_points": 0}
    
    filled = False
    fill_bar_idx = -1
    
    # Check for limit order fill
    for i, (idx, bar) in enumerate(future.iterrows()):
        if direction == "LONG":
            if bar['low'] <= entry_price:
                filled = True
                fill_bar_idx = i
                break
        else:  # SHORT
            if bar['high'] >= entry_price:
                filled = True
                fill_bar_idx = i
                break
    
    if not filled:
        return {"filled": False, "outcome": "NOT_FILLED", "exit_price": 0, "bars_held": 0, "pnl_points": 0}
    
    # Now simulate from fill to exit
    post_fill = future.iloc[fill_bar_idx:]
    
    for i, (idx, bar) in enumerate(post_fill.iterrows()):
        if direction == "LONG":
            # Check stop first (touched lower)
            if bar['low'] <= stop_price:
                pnl = stop_price - entry_price
                return {
                    "filled": True,
                    "outcome": "LOSS",
                    "exit_price": stop_price,
                    "bars_held": i + 1,
                    "pnl_points": pnl
                }
            # Check TP
            if bar['high'] >= tp_price:
                pnl = tp_price - entry_price
                return {
                    "filled": True,
                    "outcome": "WIN",
                    "exit_price": tp_price,
                    "bars_held": i + 1,
                    "pnl_points": pnl
                }
        else:  # SHORT
            # Check stop first (touched higher)
            if bar['high'] >= stop_price:
                pnl = entry_price - stop_price
                return {
                    "filled": True,
                    "outcome": "LOSS",
                    "exit_price": stop_price,
                    "bars_held": i + 1,
                    "pnl_points": pnl
                }
            # Check TP
            if bar['low'] <= tp_price:
                pnl = entry_price - tp_price
                return {
                    "filled": True,
                    "outcome": "WIN",
                    "exit_price": tp_price,
                    "bars_held": i + 1,
                    "pnl_points": pnl
                }
    
    # Timeout - exit at last close
    last_close = post_fill.iloc[-1]['close']
    if direction == "LONG":
        pnl = last_close - entry_price
    else:
        pnl = entry_price - last_close
    
    return {
        "filled": True,
        "outcome": "TIMEOUT",
        "exit_price": last_close,
        "bars_held": len(post_fill),
        "pnl_points": pnl
    }


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze_day(
    trading_date: datetime.date,
    df_1m: pd.DataFrame,
    df_5m: pd.DataFrame,
    scanner: ICTIFVGScanner
) -> List[Dict[str, Any]]:
    """
    Analyze a single trading day for IFVG setups.
    
    Returns list of trade records.
    """
    records = []
    
    # Define trade window
    window_start = datetime.combine(trading_date, TRADE_WINDOW_START).replace(tzinfo=NY_TZ)
    window_end = datetime.combine(trading_date, TRADE_WINDOW_END).replace(tzinfo=NY_TZ)
    
    # Filter 5m data to trade window
    df_5m_window = df_5m[
        (df_5m['time'] >= window_start) & 
        (df_5m['time'] <= window_end)
    ].copy()
    
    if df_5m_window.empty:
        return records
    
    # Calculate ATR from recent data
    df_5m_recent = df_5m[df_5m['time'] < window_start].tail(20)
    if len(df_5m_recent) > 0:
        atr = (df_5m_recent['high'] - df_5m_recent['low']).mean()
    else:
        atr = 5.0
    
    # Reset scanner for new day
    scanner.reset()
    
    # Warm up scanner with pre-window data
    warmup = df_5m[df_5m['time'] < window_start].tail(30)
    for idx in warmup.index:
        data_up_to = df_5m[df_5m.index <= idx]
        scanner.check(data_up_to, idx, atr)
    
    # Scan through trade window
    for bar_idx in df_5m_window.index:
        data_up_to = df_5m[df_5m.index <= bar_idx]
        setup = scanner.check(data_up_to, bar_idx, atr)
        
        if setup:
            # Get current bar info
            current_bar = df_5m.loc[bar_idx]
            entry_time = current_bar['time']
            
            # Simulate the trade
            outcome = simulate_limit_order_outcome(
                df_1m,
                entry_time,
                setup.entry_price,
                setup.stop_price,
                setup.tp_price,
                setup.direction
            )
            
            # Calculate contracts and PnL
            risk_points = abs(setup.entry_price - setup.stop_price)
            risk_dollars = risk_points * POINT_VALUE
            contracts = max(1, int(RISK_PER_TRADE / risk_dollars)) if risk_dollars > 0 else 1
            pnl_dollars = outcome['pnl_points'] * POINT_VALUE * contracts
            
            # Get OHLCV window for visualization
            raw_ohlcv = get_raw_ohlcv_window(df_1m, entry_time)
            
            # Build record
            record = {
                "decision_id": f"ifvg_{trading_date}_{bar_idx}",
                "timestamp": entry_time.isoformat(),
                "bar_idx": int(bar_idx),
                "scanner_id": scanner.scanner_id,
                "scanner_context": make_serializable(scanner.get_context(setup)),
                "window": {
                    "raw_ohlcv_1m": raw_ohlcv
                },
                "oco": {
                    "entry_price": setup.entry_price,
                    "stop_price": setup.stop_price,
                    "tp_price": setup.tp_price,
                    "direction": setup.direction,
                    "contracts": contracts,
                    "order_type": "LIMIT"
                },
                "oco_results": {
                    "filled": outcome['filled'],
                    "outcome": outcome['outcome'],
                    "exit_price": outcome['exit_price'],
                    "bars_held": outcome['bars_held'],
                    "pnl_points": outcome['pnl_points'],
                    "pnl_dollars": pnl_dollars
                }
            }
            records.append(record)
    
    return records


def main():
    print("=" * 50)
    print("ICT Inverted FVG Strategy Backtest")
    print("=" * 50)
    
    # Load data
    print("Loading data...")
    df_1m = load_continuous_contract()
    htf = resample_all_timeframes(df_1m)
    df_5m = htf['5m']
    
    # Ensure time column
    if 'time' not in df_5m.columns and df_5m.index.name == 'time':
        df_5m = df_5m.reset_index()
    
    print(f"  {len(df_1m)} 1m bars, {len(df_5m)} 5m bars")
    
    # Date range
    start = pd.Timestamp(START_DATE).tz_localize(NY_TZ)
    end = start + timedelta(weeks=WEEKS)
    
    # Get trading days
    trading_days = pd.date_range(start.date(), end.date(), freq='B')  # Business days
    
    # Create scanner
    scanner = ICTIFVGScanner(
        min_liquidity_score=MIN_LIQUIDITY_SCORE,
        inversion_window_bars=INVERSION_WINDOW_BARS,
        risk_reward=RISK_REWARD,
        max_risk_dollars=RISK_PER_TRADE
    )
    
    # Process each day
    all_records = []
    
    for day in trading_days:
        day_date = day.date()
        records = analyze_day(day_date, df_1m, df_5m, scanner)
        all_records.extend(records)
        
        if records:
            print(f"  {day_date}: {len(records)} setups")
    
    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Records JSONL
    records_file = RESULTS_DIR / "records.jsonl"
    with open(records_file, 'w') as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")
    
    # Summary
    total = len(all_records)
    filled = [r for r in all_records if r['oco_results']['filled']]
    wins = [r for r in filled if r['oco_results']['outcome'] == 'WIN']
    total_pnl = sum(r['oco_results']['pnl_dollars'] for r in filled)
    
    summary = {
        "strategy": "ICT Inverted FVG",
        "date_range": f"{START_DATE} to {end.date()}",
        "total_setups": total,
        "filled_trades": len(filled),
        "wins": len(wins),
        "win_rate": len(wins) / len(filled) * 100 if filled else 0,
        "total_pnl": total_pnl,
        "params": {
            "min_liquidity_score": MIN_LIQUIDITY_SCORE,
            "inversion_window_bars": INVERSION_WINDOW_BARS,
            "risk_reward": RISK_REWARD,
            "risk_per_trade": RISK_PER_TRADE
        }
    }
    
    with open(RESULTS_DIR / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("=" * 50)
    print(f"Trades: {len(filled)} | Wins: {len(wins)} | WR: {summary['win_rate']:.0f}% | PnL: ${total_pnl:.0f}")
    print(f"Saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()

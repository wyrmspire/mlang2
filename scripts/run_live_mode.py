#!/usr/bin/env python3
"""
Live Mode Simulator

Runs a strategy in "Real-Time" (Simulated execution on real data).
- Loads 7 days history from YFinance.
- Simulates past days to build equity curve.
- Enters LIVE mode and waits for new bars to trade in real-time.
- Emits JSON events for the frontend UI.

Usage:
    python scripts/run_live_mode.py --ticker MES=F --strategy ema_cross
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime

from src.sim.yfinance_stepper import YFinanceStepper
from src.features.indicators import calculate_atr, calculate_ema
from src.policy.library.ict_ifvg import ICTIFVGScanner
from src.policy.entry_scans import EntryOrder, EntryConfig, apply_entry_scans

# Global scanner instance (stateful)
_ifvg_scanner = None

# =============================================================================
# Strategy Logic
# =============================================================================

def check_ema_cross(df_history: pd.DataFrame) -> dict:
    """Check for 9/21 EMA cross."""
    if len(df_history) < 30:
        return None
        
    df = df_history.copy()
    df['ema_fast'] = calculate_ema(df['close'], 9)
    df['ema_slow'] = calculate_ema(df['close'], 21)
    
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Cross UP
    if prev['ema_fast'] <= prev['ema_slow'] and curr['ema_fast'] > curr['ema_slow']:
        return {'direction': 'LONG', 'confidence': 0.8}
    
    # Cross DOWN
    if prev['ema_fast'] >= prev['ema_slow'] and curr['ema_fast'] < curr['ema_slow']:
        return {'direction': 'SHORT', 'confidence': 0.8}
        
    return None

def check_orb(df_history: pd.DataFrame) -> dict:
    """Simple ORB (9:30-10:00 range) check."""
    # Need access to full day session. This is harder with just a tail df.
    # Placeholder for now.
    return None


def check_ifvg(df_history: pd.DataFrame, bar_idx: int) -> dict:
    """
    Check for IFVG setup using the real ICTIFVGScanner.
    Returns signal dict with direction, entry, stop, tp or None.
    """
    global _ifvg_scanner
    if _ifvg_scanner is None:
        _ifvg_scanner = ICTIFVGScanner(
            min_liquidity_score=2,  # Relaxed for more signals
            inversion_window_bars=6,
            swing_lookback=5,
            min_gap_atr=0.15,
            risk_reward=2.0,
            cooldown_bars=6
        )
    
    if len(df_history) < 30:
        return None
    
    # Calculate ATR
    atr_series = calculate_atr(df_history, 14)
    atr = float(atr_series.iloc[-1]) if len(atr_series) > 0 else 5.0
    
    # Check for setup
    setup = _ifvg_scanner.check(df_history, bar_idx, atr=atr)
    
    if setup:
        return {
            'direction': setup.direction,  # 'LONG' or 'SHORT'
            'confidence': 0.7 + (setup.liquidity_score * 0.05),  # Score-based confidence
            'entry_price': setup.entry_price,
            'stop_price': setup.stop_price,
            'tp_price': setup.tp_price
        }
    return None

# =============================================================================
# Helper
# =============================================================================

def emit(event_type: str, data: dict):
    """Emit JSON event."""
    msg = {'type': event_type, **data}
    print(json.dumps(msg), flush=True)

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Live Trading Simulator")
    parser.add_argument("--ticker", type=str, default="MES=F", help="Ticker symbol")
    parser.add_argument("--strategy", type=str, default="ema_cross", help="Strategy to run")
    parser.add_argument("--days", type=int, default=7, help="History days")
    parser.add_argument("--speed", type=float, default=10.0, help="Historical playback speed")
    
    # Entry scan configuration
    parser.add_argument("--entry-type", type=str, default="market", 
                       choices=["market", "limit"], help="Entry type")
    parser.add_argument("--stop-method", type=str, default="atr",
                       choices=["atr", "swing", "fixed_bars"], help="Stop placement method")
    parser.add_argument("--tp-method", type=str, default="atr",
                       choices=["atr", "r_multiple"], help="Take profit method")
    parser.add_argument("--stop-atr", type=float, default=1.0, help="ATR multiple for stop")
    parser.add_argument("--tp-atr", type=float, default=2.0, help="ATR multiple for TP")
    parser.add_argument("--tp-r", type=float, default=2.0, help="R-multiple for TP")
    
    args = parser.parse_args()
    
    # Build entry config
    entry_config = EntryConfig(
        entry_type=args.entry_type,
        stop_method=args.stop_method,
        tp_method=args.tp_method,
        stop_atr_multiple=args.stop_atr,
        tp_atr_multiple=args.tp_atr,
        tp_r_multiple=args.tp_r
    )
    
    emit('STATUS', {'message': f'Initializing Live Mode for {args.ticker}...'})
    
    try:
        stepper = YFinanceStepper(ticker=args.ticker, days_back=args.days)
    except Exception as e:
        emit('ERROR', {'message': f'Failed to init stepper: {str(e)}'})
        return

    emit('REPLAY_START', {
        'start_date': str(stepper.df['time'].iloc[0]),
        'total_bars': len(stepper.df),
        'strategy': args.strategy,
        'mode': 'LIVE_SIMULATION'
    })

    # EMIT INITIAL HISTORY BATCH
    # Convert entire history DataFrame to list of dicts for frontend
    history_bars = []
    for _, row in stepper.df.iterrows():
        history_bars.append({
            'timestamp': str(row['time']),
            'close': float(row['close']),
            'high': float(row['high']),
            'low': float(row['low']),
            'open': float(row['open']),
            'volume': float(row['volume'])
        })
    emit('HISTORY', {'bars': history_bars})
    
    decision_count = 0
    # bar_delay = 1.0 / args.speed # No delay needed for history batch
    
    print(f"Processing history...", file=sys.stderr)
    
    live_mode_notified = False

    while True:
        # Step
        step = stepper.step()
        
        # If None, it means we are waiting for live data
        if step is None and stepper.live_mode:
            if not live_mode_notified:
                print(">>> ENTERING LIVE MODE - Waiting for market updates <<<", file=sys.stderr)
                emit('STATUS', {'message': 'History complete. Entered LIVE mode.'})
                live_mode_notified = True
            time.sleep(1)
            continue
            
        bar = step.bar
        
        # Determine if this is a "New Live Bar" or "History Bar"
        # Since we sent history in batch, we ONLY emit BAR events for new updates
        # How to distinguish? `stepper.live_mode` might be set AFTER we consume history.
        # But `stepper.step()` returns bars from the DF first.
        # Simple check: Is this bar's timestamp in our initial history batch?
        # A crude but effective way is just to check `stepper.live_mode`.
        # However, `YFinanceStepper` might not set `live_mode=True` until it exhausts history.
        
        # Logic:
        # If we are in history (not live_mode), do NOT emit BAR (frontend has it).
        # We STILL run strategy to track state/trades.
        # If we are live (live_mode=True), we EMIT BAR.
        
        if stepper.live_mode:
            emit('BAR', {
                'bar_idx': step.bar_idx,
                'timestamp': str(bar['time']),
                'close': float(bar['close']),
                'high': float(bar['high']),
                'low': float(bar['low']),
                'open': float(bar['open']),
                'volume': float(bar['volume'])
            })
        
        # Run Strategy
        history = stepper.get_history(lookback=60)
        signal = None
        
        if args.strategy == "ema_cross":
            signal = check_ema_cross(history)
        elif args.strategy == "ifvg":
            signal = check_ifvg(history, step.bar_idx)
            
        if signal:
            decision_count += 1
            emit('DECISION', {
                'decision_id': f'live_{decision_count:04d}',
                'bar_idx': step.bar_idx,
                'timestamp': str(bar['time']),
                'direction': signal['direction'],
                'confidence': signal['confidence'],
                'price': float(bar['close']),
                'triggered': True
            })
            
            # Build base order from signal
            if 'entry_price' in signal and 'stop_price' in signal and 'tp_price' in signal:
                # Signal provides full bracket (IFVG)
                base_order = EntryOrder(
                    direction=signal['direction'],
                    entry_price=signal['entry_price'],
                    stop_price=signal['stop_price'],
                    tp_price=signal['tp_price']
                )
            else:
                # Start with market entry, will be modified by entry scans
                base_order = EntryOrder(
                    direction=signal['direction'],
                    entry_price=float(bar['close']),
                    stop_price=float(bar['close']),  # Will be recalculated
                    tp_price=float(bar['close'])     # Will be recalculated
                )
            
            # Apply entry scans to modify the order
            current_bar = pd.Series({
                'open': bar['open'], 'high': bar['high'], 
                'low': bar['low'], 'close': bar['close']
            })
            final_order = apply_entry_scans(base_order, history, entry_config, current_bar)
            
            emit('OCO_OPEN', {
                'decision_id': f'live_{decision_count:04d}',
                'direction': final_order.direction,
                'entry_price': round(final_order.entry_price, 2),
                'stop_price': round(final_order.stop_price, 2),
                'tp_price': round(final_order.tp_price, 2),
                'entry_type': final_order.entry_type
            })
            
        # No artificial delay needed in history since we sent batch

if __name__ == "__main__":
    main()

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
    
    args = parser.parse_args()
    
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
    
    decision_count = 0
    bar_delay = 1.0 / args.speed
    
    print(f"Starting simulation... (History speed: {args.speed}x)", file=sys.stderr)
    
    while True:
        # Step
        step = stepper.step()
        bar = step.bar
        
        # If we just entered live mode, notify
        if stepper.live_mode and bar_delay != 1.0:
            print(">>> ENTERING LIVE MODE - Waiting for market updates <<<", file=sys.stderr)
            emit('STATUS', {'message': 'History complete. Entered LIVE mode.'})
            bar_delay = 1.0  # Reset speed to real-time
        
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
            
            # Simple bracket
            atr = calculate_atr(history, 14).iloc[-1]
            entry = float(bar['close'])
            
            if signal['direction'] == 'LONG':
                tp = entry + (atr * 2)
                sl = entry - (atr * 1)
            else:
                tp = entry - (atr * 2)
                sl = entry + (atr * 1)
            
            emit('OCO_OPEN', {
                'decision_id': f'live_{decision_count:04d}',
                'direction': signal['direction'],
                'entry_price': round(entry, 2),
                'stop_price': round(sl, 2),
                'tp_price': round(tp, 2)
            })
            
        # Pacing
        # In history: sleep(delay)
        # In live: poll already slept in stepper, so we don't need to sleep here?
        # Actually stepper returns immediately if bar found.
        # But if we loop tight, we might process same bar? No, step() increments.
        # So we just need delay for visualization of history.
        if not stepper.live_mode:
            time.sleep(bar_delay)

if __name__ == "__main__":
    main()

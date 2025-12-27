#!/usr/bin/env python3
"""
Live Mode Simulator

Runs a strategy in "Real-Time" (Simulated execution on real data).
- Loads 7 days history from YFinance.
- Simulates past days to build equity curve.
- Enters LIVE mode and waits for new bars to trade in real-time.
- Emits JSON events for the frontend UI.

Usage:
    python scripts/session_live.py --ticker MES=F --strategy ema_cross
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
    parser.add_argument("--entry-params", type=str, default="{}", help="JSON entry params")
    
    args = parser.parse_args()
    
    # Initialize OCO Engine
    from src.sim.oco_engine import OCOEngine, OCOConfig, StopConfig
    from src.sim.costs import DEFAULT_COSTS
    
    oco_engine = OCOEngine(costs=DEFAULT_COSTS)
    entry_params = json.loads(args.entry_params)
    
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
            
            # Use OCOEngine to create bracket
            # If signal provides explicit prices (IFVG), favor those?
            # Or use dynamic entry parameters if override not set?
            
            # For this modular update, we prioritize the USER SELECTED entry strategy
            # UNLESS the signal is "High Precision" (like IFVG providing exact levels).
            # But the user specifically asked for modular entry tools.
            # So we will use the OCO Engine's calculation.
            
            # Construct OCO Config
            config = OCOConfig(
                direction=signal['direction'],
                entry_type=args.entry_type.upper(), # 'MARKET', 'LIMIT', 'RETRACE_SIGNAL'
                entry_params=entry_params,
                stop_atr=args.stop_atr,
                tp_multiple=args.tp_r,
                entry_offset_atr=0.0 # Legacy
            )
            
            # Create Bracket (Calculates prices)
            atr_series = calculate_atr(history, 14)
            atr = float(atr_series.iloc[-1]) if len(atr_series) > 0 else 5.0
            
            bracket = oco_engine.create_bracket(
                config=config,
                base_price=float(bar['close']),
                atr=atr,
                df_1m=history, # Pass history as 1m context
                df_htf=history, # Pass history as htf (simplification for now)
                current_idx=len(history)-1
            )
            
            emit('OCO_OPEN', {
                'decision_id': f'live_{decision_count:04d}',
                'direction': bracket.config.direction,
                'entry_price': round(bracket.entry_price, 2),
                'stop_price': round(bracket.stop_price, 2),
                'tp_price': round(bracket.tp_price, 2),
                'entry_type': bracket.config.entry_type
            })
            
        # No artificial delay needed in history since we sent batch

if __name__ == "__main__":
    main()

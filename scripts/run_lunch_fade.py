#!/usr/bin/env python3
"""
Lunch Hour Fade Strategy

Theory: Breakouts fail between 11:30 AM - 1:00 PM EST (lunch hours).
Action: SHORT when price breaks above 15-minute swing high during lunch.
Stop: 0.5 ATR (tight - get out if it's a real breakout)
Target: 2R (risk/reward = 1:2)

Run:
    python scripts/run_lunch_fade.py --days 28
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, time
from typing import Dict, List, Any
from zoneinfo import ZoneInfo

from src.features.indicators import calculate_atr
from src.storage import ExperimentDB


# Strategy parameters
LUNCH_START = time(11, 30)  # 11:30 AM EST
LUNCH_END = time(13, 0)     # 1:00 PM EST
SWING_LOOKBACK = 15         # 15 bars for swing high detection
STOP_ATR_MULT = 0.5         # Tight stop
TP_R_MULT = 2.0             # 2R target
EST = ZoneInfo("America/New_York")


def find_swing_high(highs: np.ndarray, idx: int, lookback: int = 5) -> float:
    """
    Find the most recent swing high (higher than neighbors).
    Returns the swing high price, or np.nan if none found.
    """
    if idx < lookback * 2:
        return np.nan
    
    # Look for swing highs in the lookback window
    for i in range(idx - lookback, max(lookback, idx - lookback * 3), -1):
        # Is this a swing high? (higher than bars before and after)
        if i >= lookback and i < len(highs) - lookback:
            center = highs[i]
            left_max = max(highs[i-lookback:i])
            right_max = max(highs[i+1:i+lookback+1]) if i + lookback + 1 <= len(highs) else center
            
            if center >= left_max and center >= right_max:
                return center
    
    # Fallback: use rolling max
    return max(highs[idx-lookback:idx])


def is_lunch_hour(t: datetime) -> bool:
    """Check if time is in lunch hours (11:30 AM - 1:00 PM EST)."""
    # Convert to EST if needed
    if t.tzinfo is None:
        t = t.replace(tzinfo=EST)
    else:
        t = t.astimezone(EST)
    
    current_time = t.time()
    return LUNCH_START <= current_time <= LUNCH_END


def run_lunch_fade_strategy(days: int = 28, verbose: bool = True) -> Dict[str, Any]:
    """
    Run the Lunch Hour Fade strategy simulation.
    
    Args:
        days: Number of days to simulate (max 7 for 1m data)
        verbose: Print trade details
    
    Returns:
        Dict with strategy results
    """
    print("=" * 60)
    print("LUNCH HOUR FADE STRATEGY")
    print("=" * 60)
    print(f"Theory: Fade breakouts during 11:30 AM - 1:00 PM EST")
    print(f"Action: SHORT on 15m swing high break during lunch")
    print(f"Stop: {STOP_ATR_MULT} ATR | Target: {TP_R_MULT}R")
    print("=" * 60)
    
    # Load data (yfinance 1m limit is 7 days)
    actual_days = min(days, 7)
    if days > 7:
        print(f"\n[!] yfinance 1m limit is 7 days. Running with {actual_days} days.")
    
    end = datetime.now()
    start = end - timedelta(days=actual_days)
    
    print(f"\n[1] Loading {actual_days} days of ES data...")
    ticker = yf.Ticker("ES=F")
    df = ticker.history(start=start, end=end, interval="1m")
    
    if df is None or len(df) == 0:
        print("ERROR: No data available")
        return {'trades': 0, 'win_rate': 0, 'total_pnl': 0}
    
    # Standardize columns
    df.columns = [c.lower() for c in df.columns]
    df = df.reset_index()
    df['time'] = df['Datetime'] if 'Datetime' in df.columns else df['datetime']
    
    print(f"    Loaded {len(df)} bars")
    
    # Calculate ATR
    print("\n[2] Computing indicators...")
    df['atr'] = calculate_atr(df, period=14)
    df['atr'] = df['atr'].ffill().bfill()  # Fill NaN
    
    # Run simulation
    print(f"\n[3] Scanning for lunch hour breakouts...")
    
    trades = []
    active_trade = None
    lookback = SWING_LOOKBACK
    
    for i in range(lookback + 14, len(df)):
        current_bar = df.iloc[i]
        current_time = pd.to_datetime(current_bar['time'])
        current_price = current_bar['close']
        current_high = current_bar['high']
        current_low = current_bar['low']
        atr = current_bar['atr']
        
        if pd.isna(atr) or atr < 0.5:
            continue
        
        # Check if in active trade
        if active_trade is not None:
            # Check stop loss (price goes UP for short)
            if current_high >= active_trade['stop']:
                pnl = -(active_trade['stop'] - active_trade['entry']) * 50  # Negative for loss
                trades.append({
                    'entry_time': active_trade['entry_time'],
                    'exit_time': current_time,
                    'entry': active_trade['entry'],
                    'exit': active_trade['stop'],
                    'pnl': pnl,
                    'result': 'LOSS'
                })
                if verbose:
                    print(f"  [LOSS] SHORT @ {active_trade['entry']:.2f} -> SL @ {active_trade['stop']:.2f} = ${pnl:.2f}")
                active_trade = None
            # Check take profit (price goes DOWN for short)
            elif current_low <= active_trade['tp']:
                pnl = (active_trade['entry'] - active_trade['tp']) * 50  # Positive for win
                trades.append({
                    'entry_time': active_trade['entry_time'],
                    'exit_time': current_time,
                    'entry': active_trade['entry'],
                    'exit': active_trade['tp'],
                    'pnl': pnl,
                    'result': 'WIN'
                })
                if verbose:
                    print(f"  [WIN] SHORT @ {active_trade['entry']:.2f} -> TP @ {active_trade['tp']:.2f} = ${pnl:.2f}")
                active_trade = None
            continue
        
        # Look for entry during lunch hours only
        if not is_lunch_hour(current_time):
            continue
        
        # Find swing high
        swing_high = find_swing_high(df['high'].values, i, lookback)
        if pd.isna(swing_high):
            continue
        
        # Check for breakout (price breaks above swing high)
        if current_high > swing_high:
            # SHORT entry
            entry = current_price
            stop = entry + (atr * STOP_ATR_MULT)
            risk = stop - entry
            tp = entry - (risk * TP_R_MULT)
            
            active_trade = {
                'entry_time': current_time,
                'entry': entry,
                'stop': stop,
                'tp': tp,
                'swing_high': swing_high,
            }
            
            if verbose:
                print(f"  [TRIGGER] SHORT @ {entry:.2f} (broke {swing_high:.2f}) SL={stop:.2f} TP={tp:.2f}")
    
    # Summary
    wins = sum(1 for t in trades if t['result'] == 'WIN')
    losses = len(trades) - wins
    total_pnl = sum(t['pnl'] for t in trades)
    win_rate = wins / len(trades) if trades else 0
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Period: {actual_days} days")
    print(f"  Total Trades: {len(trades)}")
    print(f"  Wins: {wins} | Losses: {losses}")
    print(f"  Win Rate: {win_rate:.1%}")
    print(f"  Total PnL: ${total_pnl:.2f}")
    if trades:
        print(f"  Avg PnL/Trade: ${total_pnl/len(trades):.2f}")
    
    return {
        'trades': len(trades),
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'strategy': 'lunch_fade',
        'params': {
            'lunch_start': str(LUNCH_START),
            'lunch_end': str(LUNCH_END),
            'stop_atr': STOP_ATR_MULT,
            'tp_r': TP_R_MULT,
        }
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Lunch Hour Fade Strategy")
    parser.add_argument("--days", type=int, default=7, help="Days to simulate")
    parser.add_argument("--save", action="store_true", help="Save to ExperimentDB")
    parser.add_argument("--quiet", action="store_true", help="Suppress trade details")
    
    args = parser.parse_args()
    
    results = run_lunch_fade_strategy(days=args.days, verbose=not args.quiet)
    
    if args.save and results['trades'] > 0:
        db = ExperimentDB()
        run_id = f"lunch_fade_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        db.store_run(
            run_id=run_id,
            strategy="lunch_fade",
            config=results['params'],
            metrics={
                'total_trades': results['trades'],
                'wins': results['wins'],
                'losses': results['losses'],
                'win_rate': results['win_rate'],
                'total_pnl': results['total_pnl'],
            }
        )
        print(f"\n[+] Saved to ExperimentDB: {run_id}")

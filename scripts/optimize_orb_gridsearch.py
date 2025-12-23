#!/usr/bin/env python3
"""
Opening Range Breakout - Grid Search

Massive parameter sweep:
- Stop Loss: 0.5 to 3.0 ATR (0.25 increments) = 11 values
- Take Profit: 1.0 to 5.0 R = 9 values
- Total: 99 combinations

Optimizes for: PROFIT FACTOR (not just Net PnL)
Profit Factor = Gross Wins / Gross Losses

Run:
    python scripts/run_orb_gridsearch.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, time
from typing import Dict, Any, List, Tuple
from zoneinfo import ZoneInfo
import itertools

from src.storage import ExperimentDB


# Strategy parameters
OR_START = time(9, 30)   # Opening range start (RTH open)
OR_END = time(10, 0)     # Opening range end (first 30 min)
ATR_PERIOD = 14
EST = ZoneInfo("America/New_York")

# Grid search parameters
STOP_ATR_RANGE = np.arange(0.5, 3.25, 0.25)  # 0.5, 0.75, 1.0, ..., 3.0
TP_R_RANGE = np.arange(1.0, 5.5, 0.5)        # 1.0, 1.5, 2.0, ..., 5.0


def run_orb_single(
    df: pd.DataFrame,
    stop_atr: float,
    tp_r: float,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run Opening Range Breakout for a single parameter combination.
    
    Returns trade stats including profit factor.
    """
    trades = []
    active_trade = None
    
    # Pre-compute daily stats
    df = df.copy()
    df['date'] = df['time'].dt.date
    df['hour'] = df['time'].dt.hour
    df['minute'] = df['time'].dt.minute
    
    # Calculate ATR
    high = df['high']
    low = df['low']
    close = df['close']
    tr = pd.concat([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1))
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=ATR_PERIOD).mean().shift(1)
    
    # Group by date for opening range calculation
    dates = df['date'].unique()
    
    for date in dates:
        day_data = df[df['date'] == date].copy()
        if len(day_data) < 60:  # Need enough bars
            continue
        
        # Find opening range (9:30 - 10:00)
        or_data = day_data[
            (day_data['hour'] == 9) & (day_data['minute'] >= 30) |
            (day_data['hour'] == 10) & (day_data['minute'] == 0)
        ]
        
        if len(or_data) < 5:
            continue
        
        or_high = or_data['high'].max()
        or_low = or_data['low'].min()
        or_close_idx = or_data.index[-1]
        
        # Get ATR at OR close
        atr = day_data.loc[or_close_idx, 'atr']
        if pd.isna(atr) or atr < 0.5:
            continue
        
        # Trade after OR (10:00 onwards)
        after_or = day_data[day_data.index > or_close_idx]
        
        for idx, bar in after_or.iterrows():
            if active_trade is not None:
                # Check exit conditions
                if active_trade['direction'] == 'LONG':
                    if bar['low'] <= active_trade['stop']:
                        pnl = (active_trade['stop'] - active_trade['entry']) * 50
                        trades.append({'pnl': pnl, 'result': 'LOSS', 'gross': pnl})
                        active_trade = None
                    elif bar['high'] >= active_trade['tp']:
                        pnl = (active_trade['tp'] - active_trade['entry']) * 50
                        trades.append({'pnl': pnl, 'result': 'WIN', 'gross': pnl})
                        active_trade = None
                else:  # SHORT
                    if bar['high'] >= active_trade['stop']:
                        pnl = (active_trade['entry'] - active_trade['stop']) * 50
                        trades.append({'pnl': pnl, 'result': 'LOSS', 'gross': pnl})
                        active_trade = None
                    elif bar['low'] <= active_trade['tp']:
                        pnl = (active_trade['entry'] - active_trade['tp']) * 50
                        trades.append({'pnl': pnl, 'result': 'WIN', 'gross': pnl})
                        active_trade = None
                continue
            
            # Check for breakout entry
            if bar['high'] > or_high:
                # LONG breakout
                entry = bar['close']
                stop = entry - (atr * stop_atr)
                risk = entry - stop
                tp = entry + (risk * tp_r)
                
                active_trade = {
                    'direction': 'LONG',
                    'entry': entry,
                    'stop': stop,
                    'tp': tp
                }
                
            elif bar['low'] < or_low:
                # SHORT breakout
                entry = bar['close']
                stop = entry + (atr * stop_atr)
                risk = stop - entry
                tp = entry - (risk * tp_r)
                
                active_trade = {
                    'direction': 'SHORT',
                    'entry': entry,
                    'stop': stop,
                    'tp': tp
                }
    
    # Calculate stats
    if not trades:
        return {
            'trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0,
            'total_pnl': 0, 'profit_factor': 0,
            'stop_atr': stop_atr, 'tp_r': tp_r
        }
    
    wins = sum(1 for t in trades if t['result'] == 'WIN')
    losses = len(trades) - wins
    total_pnl = sum(t['pnl'] for t in trades)
    
    gross_wins = sum(t['gross'] for t in trades if t['gross'] > 0)
    gross_losses = abs(sum(t['gross'] for t in trades if t['gross'] < 0))
    
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')
    
    return {
        'trades': len(trades),
        'wins': wins,
        'losses': losses,
        'win_rate': wins / len(trades) if trades else 0,
        'total_pnl': total_pnl,
        'gross_wins': gross_wins,
        'gross_losses': gross_losses,
        'profit_factor': profit_factor,
        'stop_atr': stop_atr,
        'tp_r': tp_r
    }


def run_gridsearch(days: int = 7) -> List[Dict]:
    """
    Run the full grid search.
    """
    print("=" * 70)
    print("OPENING RANGE BREAKOUT - GRID SEARCH")
    print("=" * 70)
    print(f"Stop ATR: {STOP_ATR_RANGE[0]:.2f} to {STOP_ATR_RANGE[-1]:.2f} ({len(STOP_ATR_RANGE)} values)")
    print(f"TP R:     {TP_R_RANGE[0]:.1f} to {TP_R_RANGE[-1]:.1f} ({len(TP_R_RANGE)} values)")
    print(f"Total combinations: {len(STOP_ATR_RANGE) * len(TP_R_RANGE)}")
    print("=" * 70)
    
    # Load data (max 7 days for 1m)
    actual_days = min(days, 7)
    end = datetime.now()
    start = end - timedelta(days=actual_days)
    
    print(f"\n[1] Loading {actual_days} days of ES data...")
    ticker = yf.Ticker("ES=F")
    df = ticker.history(start=start, end=end, interval="1m")
    
    if df is None or len(df) == 0:
        print("ERROR: No data!")
        return []
    
    # Standardize
    df.columns = [c.lower() for c in df.columns]
    df = df.reset_index()
    col_name = 'Datetime' if 'Datetime' in df.columns else 'datetime'
    df['time'] = pd.to_datetime(df[col_name]).dt.tz_convert(EST)
    
    print(f"    Loaded {len(df)} bars")
    
    # Run grid search
    print(f"\n[2] Running {len(STOP_ATR_RANGE) * len(TP_R_RANGE)} combinations...")
    
    results = []
    total = len(STOP_ATR_RANGE) * len(TP_R_RANGE)
    
    for i, (stop_atr, tp_r) in enumerate(itertools.product(STOP_ATR_RANGE, TP_R_RANGE)):
        result = run_orb_single(df, stop_atr, tp_r)
        results.append(result)
        
        if (i + 1) % 20 == 0:
            print(f"    Progress: {i+1}/{total}")
    
    print(f"\n[3] Grid search complete!")
    
    # Sort by profit factor
    valid_results = [r for r in results if r['trades'] >= 3 and r['profit_factor'] > 0]
    valid_results.sort(key=lambda x: x['profit_factor'], reverse=True)
    
    # Top 10 by Profit Factor
    print("\n" + "=" * 70)
    print("TOP 10 BY PROFIT FACTOR (min 3 trades)")
    print("=" * 70)
    print(f"{'Stop ATR':>10} | {'TP R':>6} | {'Trades':>7} | {'WR':>6} | {'PnL':>10} | {'PF':>6}")
    print("-" * 70)
    
    for r in valid_results[:10]:
        print(f"{r['stop_atr']:>10.2f} | {r['tp_r']:>6.1f} | {r['trades']:>7} | "
              f"{r['win_rate']:>5.1%} | ${r['total_pnl']:>9.2f} | {r['profit_factor']:>6.2f}")
    
    # Best configuration
    if valid_results:
        best = valid_results[0]
        print("\n" + "=" * 70)
        print("BEST CONFIGURATION")
        print("=" * 70)
        print(f"  Stop: {best['stop_atr']:.2f} ATR")
        print(f"  Take Profit: {best['tp_r']:.1f}R")
        print(f"  Trades: {best['trades']}")
        print(f"  Win Rate: {best['win_rate']:.1%}")
        print(f"  Profit Factor: {best['profit_factor']:.2f}")
        print(f"  Total PnL: ${best['total_pnl']:.2f}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ORB Grid Search")
    parser.add_argument("--days", type=int, default=7, help="Days to test")
    parser.add_argument("--save", action="store_true", help="Save all results to DB")
    
    args = parser.parse_args()
    
    results = run_gridsearch(days=args.days)
    
    if args.save and results:
        db = ExperimentDB()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for r in results:
            if r['trades'] > 0:
                run_id = f"orb_grid_{r['stop_atr']:.2f}atr_{r['tp_r']:.1f}r_{timestamp}"
                db.store_run(
                    run_id=run_id,
                    strategy="orb_gridsearch",
                    config={'stop_atr': r['stop_atr'], 'tp_r': r['tp_r']},
                    metrics={
                        'total_trades': r['trades'],
                        'wins': r['wins'],
                        'losses': r['losses'],
                        'win_rate': r['win_rate'],
                        'total_pnl': r['total_pnl'],
                        'profit_factor': r['profit_factor'],
                    }
                )
        
        print(f"\n[+] Saved {sum(1 for r in results if r['trades'] > 0)} configs to ExperimentDB")

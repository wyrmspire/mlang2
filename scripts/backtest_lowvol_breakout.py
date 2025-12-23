#!/usr/bin/env python3
"""
Low Volatility Breakout Strategy

Theory: Enter when volatility is dead (15m ATR at 5-day low).
Action: Trade the breakout when price moves.
Stop: DYNAMIC - 2x current candle's range (not lagging ATR)
Target: 2R

Hypothesis: Candle-range stop adapts faster than ATR.

Run:
    python scripts/backtest_lowvol_breakout.py --days 7
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Any

from src.storage import ExperimentDB


# Strategy parameters
ATR_PERIOD = 14
ATR_LOW_LOOKBACK = 5 * 24 * 4  # 5 days of 15m bars (approx)
STOP_CANDLE_MULT = 2.0         # Stop = 2x candle range
TP_R_MULT = 2.0                # 2R target
BREAKOUT_THRESHOLD = 0.3       # Price must move 0.3 ATR to trigger


def resample_to_15m(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1m data to 15m bars."""
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    
    resampled = df.resample('15min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    return resampled.reset_index()


def compute_atr_rolling_min(atr_series: pd.Series, window: int) -> pd.Series:
    """Compute rolling minimum of ATR."""
    return atr_series.rolling(window=window, min_periods=1).min()


def run_lowvol_breakout_strategy(days: int = 7, verbose: bool = True) -> Dict[str, Any]:
    """
    Run the Low Volatility Breakout strategy simulation.
    """
    print("=" * 60)
    print("LOW VOLATILITY BREAKOUT STRATEGY")
    print("=" * 60)
    print(f"Theory: Enter when ATR is at 5-day low (coiled market)")
    print(f"Stop: 2x current candle range (dynamic)")
    print(f"Target: {TP_R_MULT}R")
    print("=" * 60)
    
    # Load data
    actual_days = min(days, 7)
    end = datetime.now()
    start = end - timedelta(days=actual_days)
    
    print(f"\n[1] Loading {actual_days} days of ES data...")
    ticker = yf.Ticker("ES=F")
    df_1m = ticker.history(start=start, end=end, interval="1m")
    
    if df_1m is None or len(df_1m) == 0:
        print("ERROR: No data available")
        return {'trades': 0, 'win_rate': 0, 'total_pnl': 0}
    
    # Standardize
    df_1m.columns = [c.lower() for c in df_1m.columns]
    df_1m = df_1m.reset_index()
    df_1m['time'] = df_1m['Datetime'] if 'Datetime' in df_1m.columns else df_1m['datetime']
    
    print(f"    Loaded {len(df_1m)} 1m bars")
    
    # Resample to 15m
    print("\n[2] Resampling to 15m...")
    df = resample_to_15m(df_1m)
    print(f"    {len(df)} 15m bars")
    
    # Compute ATR
    print("\n[3] Computing 15m ATR and rolling minimum...")
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    df['atr'] = tr.rolling(window=ATR_PERIOD).mean()
    df['atr_min_5d'] = compute_atr_rolling_min(df['atr'], window=ATR_LOW_LOOKBACK)
    df['candle_range'] = df['high'] - df['low']
    
    # Run simulation
    print(f"\n[4] Scanning for low-ATR breakout entries...")
    
    trades = []
    active_trade = None
    lookback = max(ATR_PERIOD, 20)
    
    for i in range(lookback, len(df)):
        bar = df.iloc[i]
        current_time = bar['time']
        current_close = bar['close']
        current_high = bar['high']
        current_low = bar['low']
        current_atr = bar['atr']
        atr_min = bar['atr_min_5d']
        candle_range = bar['candle_range']
        
        if pd.isna(current_atr) or pd.isna(atr_min) or candle_range < 0.5:
            continue
        
        # Check active trade
        if active_trade is not None:
            if active_trade['direction'] == 'LONG':
                if current_low <= active_trade['stop']:
                    pnl = (active_trade['stop'] - active_trade['entry']) * 50
                    trades.append({
                        'entry_time': active_trade['entry_time'],
                        'exit_time': current_time,
                        'direction': 'LONG',
                        'entry': active_trade['entry'],
                        'exit': active_trade['stop'],
                        'pnl': pnl,
                        'result': 'LOSS'
                    })
                    if verbose:
                        print(f"  [LOSS] LONG @ {active_trade['entry']:.2f} -> SL @ {active_trade['stop']:.2f} = ${pnl:.2f}")
                    active_trade = None
                elif current_high >= active_trade['tp']:
                    pnl = (active_trade['tp'] - active_trade['entry']) * 50
                    trades.append({
                        'entry_time': active_trade['entry_time'],
                        'exit_time': current_time,
                        'direction': 'LONG',
                        'entry': active_trade['entry'],
                        'exit': active_trade['tp'],
                        'pnl': pnl,
                        'result': 'WIN'
                    })
                    if verbose:
                        print(f"  [WIN] LONG @ {active_trade['entry']:.2f} -> TP @ {active_trade['tp']:.2f} = ${pnl:.2f}")
                    active_trade = None
            else:  # SHORT
                if current_high >= active_trade['stop']:
                    pnl = (active_trade['entry'] - active_trade['stop']) * 50
                    trades.append({
                        'entry_time': active_trade['entry_time'],
                        'exit_time': current_time,
                        'direction': 'SHORT',
                        'entry': active_trade['entry'],
                        'exit': active_trade['stop'],
                        'pnl': pnl,
                        'result': 'LOSS'
                    })
                    if verbose:
                        print(f"  [LOSS] SHORT @ {active_trade['entry']:.2f} -> SL @ {active_trade['stop']:.2f} = ${pnl:.2f}")
                    active_trade = None
                elif current_low <= active_trade['tp']:
                    pnl = (active_trade['entry'] - active_trade['tp']) * 50
                    trades.append({
                        'entry_time': active_trade['entry_time'],
                        'exit_time': current_time,
                        'direction': 'SHORT',
                        'entry': active_trade['entry'],
                        'exit': active_trade['tp'],
                        'pnl': pnl,
                        'result': 'WIN'
                    })
                    if verbose:
                        print(f"  [WIN] SHORT @ {active_trade['entry']:.2f} -> TP @ {active_trade['tp']:.2f} = ${pnl:.2f}")
                    active_trade = None
            continue
        
        # Check for low ATR condition (ATR at or near 5-day low)
        is_low_vol = current_atr <= atr_min * 1.1  # Within 10% of 5-day low
        
        if not is_low_vol:
            continue
        
        # Check for breakout (price moves beyond previous bar)
        prev_high = df.iloc[i-1]['high']
        prev_low = df.iloc[i-1]['low']
        
        # DYNAMIC STOP: 2x current candle range
        stop_distance = candle_range * STOP_CANDLE_MULT
        
        # Breakout detection
        if current_high > prev_high:
            # Bullish breakout -> LONG
            entry = current_close
            stop = entry - stop_distance
            tp = entry + (stop_distance * TP_R_MULT)
            
            active_trade = {
                'entry_time': current_time,
                'entry': entry,
                'stop': stop,
                'tp': tp,
                'direction': 'LONG'
            }
            
            if verbose:
                print(f"  [TRIGGER] LONG @ {entry:.2f} (low ATR={current_atr:.2f}) Stop={stop:.2f} TP={tp:.2f}")
                
        elif current_low < prev_low:
            # Bearish breakout -> SHORT
            entry = current_close
            stop = entry + stop_distance
            tp = entry - (stop_distance * TP_R_MULT)
            
            active_trade = {
                'entry_time': current_time,
                'entry': entry,
                'stop': stop,
                'tp': tp,
                'direction': 'SHORT'
            }
            
            if verbose:
                print(f"  [TRIGGER] SHORT @ {entry:.2f} (low ATR={current_atr:.2f}) Stop={stop:.2f} TP={tp:.2f}")
    
    # Summary
    wins = sum(1 for t in trades if t['result'] == 'WIN')
    losses = len(trades) - wins
    total_pnl = sum(t['pnl'] for t in trades)
    win_rate = wins / len(trades) if trades else 0
    
    # Direction breakdown
    longs = [t for t in trades if t['direction'] == 'LONG']
    shorts = [t for t in trades if t['direction'] == 'SHORT']
    long_wr = sum(1 for t in longs if t['result'] == 'WIN') / len(longs) if longs else 0
    short_wr = sum(1 for t in shorts if t['result'] == 'WIN') / len(shorts) if shorts else 0
    
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
    print(f"\n  LONG trades: {len(longs)} @ {long_wr:.1%} WR")
    print(f"  SHORT trades: {len(shorts)} @ {short_wr:.1%} WR")
    
    return {
        'trades': len(trades),
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'long_trades': len(longs),
        'short_trades': len(shorts),
        'long_wr': long_wr,
        'short_wr': short_wr,
        'strategy': 'lowvol_breakout',
        'params': {
            'stop_candle_mult': STOP_CANDLE_MULT,
            'tp_r': TP_R_MULT,
            'atr_period': ATR_PERIOD,
        }
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Low Volatility Breakout Strategy")
    parser.add_argument("--days", type=int, default=7, help="Days to simulate")
    parser.add_argument("--save", action="store_true", help="Save to ExperimentDB")
    parser.add_argument("--quiet", action="store_true", help="Suppress trade details")
    
    args = parser.parse_args()
    
    results = run_lowvol_breakout_strategy(days=args.days, verbose=not args.quiet)
    
    if args.save and results['trades'] > 0:
        db = ExperimentDB()
        run_id = f"lowvol_breakout_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        db.store_run(
            run_id=run_id,
            strategy="lowvol_breakout",
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

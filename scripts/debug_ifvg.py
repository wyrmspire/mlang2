#!/usr/bin/env python3
"""
IFVG Debug Scanner

Minimal scanner for troubleshooting FVG detection.
Runs on 1 week, outputs detailed info for each detection.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
from typing import List, Dict, Any

from src.data.loader import load_continuous_contract
from src.data.resample import resample_all_timeframes

# ============================================================================
# CONFIGURATION
# ============================================================================

NY_TZ = ZoneInfo("America/New_York")
RESULTS_DIR = Path("results/ifvg_debug")

# Date range - 3 months for training data
START_DATE = "2025-03-24"  # Monday
DAYS = 90  # 3 months

# Trade window
TRADE_WINDOW_START = time(9, 30)
TRADE_WINDOW_END = time(15, 30)

# FVG Detection params
MIN_FVG_POINTS = 2.0  # Minimum gap size in points
SL_PADDING = 1.0  # Extra padding on stop loss
RISK_REWARD = 3.0  # 1:3 R:R
INVERSION_WINDOW = 12  # 1 hour on 5m (was 6 = 30min)
POINT_VALUE = 50  # ES point value


# ============================================================================
# OUTCOME SIMULATION (copied from backtest_ict_ifvg.py)
# ============================================================================

def simulate_limit_order_outcome(
    df_1m: pd.DataFrame,
    entry_time,
    entry_price: float,
    stop_price: float,
    tp_price: float,
    direction: str,
    max_bars: int = 240
) -> Dict[str, Any]:
    """Simulate limit order fill and outcome."""
    future = df_1m[df_1m['time'] > entry_time].head(max_bars)
    
    if future.empty:
        return {"filled": False, "outcome": "NO_DATA", "exit_price": 0, "bars_held": 0, "pnl_points": 0}
    
    filled = False
    fill_bar_idx = -1
    
    for i, (idx, bar) in enumerate(future.iterrows()):
        if direction == "LONG":
            if bar['low'] <= entry_price:
                filled = True
                fill_bar_idx = i
                break
        else:
            if bar['high'] >= entry_price:
                filled = True
                fill_bar_idx = i
                break
    
    if not filled:
        return {"filled": False, "outcome": "NOT_FILLED", "exit_price": 0, "bars_held": 0, "pnl_points": 0}
    
    post_fill = future.iloc[fill_bar_idx:]
    
    for i, (idx, bar) in enumerate(post_fill.iterrows()):
        if direction == "LONG":
            if bar['low'] <= stop_price:
                return {"filled": True, "outcome": "LOSS", "exit_price": stop_price, "bars_held": i + 1, "pnl_points": stop_price - entry_price}
            if bar['high'] >= tp_price:
                return {"filled": True, "outcome": "WIN", "exit_price": tp_price, "bars_held": i + 1, "pnl_points": tp_price - entry_price}
        else:
            if bar['high'] >= stop_price:
                return {"filled": True, "outcome": "LOSS", "exit_price": stop_price, "bars_held": i + 1, "pnl_points": entry_price - stop_price}
            if bar['low'] <= tp_price:
                return {"filled": True, "outcome": "WIN", "exit_price": tp_price, "bars_held": i + 1, "pnl_points": entry_price - tp_price}
    
    last_close = post_fill.iloc[-1]['close']
    pnl = (last_close - entry_price) if direction == "LONG" else (entry_price - last_close)
    return {"filled": True, "outcome": "TIMEOUT", "exit_price": last_close, "bars_held": len(post_fill), "pnl_points": pnl}


# ============================================================================
# SIMPLE FVG DETECTION
# ============================================================================

def find_fvgs_simple(df_5m: pd.DataFrame, min_gap: float = 2.0) -> List[Dict]:
    """
    Simple FVG detection without ATR scaling.
    
    Bullish FVG: candle[i+1].low > candle[i-1].high (gap up)
    Bearish FVG: candle[i-1].low > candle[i+1].high (gap down)
    """
    fvgs = []
    
    for i in range(1, len(df_5m) - 1):
        prev = df_5m.iloc[i-1]
        curr = df_5m.iloc[i]  # Impulse candle
        next_ = df_5m.iloc[i+1]
        
        bar_time = curr['time'] if 'time' in curr else df_5m.index[i]
        
        # Bullish FVG: gap up
        bullish_gap = next_['low'] - prev['high']
        if bullish_gap >= min_gap:
            fvgs.append({
                'type': 'BULLISH',
                'time': bar_time,
                'high': next_['low'],
                'low': prev['high'],
                'gap': bullish_gap,
                'midpoint': (next_['low'] + prev['high']) / 2,
                'bar_idx': i
            })
        
        # Bearish FVG: gap down
        bearish_gap = prev['low'] - next_['high']
        if bearish_gap >= min_gap:
            fvgs.append({
                'type': 'BEARISH',
                'time': bar_time,
                'high': prev['low'],
                'low': next_['high'],
                'gap': bearish_gap,
                'midpoint': (prev['low'] + next_['high']) / 2,
                'bar_idx': i
            })
    
    return fvgs


def find_inversions(fvgs: List[Dict], window_bars: int = INVERSION_WINDOW) -> List[Dict]:
    """
    Find inverted FVGs - where a new FVG is opposite to a recent one.
    """
    inversions = []
    
    for i, new_fvg in enumerate(fvgs):
        # Look back for opposite FVG
        for j in range(i-1, -1, -1):
            old_fvg = fvgs[j]
            
            # Must be opposite direction
            if old_fvg['type'] == new_fvg['type']:
                continue
            
            # Must be within window
            bar_diff = new_fvg['bar_idx'] - old_fvg['bar_idx']
            if bar_diff > window_bars:
                break  # Too far back
            
            inversions.append({
                'new_fvg': new_fvg,
                'old_fvg': old_fvg,
                'bar_diff': bar_diff
            })
            break  # Found one, stop looking
    
    return inversions


# ============================================================================
# MAIN
# ============================================================================

def analyze_day(day_date, df_1m, df_5m):
    """Analyze a single day and return all FVGs and inversions."""
    
    window_start = datetime.combine(day_date, TRADE_WINDOW_START).replace(tzinfo=NY_TZ)
    window_end = datetime.combine(day_date, TRADE_WINDOW_END).replace(tzinfo=NY_TZ)
    
    # Filter to trade window
    df_5m_day = df_5m[(df_5m['time'] >= window_start) & (df_5m['time'] <= window_end)].copy()
    
    if df_5m_day.empty:
        return {'date': str(day_date), 'fvgs': [], 'inversions': 0, 'trades': [], 'total_fvgs': 0, 'bullish_fvgs': 0, 'bearish_fvgs': 0, 'all_fvgs': []}
    
    # Find all FVGs
    fvgs = find_fvgs_simple(df_5m_day, min_gap=MIN_FVG_POINTS)
    
    # Find inversions
    inversions = find_inversions(fvgs, window_bars=INVERSION_WINDOW)
    
    # Create trade setups from inversions
    trades = []
    for inv in inversions:
        new_fvg = inv['new_fvg']
        
        if new_fvg['type'] == 'BEARISH':
            direction = 'SHORT'
            entry = new_fvg['midpoint']
            stop = new_fvg['high'] + SL_PADDING  # Add padding
            risk = stop - entry
            tp = entry - (RISK_REWARD * risk)  # 1:3 R:R
        else:
            direction = 'LONG'
            entry = new_fvg['midpoint']
            stop = new_fvg['low'] - SL_PADDING  # Add padding
            risk = entry - stop
            tp = entry + (RISK_REWARD * risk)  # 1:3 R:R
        
        trades.append({
            'time': str(new_fvg['time']),
            'direction': direction,
            'entry': round(entry, 2),
            'stop': round(stop, 2),
            'tp': round(tp, 2),
            'risk_pts': round(risk, 2),
            'fvg_gap': round(new_fvg['gap'], 2),
            'inversion_bars': inv['bar_diff']
        })
    
    return {
        'date': str(day_date),
        'total_fvgs': len(fvgs),
        'bullish_fvgs': len([f for f in fvgs if f['type'] == 'BULLISH']),
        'bearish_fvgs': len([f for f in fvgs if f['type'] == 'BEARISH']),
        'inversions': len(inversions),
        'trades': trades,
        'all_fvgs': [
            {
                'time': str(f['time']),
                'type': f['type'],
                'gap': round(f['gap'], 2),
                'high': round(f['high'], 2),
                'low': round(f['low'], 2)
            }
            for f in fvgs
        ]
    }


def get_ohlcv_window(df_1m: pd.DataFrame, entry_time, history_bars: int = 60, future_bars: int = 120) -> List[Dict]:
    """Extract raw OHLCV window for visualization."""
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


def main():
    print("=" * 60)
    print("IFVG Debug Scanner")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    df_1m = load_continuous_contract()
    htf = resample_all_timeframes(df_1m)
    df_5m = htf['5m']
    
    if 'time' not in df_5m.columns:
        df_5m = df_5m.reset_index()
    
    print(f"  {len(df_1m)} 1m bars, {len(df_5m)} 5m bars")
    
    # Date range
    start = pd.Timestamp(START_DATE).tz_localize(NY_TZ)
    
    # Process each day
    print(f"\nScanning {START_DATE} + {DAYS} days...")
    print(f"Minimum FVG gap: {MIN_FVG_POINTS} points")
    print()
    
    all_results = []
    all_records = []  # For UI visualization
    record_id = 0
    
    for d in range(DAYS):
        day = start + timedelta(days=d)
        day_date = day.date()
        
        result = analyze_day(day_date, df_1m, df_5m)
        all_results.append(result)
        
        print(f"{day_date}: {result['total_fvgs']} FVGs ({result['bullish_fvgs']}B {result['bearish_fvgs']}S) | {result['inversions']} inversions | {len(result['trades'])} trades")
        
        for t in result['trades']:
            print(f"    {t['time'][11:16]} {t['direction']} @ {t['entry']} (risk {t['risk_pts']}pts, gap {t['fvg_gap']}pts)")
            
            # Create standard record for UI
            record_id += 1
            entry_time = pd.Timestamp(t['time'])
            
            # Simulate the trade outcome
            outcome = simulate_limit_order_outcome(
                df_1m, entry_time, t['entry'], t['stop'], t['tp'], t['direction']
            )
            pnl_dollars = outcome['pnl_points'] * POINT_VALUE
            
            record = {
                "decision_id": f"debug_{record_id:04d}",
                "timestamp": t['time'],
                "scanner_id": "ifvg_debug",
                "window": {
                    "raw_ohlcv_1m": get_ohlcv_window(df_1m, entry_time)
                },
                "oco": {
                    "entry_price": t['entry'],
                    "stop_price": t['stop'],
                    "tp_price": t['tp'],
                    "direction": t['direction'],
                    "contracts": 1,
                    "order_type": "LIMIT"
                },
                "oco_results": {
                    "filled": outcome['filled'],
                    "outcome": outcome['outcome'],
                    "exit_price": outcome['exit_price'],
                    "bars_held": outcome['bars_held'],
                    "pnl_points": outcome['pnl_points'],
                    "pnl_dollars": pnl_dollars
                },
                "scanner_context": {
                    "fvg_gap": t['fvg_gap'],
                    "risk_pts": t['risk_pts'],
                    "inversion_bars": t['inversion_bars']
                }
            }
            all_records.append(record)
    
    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Debug output
    with open(RESULTS_DIR / "debug_output.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Standard records.jsonl for UI
    with open(RESULTS_DIR / "records.jsonl", 'w') as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")
    
    # Summary for UI - calculate stats from records
    wins = sum(1 for r in all_records if r['oco_results']['outcome'] == 'WIN')
    losses = sum(1 for r in all_records if r['oco_results']['outcome'] == 'LOSS')
    total_pnl = sum(r['oco_results']['pnl_dollars'] for r in all_records)
    
    summary = {
        "strategy": "IFVG Debug",
        "date_range": f"{START_DATE} + {DAYS} days",
        "trades": len(all_records),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / (wins + losses) if (wins + losses) > 0 else 0,
        "total_pnl": total_pnl,
        "params": {
            "min_fvg_points": MIN_FVG_POINTS,
            "inversion_window": INVERSION_WINDOW,
            "sl_padding": SL_PADDING,
            "risk_reward": RISK_REWARD
        }
    }
    with open(RESULTS_DIR / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {RESULTS_DIR}")
    print(f"  - debug_output.json (detailed analysis)")
    print(f"  - records.jsonl ({len(all_records)} trades for UI)")
    print(f"  - summary.json")


if __name__ == "__main__":
    main()

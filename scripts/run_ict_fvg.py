"""
ICT Fair Value Gap Strategy Backtest (Batch Mode)

Efficient backtesting: processes one day at a time instead of bar-by-bar.
Records are compatible with the standard visualization format.

Usage:
    python scripts/run_ict_fvg.py --start-date 2025-03-18 --weeks 4
"""

import sys
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from pathlib import Path
from zoneinfo import ZoneInfo

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import RESULTS_DIR, NY_TZ, POINT_VALUE, TICK_SIZE
from src.data.loader import load_continuous_contract
from src.data.resample import resample_all_timeframes
from src.features.indicators import calculate_atr
from src.features.levels import get_previous_day_levels


# Strategy parameters
MAX_RISK_DOLLARS = 300.0
MIN_RR = 1.5
ATR_BUFFER = 0.25

# Session times (NY timezone)
ASIAN_START = time(19, 0)
ASIAN_END = time(0, 0)
LONDON_START = time(2, 0)
LONDON_END = time(8, 30)
TRADE_START = time(9, 30)
TRADE_END = time(11, 30)


def make_serializable(obj):
    """Convert numpy types to Python native types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(i) for i in obj]
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def get_raw_ohlcv_window(df_1m, entry_idx, lookback=60, lookahead=30):
    """Get raw OHLCV window for chart visualization."""
    start_idx = max(0, entry_idx - lookback)
    end_idx = min(len(df_1m), entry_idx + lookahead + 1)
    
    window = df_1m.iloc[start_idx:end_idx]
    
    return [
        [float(r['open']), float(r['high']), float(r['low']), float(r['close']), float(r.get('volume', 0))]
        for _, r in window.iterrows()
    ]


def get_session_levels(df_1m, trade_date, tz=NY_TZ):
    """Get Asian and London session levels for a specific trading day."""
    prev_day = trade_date - timedelta(days=1)
    
    asian_start = datetime.combine(prev_day, ASIAN_START).replace(tzinfo=tz)
    asian_end = datetime.combine(trade_date, time(0, 0)).replace(tzinfo=tz)
    london_start = datetime.combine(trade_date, LONDON_START).replace(tzinfo=tz)
    london_end = datetime.combine(trade_date, LONDON_END).replace(tzinfo=tz)
    
    asian_mask = (df_1m['time_ny'] >= asian_start) & (df_1m['time_ny'] < asian_end)
    london_mask = (df_1m['time_ny'] >= london_start) & (df_1m['time_ny'] < london_end)
    
    asian_bars = df_1m.loc[asian_mask]
    london_bars = df_1m.loc[london_mask]
    
    return {
        'asian_high': float(asian_bars['high'].max()) if not asian_bars.empty else 0,
        'asian_low': float(asian_bars['low'].min()) if not asian_bars.empty else 0,
        'london_high': float(london_bars['high'].max()) if not london_bars.empty else 0,
        'london_low': float(london_bars['low'].min()) if not london_bars.empty else 0,
    }


def find_fvg(df_5m, direction, min_gap=0.5):
    """Find FVG in 5m window. Returns dict or None."""
    if len(df_5m) < 3:
        return None
    
    for i in range(1, len(df_5m) - 1):
        prev_bar = df_5m.iloc[i - 1]
        impulse = df_5m.iloc[i]
        next_bar = df_5m.iloc[i + 1]
        
        if direction == "LONG":
            gap = next_bar['low'] - prev_bar['high']
            if gap > min_gap:
                return {
                    'fvg_high': float(next_bar['low']),
                    'fvg_low': float(prev_bar['high']),
                    'fvg_midpoint': float((next_bar['low'] + prev_bar['high']) / 2),
                    'fvg_bar_idx': i,
                    'fvg_time': impulse['time']
                }
        else:
            gap = prev_bar['low'] - next_bar['high']
            if gap > min_gap:
                return {
                    'fvg_high': float(prev_bar['low']),
                    'fvg_low': float(next_bar['high']),
                    'fvg_midpoint': float((prev_bar['low'] + next_bar['high']) / 2),
                    'fvg_bar_idx': i,
                    'fvg_time': impulse['time']
                }
    return None


def check_retracement(df_5m_after, fvg, direction):
    """Check if price retraced 50% into FVG. Returns entry info or None."""
    if fvg is None or df_5m_after.empty:
        return None
    
    threshold = fvg['fvg_midpoint']
    
    for idx, bar in df_5m_after.iterrows():
        if direction == "LONG" and bar['low'] <= threshold:
            return {'entry_price': threshold, 'entry_time': bar['time'], 'entry_bar_idx': idx}
        elif direction == "SHORT" and bar['high'] >= threshold:
            return {'entry_price': threshold, 'entry_time': bar['time'], 'entry_bar_idx': idx}
    return None


def compute_outcome(df_1m_after, entry_price, stop_price, tp_price, direction, max_bars=200):
    """Compute trade outcome from 1m data."""
    if df_1m_after.empty:
        return {'outcome': 'NO_DATA', 'pnl_dollars': 0, 'bars_held': 0, 'exit_price': entry_price}
    
    for i, (_, bar) in enumerate(df_1m_after.iterrows()):
        if i >= max_bars:
            pnl = (bar['close'] - entry_price) * POINT_VALUE if direction == 'LONG' else (entry_price - bar['close']) * POINT_VALUE
            return {'outcome': 'TIMEOUT', 'pnl_dollars': pnl, 'bars_held': i, 'exit_price': float(bar['close'])}
        
        if direction == 'LONG':
            if bar['low'] <= stop_price:
                return {'outcome': 'LOSS', 'pnl_dollars': (stop_price - entry_price) * POINT_VALUE, 'bars_held': i, 'exit_price': stop_price}
            if bar['high'] >= tp_price:
                return {'outcome': 'WIN', 'pnl_dollars': (tp_price - entry_price) * POINT_VALUE, 'bars_held': i, 'exit_price': tp_price}
        else:
            if bar['high'] >= stop_price:
                return {'outcome': 'LOSS', 'pnl_dollars': (entry_price - stop_price) * POINT_VALUE, 'bars_held': i, 'exit_price': stop_price}
            if bar['low'] <= tp_price:
                return {'outcome': 'WIN', 'pnl_dollars': (entry_price - tp_price) * POINT_VALUE, 'bars_held': i, 'exit_price': tp_price}
    
    last_close = float(df_1m_after.iloc[-1]['close'])
    pnl = (last_close - entry_price) * POINT_VALUE if direction == 'LONG' else (entry_price - last_close) * POINT_VALUE
    return {'outcome': 'TIMEOUT', 'pnl_dollars': pnl, 'bars_held': len(df_1m_after), 'exit_price': last_close}


def analyze_day(df_1m, df_5m, trade_date, pdh, pdl, avg_atr, decision_idx, tz=NY_TZ):
    """Analyze one trading day. Returns record in standard format or None."""
    
    levels = get_session_levels(df_1m, trade_date, tz)
    if all(v == 0 for v in levels.values()):
        return None
    
    # Get trade window
    trade_start = datetime.combine(trade_date, TRADE_START).replace(tzinfo=tz)
    trade_end = datetime.combine(trade_date, TRADE_END).replace(tzinfo=tz)
    
    window_1m = df_1m[(df_1m['time_ny'] >= trade_start) & (df_1m['time_ny'] <= trade_end)]
    window_5m = df_5m[(df_5m['time_ny'] >= trade_start) & (df_5m['time_ny'] <= trade_end)]
    
    if window_1m.empty or window_5m.empty:
        return None
    
    # Check for setups (first one wins)
    setup = None
    
    # Asian low break -> LONG
    if levels['asian_low'] > 0 and window_1m['low'].min() < levels['asian_low']:
        break_mask = window_5m['low'] < levels['asian_low']
        if break_mask.any():
            after_break = window_5m.loc[break_mask.idxmax():]
            fvg = find_fvg(after_break, "LONG")
            if fvg:
                after_fvg = after_break.iloc[fvg['fvg_bar_idx']+1:]
                entry = check_retracement(after_fvg, fvg, "LONG")
                if entry:
                    setup = {'direction': 'LONG', 'level': 'asian_low', 'level_price': levels['asian_low'],
                             'wick': float(window_1m['low'].min()), 'fvg': fvg, 'entry': entry}
    
    # Asian high break -> SHORT
    if not setup and levels['asian_high'] > 0 and window_1m['high'].max() > levels['asian_high']:
        break_mask = window_5m['high'] > levels['asian_high']
        if break_mask.any():
            after_break = window_5m.loc[break_mask.idxmax():]
            fvg = find_fvg(after_break, "SHORT")
            if fvg:
                after_fvg = after_break.iloc[fvg['fvg_bar_idx']+1:]
                entry = check_retracement(after_fvg, fvg, "SHORT")
                if entry:
                    setup = {'direction': 'SHORT', 'level': 'asian_high', 'level_price': levels['asian_high'],
                             'wick': float(window_1m['high'].max()), 'fvg': fvg, 'entry': entry}
    
    # London low break -> LONG
    if not setup and levels['london_low'] > 0 and window_1m['low'].min() < levels['london_low']:
        break_mask = window_5m['low'] < levels['london_low']
        if break_mask.any():
            after_break = window_5m.loc[break_mask.idxmax():]
            fvg = find_fvg(after_break, "LONG")
            if fvg:
                after_fvg = after_break.iloc[fvg['fvg_bar_idx']+1:]
                entry = check_retracement(after_fvg, fvg, "LONG")
                if entry:
                    setup = {'direction': 'LONG', 'level': 'london_low', 'level_price': levels['london_low'],
                             'wick': float(window_1m['low'].min()), 'fvg': fvg, 'entry': entry}
    
    # London high break -> SHORT
    if not setup and levels['london_high'] > 0 and window_1m['high'].max() > levels['london_high']:
        break_mask = window_5m['high'] > levels['london_high']
        if break_mask.any():
            after_break = window_5m.loc[break_mask.idxmax():]
            fvg = find_fvg(after_break, "SHORT")
            if fvg:
                after_fvg = after_break.iloc[fvg['fvg_bar_idx']+1:]
                entry = check_retracement(after_fvg, fvg, "SHORT")
                if entry:
                    setup = {'direction': 'SHORT', 'level': 'london_high', 'level_price': levels['london_high'],
                             'wick': float(window_1m['high'].max()), 'fvg': fvg, 'entry': entry}
    
    if not setup:
        return None
    
    # Calculate levels
    entry_price = setup['entry']['entry_price']
    direction = setup['direction']
    
    if direction == 'LONG':
        stop_price = setup['wick'] - (ATR_BUFFER * avg_atr)
        risk = entry_price - stop_price
        tp_price = pdh if pdh > 0 and (pdh - entry_price) >= MIN_RR * risk else entry_price + MIN_RR * risk
    else:
        stop_price = setup['wick'] + (ATR_BUFFER * avg_atr)
        risk = stop_price - entry_price
        tp_price = pdl if pdl > 0 and (entry_price - pdl) >= MIN_RR * risk else entry_price - MIN_RR * risk
    
    if risk <= 0:
        return None
    
    # Position sizing
    contracts = max(1, int(MAX_RISK_DOLLARS // (risk * POINT_VALUE)))
    risk_dollars = contracts * risk * POINT_VALUE
    
    # Find entry bar index in main df
    entry_time = setup['entry']['entry_time']
    entry_mask = df_1m['time'] == entry_time
    if not entry_mask.any():
        # Find closest bar
        entry_mask = df_1m['time_ny'] >= entry_time
    entry_bar_idx = df_1m[entry_mask].index[0] if entry_mask.any() else 0
    
    # Compute outcome
    after_entry = df_1m.loc[entry_bar_idx + 1:]
    result = compute_outcome(after_entry, entry_price, stop_price, tp_price, direction)
    
    # Get OHLCV for visualization
    raw_ohlcv = get_raw_ohlcv_window(df_1m, entry_bar_idx)
    
    # Build record in STANDARD FORMAT
    return {
        'decision_id': f"ict_{decision_idx:04d}",
        'timestamp': entry_time.isoformat() if hasattr(entry_time, 'isoformat') else str(entry_time),
        'bar_idx': int(entry_bar_idx),
        'index': decision_idx,
        'scanner_id': 'ict_fvg_5m',
        'scanner_context': {
            'direction': direction,
            'level_broken': setup['level'],
            'level_price': setup['level_price'],
            'penetrating_wick': setup['wick'],
            'fvg_high': setup['fvg']['fvg_high'],
            'fvg_low': setup['fvg']['fvg_low'],
            'fvg_midpoint': setup['fvg']['fvg_midpoint'],
            'session_levels': levels,
        },
        'current_price': entry_price,
        'atr': avg_atr,
        'stop_price': stop_price,
        'stop_reason': f"ICT_WICK_{setup['level'].upper()}",
        'tp_price': tp_price,
        'tp_reason': 'PDH_PDL' if (direction == 'LONG' and tp_price == pdh) or (direction == 'SHORT' and tp_price == pdl) else 'MIN_RR',
        'risk_points': risk,
        'reward_points': abs(tp_price - entry_price),
        'contracts': contracts,
        'risk_dollars': risk_dollars,
        'window': {
            'raw_ohlcv_1m': raw_ohlcv,
            'x_context': [],
        },
        'oco': {
            'entry_price': entry_price,
            'stop_price': stop_price,
            'tp_price': tp_price,
            'direction': direction,
            'atr_at_creation': avg_atr,
            'max_bars': 200,
        },
        'oco_results': {
            'ict_fvg': {
                'outcome': result['outcome'],
                'pnl_dollars': result['pnl_dollars'] * contracts,
                'bars_held': result['bars_held'],
                'exit_price': result['exit_price'],
            }
        },
        'best_oco': 'ict_fvg',
        'best_pnl': result['pnl_dollars'] * contracts,
    }


def main():
    parser = argparse.ArgumentParser(description="ICT FVG Strategy Backtest")
    parser.add_argument("--start-date", type=str, default="2025-03-18")
    parser.add_argument("--weeks", type=int, default=4)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    
    start_date = pd.Timestamp(args.start_date).date()
    end_date = start_date + timedelta(weeks=args.weeks)
    out_dir = Path(args.out) if args.out else RESULTS_DIR / "ict_fvg"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("ICT FVG Strategy Backtest")
    print(f"Period: {start_date} to {end_date}")
    print("=" * 60)
    
    # Load data
    print("\n[1] Loading data...")
    df = load_continuous_contract()
    df = df[(df['time'] >= str(start_date)) & (df['time'] < str(end_date))].reset_index(drop=True)
    print(f"  {len(df)} 1m bars")
    
    if len(df) == 0:
        print("No data found")
        return
    
    df['time_ny'] = df['time'].dt.tz_convert(NY_TZ)
    
    # Resample
    print("\n[2] Resampling...")
    htf_data = resample_all_timeframes(df)
    df_5m = htf_data.get('5m')
    df_5m['time_ny'] = df_5m['time'].dt.tz_convert(NY_TZ)
    
    df_5m['atr'] = calculate_atr(df_5m, 14) if len(df_5m) > 14 else 10.0
    avg_atr = float(df_5m['atr'].dropna().mean()) if len(df_5m) > 14 else 10.0
    print(f"  Avg ATR: {avg_atr:.2f}")
    
    # Analyze days
    trading_days = sorted(set(df['time_ny'].dt.date))
    print(f"\n[3] Analyzing {len(trading_days)} days...")
    
    records = []
    decision_idx = 0
    
    for trade_date in trading_days:
        pd_levels = get_previous_day_levels(df, pd.Timestamp(datetime.combine(trade_date, time(12, 0)), tz=NY_TZ))
        pdh = pd_levels.get('pdh', 0) or 0
        pdl = pd_levels.get('pdl', 0) or 0
        
        record = analyze_day(df, df_5m, trade_date, pdh, pdl, avg_atr, decision_idx)
        
        if record:
            records.append(record)
            print(f"  {trade_date}: {record['scanner_context']['direction']} -> {record['oco_results']['ict_fvg']['outcome']} (${record['best_pnl']:.0f})")
            decision_idx += 1
    
    # Summary & save
    print("\n" + "=" * 60)
    if records:
        wins = [r for r in records if r['oco_results']['ict_fvg']['outcome'] == 'WIN']
        total_pnl = sum(r['best_pnl'] for r in records)
        
        print(f"Trades: {len(records)} | Wins: {len(wins)} | WR: {len(wins)/len(records):.0%} | PnL: ${total_pnl:.0f}")
        
        with open(out_dir / "records.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(make_serializable(r)) + "\n")
        
        with open(out_dir / "summary.json", "w") as f:
            json.dump({"trades": len(records), "win_rate": len(wins)/len(records), "total_pnl": total_pnl}, f, indent=2)
        
        print(f"Saved to {out_dir}")
    else:
        print("No trades found")


if __name__ == "__main__":
    main()

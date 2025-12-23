#!/usr/bin/env python3
"""
Relative Volume at Price (RVAP) Scanner

Theory: Volume confirms breakouts.
- Approaching PDH on LOW volume → FADE (SHORT)
- Approaching PDH on HIGH volume (2x avg) → BREAKOUT (LONG)

Custom indicator: RVAP = current volume / 20-bar average volume

Usage:
    python scripts/backtest_rvap.py --days 7
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Any, List

from src.features.indicators import calculate_atr
from src.storage import ExperimentDB


# =============================================================================
# RVAP Indicator
# =============================================================================

def compute_rvap(volume: pd.Series, period: int = 20) -> pd.Series:
    """
    Compute Relative Volume at Price.
    
    RVAP = current volume / SMA(volume, period)
    
    Values:
    - < 0.5: Very low volume (fade)
    - 0.5 - 1.5: Normal volume
    - > 2.0: High volume (breakout)
    """
    avg_vol = volume.rolling(window=period).mean()
    rvap = volume / avg_vol.replace(0, np.nan)
    return rvap.fillna(1.0)


def compute_session_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Get PDH (Previous Day High), PDL, PDC."""
    df = df.copy()
    df['date'] = df['time'].dt.date
    
    daily = df.groupby('date').agg({
        'high': 'max',
        'low': 'min',
        'close': 'last'
    })
    
    # Shift by 1 day for "previous day"
    daily['pdh'] = daily['high'].shift(1)
    daily['pdl'] = daily['low'].shift(1)
    daily['pdc'] = daily['close'].shift(1)
    
    # Map back
    df['pdh'] = df['date'].map(daily['pdh'])
    df['pdl'] = df['date'].map(daily['pdl'])
    df['pdc'] = df['date'].map(daily['pdc'])
    
    return df


# =============================================================================
# RVAP Scanner
# =============================================================================

def scan_rvap_signals(
    df: pd.DataFrame,
    high_vol_threshold: float = 2.0,
    low_vol_threshold: float = 0.7,
    approach_pct: float = 0.1,  # Within 0.1% of PDH/PDL
    lookforward: int = 20,
) -> List[Dict]:
    """
    Scan for RVAP signals near session levels.
    
    LONG (Breakout): Approach PDH on high volume (RVAP > 2.0)
    SHORT (Fade): Approach PDH on low volume (RVAP < 0.7)
    """
    df = df.copy()
    
    # Compute indicators
    df['rvap'] = compute_rvap(df['volume'])
    df = compute_session_levels(df)
    df['atr'] = calculate_atr(df, period=14).ffill().bfill()
    
    records = []
    
    for i in range(30, len(df) - lookforward):
        row = df.iloc[i]
        
        if pd.isna(row['pdh']) or pd.isna(row['rvap']):
            continue
        
        close = row['close']
        high = row['high']
        pdh = row['pdh']
        pdl = row['pdl']
        rvap = row['rvap']
        atr = row['atr'] if not pd.isna(row['atr']) else 2.0
        
        # Check if approaching PDH (within threshold)
        pdh_distance = abs(high - pdh) / pdh * 100
        approaching_pdh = pdh_distance < approach_pct and high >= pdh * 0.998
        
        # Check if approaching PDL
        pdl_distance = abs(row['low'] - pdl) / pdl * 100
        approaching_pdl = pdl_distance < approach_pct and row['low'] <= pdl * 1.002
        
        direction = None
        signal_type = None
        
        if approaching_pdh:
            if rvap >= high_vol_threshold:
                # High volume at resistance → BREAKOUT LONG
                direction = 'LONG'
                signal_type = 'breakout'
            elif rvap <= low_vol_threshold:
                # Low volume at resistance → FADE SHORT
                direction = 'SHORT'
                signal_type = 'fade'
        
        elif approaching_pdl:
            if rvap >= high_vol_threshold:
                # High volume at support → BREAKDOWN SHORT
                direction = 'SHORT'
                signal_type = 'breakdown'
            elif rvap <= low_vol_threshold:
                # Low volume at support → BOUNCE LONG
                direction = 'LONG'
                signal_type = 'bounce'
        
        if direction is None:
            continue
        
        # Check outcome
        entry_price = close
        future_bars = df.iloc[i+1:i+1+lookforward]
        
        if direction == 'LONG':
            target = entry_price + atr
            stop = entry_price - atr
            hit_target = (future_bars['high'] >= target).any()
            hit_stop = (future_bars['low'] <= stop).any()
        else:
            target = entry_price - atr
            stop = entry_price + atr
            hit_target = (future_bars['low'] <= target).any()
            hit_stop = (future_bars['high'] >= stop).any()
        
        if hit_target and hit_stop:
            if direction == 'LONG':
                target_idx = future_bars[future_bars['high'] >= target].index[0]
                stop_idx = future_bars[future_bars['low'] <= stop].index[0]
            else:
                target_idx = future_bars[future_bars['low'] <= target].index[0]
                stop_idx = future_bars[future_bars['high'] >= stop].index[0]
            outcome = 'WIN' if target_idx < stop_idx else 'LOSS'
        elif hit_target:
            outcome = 'WIN'
        else:
            outcome = 'LOSS'
        
        # Build record with window
        window_start = max(0, i - 60)
        ohlcv_window = df.iloc[window_start:i][['open', 'high', 'low', 'close', 'volume']].values.tolist()
        
        records.append({
            'time': str(row['time']),
            'direction': direction,
            'signal_type': signal_type,
            'label': outcome,
            'entry_price': entry_price,
            'rvap': round(rvap, 2),
            'pdh': pdh,
            'pdl': pdl,
            'atr': atr,
            'window': {
                'raw_ohlcv_1m': [
                    {'open': o, 'high': h, 'low': l, 'close': c, 'volume': v}
                    for o, h, l, c, v in ohlcv_window
                ]
            },
            'strategy': 'rvap',
        })
    
    return records


def run_rvap_scan(days: int = 7) -> Dict[str, Any]:
    """Run RVAP scanner."""
    
    print("=" * 60)
    print("RELATIVE VOLUME AT PRICE (RVAP) SCANNER")
    print("=" * 60)
    print("Logic:")
    print("  Approach PDH + HIGH volume (2x) → LONG (breakout)")
    print("  Approach PDH + LOW volume (<0.7x) → SHORT (fade)")
    print("=" * 60)
    
    # Load data
    actual_days = min(days, 7)
    end = datetime.now()
    start = end - timedelta(days=actual_days)
    
    print(f"\n[1] Loading {actual_days} days of ES data...")
    ticker = yf.Ticker("ES=F")
    df = ticker.history(start=start, end=end, interval="1m")
    
    if df is None or len(df) == 0:
        print("ERROR: No data!")
        return {'records': 0}
    
    df.columns = [c.lower() for c in df.columns]
    df = df.reset_index()
    df['time'] = pd.to_datetime(df['Datetime'] if 'Datetime' in df.columns else df['datetime'])
    
    print(f"    Loaded {len(df)} bars")
    
    # Scan
    print(f"\n[2] Scanning for RVAP signals...")
    records = scan_rvap_signals(df)
    
    if not records:
        print("    No signals found!")
        return {'records': 0}
    
    # Stats
    wins = sum(1 for r in records if r['label'] == 'WIN')
    
    # Breakdown by signal type
    breakouts = [r for r in records if r['signal_type'] == 'breakout']
    fades = [r for r in records if r['signal_type'] == 'fade']
    breakdowns = [r for r in records if r['signal_type'] == 'breakdown']
    bounces = [r for r in records if r['signal_type'] == 'bounce']
    
    print(f"\n    Total signals: {len(records)}")
    print(f"    WIN: {wins} | LOSS: {len(records) - wins}")
    print(f"    Win Rate: {wins/len(records):.1%}")
    
    print(f"\n    By signal type:")
    for name, group in [('Breakout (high vol)', breakouts), 
                        ('Fade (low vol)', fades),
                        ('Breakdown', breakdowns),
                        ('Bounce', bounces)]:
        if group:
            g_wins = sum(1 for r in group if r['label'] == 'WIN')
            print(f"      {name}: {len(group)} trades, {g_wins/len(group):.1%} WR")
    
    # Save
    output_dir = Path("results/rvap")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "records.jsonl"
    
    with open(output_path, 'w') as f:
        for rec in records:
            f.write(json.dumps(rec) + '\n')
    
    print(f"\n[3] Saved to {output_path}")
    
    # Store summary
    db = ExperimentDB()
    run_id = f"rvap_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    db.store_run(
        run_id=run_id,
        strategy="rvap",
        config={
            'high_vol_threshold': 2.0,
            'low_vol_threshold': 0.7,
        },
        metrics={
            'total_trades': len(records),
            'wins': wins,
            'losses': len(records) - wins,
            'win_rate': wins/len(records) if records else 0,
            'total_pnl': 0,
        }
    )
    print(f"    Stored: {run_id}")
    
    return {
        'records': len(records),
        'wins': wins,
        'win_rate': wins/len(records) if records else 0,
        'breakouts': len(breakouts),
        'fades': len(fades),
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RVAP Scanner")
    parser.add_argument("--days", type=int, default=7, help="Days to scan")
    
    args = parser.parse_args()
    
    results = run_rvap_scan(args.days)

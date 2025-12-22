#!/usr/bin/env python3
"""
Simple EMA Scanner

Generates clean, objective labels for model training.
Much simpler than ICT patterns - clear cause-effect relationships.

Strategies:
1. EMA Cross: 9 EMA crosses 21 EMA
2. EMA Bounce: Price touches 20 EMA and reverses
3. EMA Stack: All EMAs aligned (9 > 21 > 50 > 200)

Usage:
    python scripts/run_ema_scan.py --days 7 --strategy cross
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

from src.features.indicators import calculate_ema, calculate_atr
from src.storage import ExperimentDB


# =============================================================================
# EMA Strategies
# =============================================================================

def detect_ema_cross(
    df: pd.DataFrame,
    fast_period: int = 9,
    slow_period: int = 21,
    lookforward: int = 20,
) -> List[Dict]:
    """
    Detect EMA crossover signals.
    
    LONG: Fast EMA crosses above slow EMA
    SHORT: Fast EMA crosses below slow EMA
    Label: WIN if price moves in direction within lookforward bars
    """
    df = df.copy()
    df['ema_fast'] = calculate_ema(df['close'], fast_period)
    df['ema_slow'] = calculate_ema(df['close'], slow_period)
    
    # Calculate ATR for target sizing
    df['atr'] = calculate_atr(df, period=14).ffill()
    
    records = []
    
    for i in range(slow_period + 1, len(df) - lookforward):
        fast_prev = df['ema_fast'].iloc[i-1]
        fast_curr = df['ema_fast'].iloc[i]
        slow_prev = df['ema_slow'].iloc[i-1]
        slow_curr = df['ema_slow'].iloc[i]
        
        # Detect cross
        cross_up = fast_prev <= slow_prev and fast_curr > slow_curr
        cross_down = fast_prev >= slow_prev and fast_curr < slow_curr
        
        if not cross_up and not cross_down:
            continue
        
        direction = 'LONG' if cross_up else 'SHORT'
        entry_price = df['close'].iloc[i]
        atr = df['atr'].iloc[i] if not pd.isna(df['atr'].iloc[i]) else 2.0
        
        # Check outcome
        future_bars = df.iloc[i+1:i+1+lookforward]
        
        if direction == 'LONG':
            # Win if price goes up by 1 ATR before going down 1 ATR
            target = entry_price + atr
            stop = entry_price - atr
            hit_target = (future_bars['high'] >= target).any()
            hit_stop = (future_bars['low'] <= stop).any()
            
            if hit_target and hit_stop:
                # Both hit - check which first
                target_idx = future_bars[future_bars['high'] >= target].index[0]
                stop_idx = future_bars[future_bars['low'] <= stop].index[0]
                outcome = 'WIN' if target_idx < stop_idx else 'LOSS'
            elif hit_target:
                outcome = 'WIN'
            else:
                outcome = 'LOSS'
        else:
            # SHORT
            target = entry_price - atr
            stop = entry_price + atr
            hit_target = (future_bars['low'] <= target).any()
            hit_stop = (future_bars['high'] >= stop).any()
            
            if hit_target and hit_stop:
                target_idx = future_bars[future_bars['low'] <= target].index[0]
                stop_idx = future_bars[future_bars['high'] >= stop].index[0]
                outcome = 'WIN' if target_idx < stop_idx else 'LOSS'
            elif hit_target:
                outcome = 'WIN'
            else:
                outcome = 'LOSS'
        
        # Build record with window for model training
        window_start = max(0, i - 60)
        ohlcv_window = df.iloc[window_start:i][['open', 'high', 'low', 'close', 'volume']].values.tolist()
        
        records.append({
            'time': str(df['time'].iloc[i]),
            'direction': direction,
            'label': outcome,
            'entry_price': entry_price,
            'atr': atr,
            'window': {
                'raw_ohlcv_1m': [
                    {'open': o, 'high': h, 'low': l, 'close': c, 'volume': v}
                    for o, h, l, c, v in ohlcv_window
                ]
            },
            'strategy': 'ema_cross',
            'params': {'fast': fast_period, 'slow': slow_period},
        })
    
    return records


def detect_ema_bounce(
    df: pd.DataFrame,
    ema_period: int = 20,
    touch_threshold: float = 0.1,  # % distance from EMA to count as "touch"
    lookforward: int = 20,
) -> List[Dict]:
    """
    Detect EMA bounce signals.
    
    LONG: Price touches EMA from above and bounces up
    SHORT: Price touches EMA from below and bounces down
    """
    df = df.copy()
    df['ema'] = calculate_ema(df['close'], ema_period)
    df['atr'] = calculate_atr(df, period=14).ffill()
    
    records = []
    
    for i in range(ema_period + 5, len(df) - lookforward):
        ema = df['ema'].iloc[i]
        low = df['low'].iloc[i]
        high = df['high'].iloc[i]
        close = df['close'].iloc[i]
        atr = df['atr'].iloc[i] if not pd.isna(df['atr'].iloc[i]) else 2.0
        
        # Touch threshold in points
        threshold = ema * touch_threshold / 100
        
        # Check for touches
        touch_from_above = (low <= ema + threshold) and (close > ema) and (df['close'].iloc[i-1] > ema)
        touch_from_below = (high >= ema - threshold) and (close < ema) and (df['close'].iloc[i-1] < ema)
        
        if not touch_from_above and not touch_from_below:
            continue
        
        direction = 'LONG' if touch_from_above else 'SHORT'
        entry_price = close
        
        # Check outcome
        future_bars = df.iloc[i+1:i+1+lookforward]
        
        if direction == 'LONG':
            target = entry_price + atr
            stop = entry_price - atr
            hit_target = (future_bars['high'] >= target).any()
            hit_stop = (future_bars['low'] <= stop).any()
            
            if hit_target and hit_stop:
                target_idx = future_bars[future_bars['high'] >= target].index[0]
                stop_idx = future_bars[future_bars['low'] <= stop].index[0]
                outcome = 'WIN' if target_idx < stop_idx else 'LOSS'
            elif hit_target:
                outcome = 'WIN'
            else:
                outcome = 'LOSS'
        else:
            target = entry_price - atr
            stop = entry_price + atr
            hit_target = (future_bars['low'] <= target).any()
            hit_stop = (future_bars['high'] >= stop).any()
            
            if hit_target and hit_stop:
                target_idx = future_bars[future_bars['low'] <= target].index[0]
                stop_idx = future_bars[future_bars['high'] >= stop].index[0]
                outcome = 'WIN' if target_idx < stop_idx else 'LOSS'
            elif hit_target:
                outcome = 'WIN'
            else:
                outcome = 'LOSS'
        
        # Build record
        window_start = max(0, i - 60)
        ohlcv_window = df.iloc[window_start:i][['open', 'high', 'low', 'close', 'volume']].values.tolist()
        
        records.append({
            'time': str(df['time'].iloc[i]),
            'direction': direction,
            'label': outcome,
            'entry_price': entry_price,
            'atr': atr,
            'window': {
                'raw_ohlcv_1m': [
                    {'open': o, 'high': h, 'low': l, 'close': c, 'volume': v}
                    for o, h, l, c, v in ohlcv_window
                ]
            },
            'strategy': 'ema_bounce',
            'params': {'ema_period': ema_period},
        })
    
    return records


def detect_ema_stack(
    df: pd.DataFrame,
    periods: List[int] = [9, 21, 50, 200],
    lookforward: int = 20,
) -> List[Dict]:
    """
    Detect EMA stack alignment signals.
    
    LONG: 9 > 21 > 50 > 200 (bullish stack)
    SHORT: 9 < 21 < 50 < 200 (bearish stack)
    
    Entry when stack first forms.
    """
    df = df.copy()
    
    for p in periods:
        df[f'ema_{p}'] = calculate_ema(df['close'], p)
    
    df['atr'] = calculate_atr(df, period=14).ffill()
    
    records = []
    prev_bullish = False
    prev_bearish = False
    
    for i in range(max(periods) + 1, len(df) - lookforward):
        emas = [df[f'ema_{p}'].iloc[i] for p in periods]
        
        # Check stack alignment
        bullish_stack = all(emas[j] > emas[j+1] for j in range(len(emas)-1))
        bearish_stack = all(emas[j] < emas[j+1] for j in range(len(emas)-1))
        
        # Detect new stack formation
        new_bullish = bullish_stack and not prev_bullish
        new_bearish = bearish_stack and not prev_bearish
        
        prev_bullish = bullish_stack
        prev_bearish = bearish_stack
        
        if not new_bullish and not new_bearish:
            continue
        
        direction = 'LONG' if new_bullish else 'SHORT'
        entry_price = df['close'].iloc[i]
        atr = df['atr'].iloc[i] if not pd.isna(df['atr'].iloc[i]) else 2.0
        
        # Check outcome
        future_bars = df.iloc[i+1:i+1+lookforward]
        
        if direction == 'LONG':
            target = entry_price + atr * 1.5  # Bigger target for trend trades
            stop = entry_price - atr
        else:
            target = entry_price - atr * 1.5
            stop = entry_price + atr
        
        if direction == 'LONG':
            hit_target = (future_bars['high'] >= target).any()
            hit_stop = (future_bars['low'] <= stop).any()
        else:
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
        
        # Build record
        window_start = max(0, i - 60)
        ohlcv_window = df.iloc[window_start:i][['open', 'high', 'low', 'close', 'volume']].values.tolist()
        
        records.append({
            'time': str(df['time'].iloc[i]),
            'direction': direction,
            'label': outcome,
            'entry_price': entry_price,
            'atr': atr,
            'window': {
                'raw_ohlcv_1m': [
                    {'open': o, 'high': h, 'low': l, 'close': c, 'volume': v}
                    for o, h, l, c, v in ohlcv_window
                ]
            },
            'strategy': 'ema_stack',
            'params': {'periods': periods},
        })
    
    return records


# =============================================================================
# Main
# =============================================================================

def run_ema_scan(
    strategy: str = 'cross',
    days: int = 7,
    save: bool = True,
) -> Dict[str, Any]:
    """Run EMA scan and save records."""
    
    print("=" * 60)
    print(f"EMA SCANNER - {strategy.upper()}")
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
    
    # Run scan
    print(f"\n[2] Scanning for {strategy} signals...")
    
    if strategy == 'cross':
        records = detect_ema_cross(df)
    elif strategy == 'bounce':
        records = detect_ema_bounce(df)
    elif strategy == 'stack':
        records = detect_ema_stack(df)
    else:
        print(f"Unknown strategy: {strategy}")
        return {'records': 0}
    
    # Stats
    wins = sum(1 for r in records if r['label'] == 'WIN')
    longs = sum(1 for r in records if r['direction'] == 'LONG')
    shorts = len(records) - longs
    
    print(f"\n    Found {len(records)} signals")
    print(f"    LONG: {longs} | SHORT: {shorts}")
    print(f"    WIN: {wins} | LOSS: {len(records) - wins}")
    print(f"    Win Rate: {wins/len(records):.1%}" if records else "    No trades")
    
    # Save
    if save and records:
        output_dir = Path(f"results/ema_{strategy}")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "records.jsonl"
        
        with open(output_path, 'w') as f:
            for rec in records:
                f.write(json.dumps(rec) + '\n')
        
        print(f"\n[3] Saved to {output_path}")
        
        # Also store summary in DB
        db = ExperimentDB()
        run_id = f"ema_{strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        db.store_run(
            run_id=run_id,
            strategy=f"ema_{strategy}",
            config={'strategy': strategy, 'days': actual_days},
            metrics={
                'total_trades': len(records),
                'wins': wins,
                'losses': len(records) - wins,
                'win_rate': wins/len(records) if records else 0,
                'total_pnl': 0,
            }
        )
        print(f"    Stored summary: {run_id}")
    
    return {
        'records': len(records),
        'wins': wins,
        'losses': len(records) - wins,
        'win_rate': wins/len(records) if records else 0,
        'longs': longs,
        'shorts': shorts,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="EMA Scanner")
    parser.add_argument("--strategy", type=str, default="cross",
                        choices=['cross', 'bounce', 'stack'],
                        help="Strategy type")
    parser.add_argument("--days", type=int, default=7, help="Days to scan")
    parser.add_argument("--all", action="store_true", help="Run all strategies")
    
    args = parser.parse_args()
    
    if args.all:
        for strat in ['cross', 'bounce', 'stack']:
            print("\n")
            run_ema_scan(strategy=strat, days=args.days)
    else:
        run_ema_scan(strategy=args.strategy, days=args.days)

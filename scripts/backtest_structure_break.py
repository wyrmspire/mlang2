#!/usr/bin/env python3
"""
Run Structure Break Strategy Scan

Uses the modular StructureBreakTrigger on 15m timeframe.
Captures 20 5m candles for training data.

Usage:
    python scripts/backtest_structure_break.py --weeks 6 --out results/structure_break_scan
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from src.data.loader import load_continuous_contract
from src.data.resample import resample_all_timeframes
from src.sim.stepper import MarketStepper
from src.features.pipeline import compute_features, FeatureConfig
from src.features.indicators import calculate_atr
from src.policy.triggers.structure_break import StructureBreakTrigger
from src.policy.triggers.base import TriggerDirection
from src.labels.counterfactual import compute_smart_stop_counterfactual
from src.config import POINT_VALUE, TICK_SIZE


def get_raw_ohlcv_window(stepper, lookback=60, lookahead=30):
    """Get raw OHLCV for chart viz."""
    current = stepper.get_current_idx()
    start = max(0, current - lookback)
    end = min(len(stepper.df), current + lookahead)
    window = stepper.df.iloc[start:end]
    return window[['open', 'high', 'low', 'close', 'volume']].values.tolist()


def get_5m_window(df_5m: pd.DataFrame, current_time: pd.Timestamp, bars: int = 20):
    """Get the last N 5m bars before current time for training."""
    if df_5m is None or df_5m.empty:
        return []
    mask = df_5m['time'] < current_time
    recent = df_5m.loc[mask].tail(bars)
    return recent[['open', 'high', 'low', 'close', 'volume']].values.tolist()


def main():
    parser = argparse.ArgumentParser(description="Run Structure Break Scan")
    parser.add_argument("--start-date", type=str, default="2025-02-10",
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--weeks", type=int, default=6, help="Weeks of data")
    parser.add_argument("--out", type=str, default="results/structure_break_scan")
    parser.add_argument("--timeframe", type=str, default="15m", 
                        choices=["5m", "15m", "1h"], help="Timeframe for swing detection")
    
    # Trigger config (agent-adjustable)
    parser.add_argument("--swing-lookback", type=int, default=5)
    parser.add_argument("--rr-ratio", type=float, default=2.0)
    parser.add_argument("--atr-padding", type=float, default=0.5)
    parser.add_argument("--cooldown", type=int, default=3)
    parser.add_argument("--max-risk", type=float, default=300.0)
    
    args = parser.parse_args()
    
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data...")
    df = load_continuous_contract()
    
    start_dt = pd.Timestamp(args.start_date, tz='America/New_York')
    end_dt = start_dt + timedelta(weeks=args.weeks)
    
    df = df[(df['time'] >= start_dt) & (df['time'] < end_dt)].copy()
    print(f"  Data range: {df['time'].min()} to {df['time'].max()}")
    print(f"  Total 1m bars: {len(df)}")
    
    # Resample
    htf = resample_all_timeframes(df)
    df_5m = htf.get('5m')
    df_15m = htf.get('15m')
    df_1h = htf.get('1h')
    
    print(f"  5m bars: {len(df_5m) if df_5m is not None else 0}")
    print(f"  15m bars: {len(df_15m) if df_15m is not None else 0}")
    print(f"  1h bars: {len(df_1h) if df_1h is not None else 0}")
    
    # Select timeframe for scanning
    tf_map = {'5m': df_5m, '15m': df_15m, '1h': df_1h}
    df_tf = tf_map.get(args.timeframe, df_15m)
    print(f"\n  Using timeframe: {args.timeframe} ({len(df_tf) if df_tf is not None else 0} bars)")
    
    # Compute ATR on selected timeframe
    if df_tf is not None and len(df_tf) > 14:
        df_tf['atr'] = calculate_atr(df_tf, period=14)
        avg_atr = df_tf['atr'].dropna().mean()
    else:
        avg_atr = 10.0
    print(f"  Avg {args.timeframe} ATR: {avg_atr:.2f}")
    
    # Create trigger with agent-configurable params
    trigger = StructureBreakTrigger(
        swing_lookback=args.swing_lookback,
        rr_ratio=args.rr_ratio,
        atr_padding=args.atr_padding,
        cooldown_bars=args.cooldown
    )
    
    print(f"\nTrigger Config: {trigger.params}")
    print("=" * 60)
    
    # Feature config
    feature_config = FeatureConfig(
        lookback_1m=60,
        lookback_5m=24,
        lookback_15m=8,
        price_norm='zscore'
    )
    
    # Initialize stepper with selected timeframe
    stepper = MarketStepper(df_tf if df_tf is not None else df)
    
    records = []
    decision_idx = 0
    
    while True:
        step = stepper.step()
        if step.is_done:
            break
        
        # Compute features on 15m
        features = compute_features(stepper, feature_config, df_5m=df_5m, df_15m=df_15m)
        
        # Slice df_15m to only include data up to current bar (prevent look-ahead)
        df_15m_slice = df_15m.iloc[:step.bar_idx + 1] if step.bar_idx < len(df_15m) else df_15m
        
        # Check trigger
        result = trigger.check(features, df_15m=df_15m_slice)
        
        if result.triggered:
            ctx = result.context
            direction = result.direction.value
            entry_price = ctx['entry_price']
            stop_price = ctx['stop_price']
            tp_price = ctx['tp_price']
            
            # Position sizing
            risk_points = abs(entry_price - stop_price)
            risk_per_contract = risk_points * POINT_VALUE
            contracts = max(1, int(args.max_risk // risk_per_contract)) if risk_per_contract > 0 else 1
            actual_risk = contracts * risk_per_contract
            
            print(f"  [{decision_idx:3d}] {direction:5s} @ {entry_price:.2f} | "
                  f"Stop: {stop_price:.2f} | TP: {tp_price:.2f} ({args.rr_ratio}R) | "
                  f"Size: {contracts} | Risk: ${actual_risk:.0f} | "
                  f"Broken: {ctx['broken_level']:.2f} @ {step.bar['time']}")
            
            atr = features.atr if features.atr > 0 else avg_atr
            
            # Find exact 1m bar index by timestamp (more accurate than step.bar_idx * 15)
            from src.labels.counterfactual import find_bar_idx_by_time
            entry_1m_idx = find_bar_idx_by_time(df, features.timestamp)
            
            # Counterfactual simulation
            cf = compute_smart_stop_counterfactual(
                df=df,
                entry_idx=entry_1m_idx,
                direction=direction,
                stop_price=stop_price,
                tp_multiple=args.rr_ratio,
                atr=atr,
                oco_name="structure_break"
            )
            
            total_pnl = cf.pnl_dollars * contracts
            
            # Get training data: 20 5m bars before entry
            x_5m_bars = get_5m_window(df_5m, features.timestamp, bars=20)
            
            # Raw OHLCV for viz
            raw_ohlcv = get_raw_ohlcv_window(stepper, lookback=60, lookahead=30)
            
            record = {
                'decision_id': f"sb_{decision_idx:04d}",
                'timestamp': features.timestamp.isoformat(),
                'bar_idx': step.bar_idx,
                'index': decision_idx,
                'scanner_id': trigger.trigger_id,
                'scanner_context': {
                    'direction': direction,
                    'entry_price': entry_price,
                    'stop_price': stop_price,
                    'tp_price': tp_price,
                    'broken_level': ctx['broken_level'],
                    'new_extreme': ctx['new_extreme'],
                    'risk_points': ctx['risk_points'],
                    'contracts': contracts,
                    'risk_dollars': actual_risk,
                    'reward_dollars': actual_risk * args.rr_ratio,
                    'rr_ratio': args.rr_ratio
                },
                'current_price': entry_price,
                'atr': atr,
                'contracts': contracts,
                'risk_dollars': actual_risk,
                'window': {
                    'x_price_5m': x_5m_bars,  # 20 5m bars for training
                    'raw_ohlcv_15m': raw_ohlcv,
                    'x_context': features.x_context.tolist() if features.x_context is not None else [],
                },
                'oco': {
                    'entry_price': entry_price,
                    'stop_price': stop_price,
                    'tp_price': tp_price,
                    'direction': direction,
                    'atr_at_creation': atr,
                    'max_bars': 180  # ~45 hours on 15m
                },
                'oco_results': {
                    'structure_break': {
                        'outcome': cf.outcome,
                        'pnl_dollars': float(total_pnl),
                        'bars_held': int(cf.bars_held),
                        'exit_price': float(cf.exit_price),
                    }
                },
                'best_oco': 'structure_break',
                'best_pnl': total_pnl
            }
            
            records.append(record)
            decision_idx += 1
    
    print("=" * 60)
    print(f"Done. Triggers: {len(records)}")
    
    # Write records
    records_file = out_dir / "records.jsonl"
    with open(records_file, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')
    
    # Summary
    if records:
        wins = sum(1 for r in records if r['best_pnl'] > 0)
        total_pnl = sum(r['best_pnl'] for r in records)
        summary = {
            'strategy': 'structure_break',
            'trigger_config': trigger.params,
            'date_range': f"{start_dt.date()} to {end_dt.date()}",
            'weeks': args.weeks,
            'total_triggers': len(records),
            'wins': wins,
            'losses': len(records) - wins,
            'win_rate': wins / len(records) if records else 0,
            'total_pnl': total_pnl,
            'avg_pnl': total_pnl / len(records) if records else 0
        }
    else:
        summary = {'strategy': 'structure_break', 'total_triggers': 0}
    
    with open(out_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nOutput: {out_dir}")
    if records:
        print(f"  Win Rate: {summary['win_rate']:.1%}")
        print(f"  Total PnL: ${summary['total_pnl']:.2f}")


if __name__ == "__main__":
    main()

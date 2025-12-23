#!/usr/bin/env python
"""
Swing Breakout Scan Simulation
Triggers trades on 15m swing structure breakouts with structure-based stops/TPs.
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import timedelta
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_continuous_contract
from src.data.resample import resample_all_timeframes
from src.sim.stepper import MarketStepper
from src.features.pipeline import compute_features, FeatureConfig
from src.features.indicators import calculate_atr
from src.policy.library.swing_breakout import SwingBreakoutScanner
from src.labels.counterfactual import compute_smart_stop_counterfactual
from src.config import RESULTS_DIR


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
    return obj


def get_raw_ohlcv_window(stepper: MarketStepper, lookback: int = 60, lookahead: int = 20):
    """Get raw OHLCV window including future for visualization."""
    df = stepper.df
    current_idx = stepper.current_idx
    
    # History
    start_idx = max(0, current_idx - lookback)
    history = df.iloc[start_idx:current_idx + 1]
    
    # Future
    end_idx = min(len(df), current_idx + lookahead + 1)
    future = df.iloc[current_idx + 1:end_idx]
    
    combined = pd.concat([history, future])
    
    return [
        [float(r['open']), float(r['high']), float(r['low']), float(r['close']), float(r['volume'])]
        for _, r in combined.iterrows()
    ]


def main():
    parser = argparse.ArgumentParser(description="Run Swing Breakout Scan")
    parser.add_argument("--start-date", type=str, default="2025-03-17", help="Start date")
    parser.add_argument("--weeks", type=int, default=1, help="Number of weeks to scan")
    parser.add_argument("--lookback-bars", type=int, default=10, help="15m bars to look back for swings")
    parser.add_argument("--min-atr-distance", type=float, default=0.3, help="Min breakout distance in ATR")
    parser.add_argument("--out", type=str, default=None, help="Output directory")
    
    args = parser.parse_args()
    
    start_date = pd.Timestamp(args.start_date)
    end_date = start_date + timedelta(weeks=args.weeks)
    
    out_dir = Path(args.out) if args.out else RESULTS_DIR / "swing_breakout_scan"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Swing Breakout Scan (15m Timeframe)")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Strategy: Breakout of {args.lookback_bars}-bar swing structure")
    print("=" * 60)
    
    # 1. Load Data
    print("\n[1] Loading data...")
    df = load_continuous_contract()
    df = df[(df['time'] >= str(start_date)) & (df['time'] < str(end_date))].reset_index(drop=True)
    print(f"Loaded {len(df)} 1m bars")
    
    if len(df) < 500:
        print("Warning: Not enough data for meaningful scan")
        return
    
    # 2. Resample
    print("\n[2] Resampling...")
    htf_data = resample_all_timeframes(df)
    df_5m = htf_data.get('5m')
    df_15m = htf_data.get('15m')
    
    if df_15m is None or len(df_15m) < 20:
        print("Error: Not enough 15m bars")
        return
    
    # 3. ATR
    df_5m['atr'] = calculate_atr(df_5m, 14)
    df_15m['atr'] = calculate_atr(df_15m, 14)
    avg_atr = df_5m['atr'].dropna().mean()
    print(f"Avg ATR (5m): {avg_atr:.2f}")
    
    # 4. Scanner
    scanner = SwingBreakoutScanner(
        lookback_bars=args.lookback_bars,
        min_atr_distance=args.min_atr_distance,
        cooldown_bars=15
    )
    stepper = MarketStepper(df, start_idx=200, end_idx=len(df)-200)
    feature_config = FeatureConfig(lookback_1m=60)
    
    records = []
    decision_idx = 0
    
    print("\n[3] Running Scan...")
    while True:
        step = stepper.step()
        if step.is_done:
            break
        
        features = compute_features(stepper, feature_config, df_5m=df_5m, df_15m=df_15m)
        
        # Pass df_15m to scanner for swing computation
        scan = scanner.scan(features.market_state, features, df_15m=df_15m)
        
        if scan.triggered:
            ctx = scan.context
            direction = ctx['direction']
            entry_price = ctx['entry_price']
            stop_price = ctx['stop_price']
            tp_price = ctx['tp_price']
            
            print(f"  [{decision_idx:3d}] {direction:5s} @ {entry_price:.2f} | "
                  f"Stop: {stop_price:.2f} | TP: {tp_price:.2f} | "
                  f"Swing H/L: {ctx['swing_high']:.2f}/{ctx['swing_low']:.2f}")
            
            atr = features.atr if features.atr > 0 else avg_atr
            
            # Compute Counterfactual
            cf = compute_smart_stop_counterfactual(
                df=df,
                entry_idx=step.bar_idx,
                direction=direction,
                stop_price=stop_price,
                tp_multiple=1.5,  # Nominal (actual TP is structure-based)
                atr=atr,
                oco_name="swing_breakout"
            )
            
            # Raw OHLCV for chart
            raw_ohlcv = get_raw_ohlcv_window(stepper, lookback=60, lookahead=30)
            
            record = {
                'decision_id': f"swb_{decision_idx:04d}",
                'timestamp': features.timestamp.isoformat(),
                'bar_idx': step.bar_idx,
                'index': decision_idx,
                'scanner_id': scanner.scanner_id,
                'scanner_context': ctx,
                'current_price': entry_price,
                'atr': atr,
                'stop_price': stop_price,
                'stop_reason': 'SWING_STRUCTURE',
                'tp_price': tp_price,
                'tp_reason': 'NEXT_STRUCTURE',
                'risk_points': ctx['risk_points'],
                'reward_points': ctx['reward_points'],
                'window': {
                    'x_price_1m': features.x_price_1m.tolist() if features.x_price_1m is not None else [],
                    'raw_ohlcv_1m': raw_ohlcv,
                    'x_context': features.x_context.tolist() if features.x_context is not None else [],
                },
                'oco': {
                    'entry_price': entry_price,
                    'stop_price': stop_price,
                    'tp_price': tp_price,
                    'direction': direction,
                    'atr_at_creation': atr,
                    'max_bars': 100
                },
                'oco_results': {
                    'swing_breakout': {
                        'outcome': cf.outcome,
                        'pnl_dollars': cf.pnl_dollars,
                    }
                },
                'best_oco': 'swing_breakout',
                'best_pnl': cf.pnl_dollars
            }
            records.append(record)
            decision_idx += 1
    
    # Write output
    print(f"\n[4] Writing {len(records)} records to {out_dir}")
    output_path = out_dir / "swing_breakout_records.jsonl"
    with open(output_path, 'w') as f:
        for r in records:
            f.write(json.dumps(make_serializable(r)) + '\n')
    
    # Summary
    if records:
        wins = sum(1 for r in records if r['best_pnl'] > 0)
        losses = sum(1 for r in records if r['best_pnl'] < 0)
        total_pnl = sum(r['best_pnl'] for r in records)
        
        summary = {
            'total_triggers': len(records),
            'wins': wins,
            'losses': losses,
            'win_rate': wins / len(records) if records else 0,
            'total_pnl': total_pnl,
            'strategy': f'SwingBreakout_{args.lookback_bars}bar',
            'lookback_bars': args.lookback_bars,
            'min_atr_distance': args.min_atr_distance,
        }
    else:
        summary = {
            'total_triggers': 0,
            'strategy': f'SwingBreakout_{args.lookback_bars}bar',
        }
    
    with open(out_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Triggers: {summary.get('total_triggers', 0)}")
    if records:
        print(f"  Win Rate: {summary.get('win_rate', 0):.1%}")
        print(f"  Total PnL: ${summary.get('total_pnl', 0):.2f}")
    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()

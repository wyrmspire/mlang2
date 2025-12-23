#!/usr/bin/env python
"""
Simple Time Scan Simulation
Triggers trades at specific times (10:00 NY) with fixed 15m hold or simple bracket.
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
from src.policy.library.simple_time import SimpleTimeScanner
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
    parser = argparse.ArgumentParser(description="Run Simple Time Scan")
    parser.add_argument("--start-date", type=str, default="2025-03-17", help="Start date")
    parser.add_argument("--weeks", type=int, default=1, help="Number of weeks")
    parser.add_argument("--out", type=str, default=None, help="Output directory")
    
    args = parser.parse_args()
    
    start_date = pd.Timestamp(args.start_date)
    end_date = start_date + timedelta(weeks=args.weeks)
    
    out_dir = Path(args.out) if args.out else RESULTS_DIR / "simple_time_scan"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Simple Time Scan (15m Timeframe)")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Strategy: 10:00 AM Entry, Fixed Stop/TP")
    print("=" * 60)
    
    # 1. Load Data
    print("\n[1] Loading data...")
    df = load_continuous_contract()
    df = df[(df['time'] >= str(start_date)) & (df['time'] < str(end_date))].reset_index(drop=True)
    print(f"Loaded {len(df)} 1m bars")
    
    # 2. Resample
    print("\n[2] Resampling...")
    htf_data = resample_all_timeframes(df)
    df_5m = htf_data.get('5m')
    df_15m = htf_data.get('15m')
    
    # 3. ATR
    df_5m['atr'] = calculate_atr(df_5m, 14)
    avg_atr = df_5m['atr'].dropna().mean()
    print(f"Avg ATR: {avg_atr:.2f}")
    
    # 4. Scanner
    scanner = SimpleTimeScanner(hour=10, minute=0, momentum_minutes=15)
    stepper = MarketStepper(df, start_idx=200, end_idx=len(df)-200)
    feature_config = FeatureConfig(lookback_1m=30)
    
    records = []
    decision_idx = 0
    
    print("\n[3] Running Scan...")
    while True:
        step = stepper.step()
        if step.is_done:
            break
            
        features = compute_features(stepper, feature_config, df_5m=df_5m, df_15m=df_15m)
        scan = scanner.scan(features.market_state, features)
        
        if scan.triggered:
            print(f"  Triggered {scan.context['direction']} at {features.timestamp}")
            
            direction = scan.context['direction']
            entry_price = features.current_price
            atr = features.atr if features.atr > 0 else avg_atr
            
            # Simple Fixed Strategy for visibility
            # Stop = 2 * ATR
            # TP = 3 * ATR
            risk_dist = 2.0 * atr
            reward_dist = 3.0 * atr
            
            if direction == "LONG":
                stop_price = entry_price - risk_dist
                tp_price = entry_price + reward_dist
            else:
                stop_price = entry_price + risk_dist
                tp_price = entry_price - reward_dist
                
            # Compute Counterfactual
            cf = compute_smart_stop_counterfactual(
                df=df,
                entry_idx=step.bar_idx,
                direction=direction,
                stop_price=stop_price,
                tp_multiple=1.5, # Nominal
                atr=atr,
                oco_name="simple_time"
            )
            
            # Raw OHLCV for chart
            raw_ohlcv = get_raw_ohlcv_window(stepper, lookback=60, lookahead=30)
            
            record = {
                'decision_id': f"st_{decision_idx:04d}",
                'timestamp': features.timestamp.isoformat(),
                'bar_idx': step.bar_idx,
                'index': decision_idx,
                'scanner_id': 'simple_time',
                'scanner_context': scan.context,
                'current_price': entry_price,
                'atr': atr,
                'stop_price': stop_price,
                'stop_reason': 'FIXED_2ATR',
                'risk_points': risk_dist,
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
                    'simple_time': {
                        'outcome': cf.outcome,
                        'pnl_dollars': cf.pnl_dollars,
                    }
                },
                'best_oco': 'simple_time',
                'best_pnl': cf.pnl_dollars
            }
            records.append(record)
            decision_idx += 1
            
    # Write output
    print(f"\n[4] Writing {len(records)} records to {out_dir}")
    output_path = out_dir / "simple_time_scan_records.jsonl"
    with open(output_path, 'w') as f:
        for r in records:
            f.write(json.dumps(make_serializable(r)) + '\n')
            
    # Summary
    summary = {
        'total_triggers': len(records),
        'strategy': 'SimpleTime_10am',
    }
    with open(out_dir / "summary.json", 'w') as f:
        json.dump(summary, f)
        
    print("Done.")

if __name__ == "__main__":
    main()

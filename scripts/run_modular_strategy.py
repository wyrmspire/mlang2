"""
Generic Modular Strategy Runner

Runs a backtest using a modular Trigger and Bracket configuration.
"""

import os
import sys
import argparse
import json
import pandas as pd
from datetime import timedelta
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import RESULTS_DIR, NY_TZ
from src.data.loader import load_continuous_contract
from src.data.resample import resample_all_timeframes
from src.features.indicators import calculate_atr
from src.sim.stepper import MarketStepper
from src.features.pipeline import compute_features, FeatureConfig
from src.policy.modular_scanner import ModularScanner
from src.policy.brackets import bracket_from_dict
from src.labels.counterfactual import compute_smart_stop_counterfactual


def get_raw_ohlcv_window(stepper, lookback=60, lookahead=30):
    """Get raw OHLCV for chart viz."""
    current = stepper.get_current_idx()
    start = max(0, current - lookback)
    end = min(len(stepper.df), current + lookahead)
    window = stepper.df.iloc[start:end]
    return window[['open', 'high', 'low', 'close', 'volume']].values.tolist()


def main():
    parser = argparse.ArgumentParser(description="Run Modular Strategy Scan")
    parser.add_argument("--config", type=str, required=True, help="JSON configuration")
    parser.add_argument("--start-date", type=str, default="2025-03-17", help="Start date")
    parser.add_argument("--weeks", type=int, default=1, help="Number of weeks")
    parser.add_argument("--timeframe", type=str, default="1m", 
                        choices=["1m", "5m", "15m", "1h"], help="Timeframe for scanning")
    parser.add_argument("--out", type=str, default=None, help="Output directory")
    
    args = parser.parse_args()
    
    # Parse config
    try:
        config = json.loads(args.config)
    except json.JSONDecodeError:
        # Try finding as file
        if os.path.exists(args.config):
            with open(args.config) as f:
                config = json.load(f)
        else:
            raise ValueError("Invalid config JSON or file path")

    trigger_config = config.get("trigger")
    bracket_config = config.get("bracket")
    
    if not trigger_config or not bracket_config:
        raise ValueError("Config must contain 'trigger' and 'bracket' sections")

    start_date = pd.Timestamp(args.start_date)
    end_date = start_date + timedelta(weeks=args.weeks)
    
    out_dir = Path(args.out) if args.out else RESULTS_DIR / "modular_scan"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Modular Strategy Scan")
    print(f"Trigger: {trigger_config.get('type')}")
    print(f"Bracket: {bracket_config.get('type')}")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print("=" * 60)
    
    # 1. Load Data
    print("\n[1] Loading data...")
    df = load_continuous_contract()
    df = df[(df['time'] >= str(start_date)) & (df['time'] < str(end_date))].reset_index(drop=True)
    print(f"Loaded {len(df)} 1m bars")
    
    # 2. Resample & Indicators
    print("\n[2] Computing indicators...")
    htf_data = resample_all_timeframes(df)
    df_5m = htf_data.get('5m')
    df_15m = htf_data.get('15m')
    df_1h = htf_data.get('1h')
    
    # Select timeframe for scanning
    tf_map = {'1m': df, '5m': df_5m, '15m': df_15m, '1h': df_1h}
    df_tf = tf_map.get(args.timeframe, df)
    
    if df_tf is not None and len(df_tf) > 14:
        df_tf['atr'] = calculate_atr(df_tf, 14)
        avg_atr = df_tf['atr'].dropna().mean()
    else:
        avg_atr = 10.0
    print(f"  Using {args.timeframe}: {len(df_tf) if df_tf is not None else 0} bars, avg ATR: {avg_atr:.2f}")
    
    # 3. Initialize Components
    scanner = ModularScanner(trigger_config)
    bracket = bracket_from_dict(bracket_config)
    
    stepper = MarketStepper(df_tf if df_tf is not None else df, start_idx=50, end_idx=len(df_tf)-50 if df_tf is not None else len(df)-200)
    feature_config = FeatureConfig(lookback_1m=30)
    
    records = []
    decision_idx = 0
    
    # 4. Run Scan
    print("\n[3] Running Scan...")
    while True:
        step = stepper.step()
        if step.is_done:
            break
            
        features = compute_features(stepper, feature_config, df_5m=df_5m, df_15m=df_15m)
        
        # Slice df_tf to current bar position (causal)
        df_tf_slice = df_tf.iloc[:step.bar_idx + 1] if step.bar_idx < len(df_tf) else df_tf
        
        # Pass HTF data to trigger via kwargs
        scan = scanner.scan(features.market_state, features, df_15m=df_tf_slice)
        
        if scan.triggered:
            direction = scan.context['direction']
            entry_price = features.current_price
            atr = features.atr if features.atr > 0 else avg_atr
            
            # Compute levels using modular bracket
            levels = bracket.compute(entry_price, direction, atr)
            
            print(f"  Triggered {direction} at {features.timestamp} | SL: {levels.stop_price:.2f} TP: {levels.tp_price:.2f}")

            # Compute Outcomes
            cf = compute_smart_stop_counterfactual(
                df=df,
                entry_idx=step.bar_idx,
                direction=direction,
                stop_price=levels.stop_price,
                tp_multiple=levels.r_multiple,
                atr=atr,
                oco_name="modular"
            )
            
            # Raw OHLCV for chart
            raw_ohlcv = get_raw_ohlcv_window(stepper, lookback=60, lookahead=30)
            
            record = {
                'decision_id': f"mod_{decision_idx:04d}",
                'timestamp': features.timestamp.isoformat(),
                'bar_idx': step.bar_idx,
                'index': decision_idx,
                'scanner_id': scanner.scanner_id,
                'scanner_context': scan.context,
                'current_price': entry_price,
                'atr': atr,
                'stop_price': levels.stop_price,
                'tp_price': levels.tp_price,
                'stop_reason': f"MODULAR_{bracket_config['type'].upper()}",
                'risk_points': levels.risk_points,
                'window': {
                    'x_price_1m': features.x_price_1m.tolist() if features.x_price_1m is not None else [],
                    'raw_ohlcv_1m': raw_ohlcv,
                    'x_context': features.x_context.tolist() if features.x_context is not None else [],
                },
                'oco': {
                    'entry_price': entry_price,
                    'stop_price': levels.stop_price,
                    'tp_price': levels.tp_price,
                    'direction': direction,
                    'atr_at_creation': atr,
                    'max_bars': 100
                },
                'oco_results': {
                    'modular': {
                        'outcome': cf.outcome,
                        'pnl_dollars': cf.pnl_dollars,
                        'bars_held': int(cf.bars_held),
                        'exit_price': float(cf.exit_price),
                    }
                },
                'best_oco': 'modular',
                'best_pnl': cf.pnl_dollars
            }
            records.append(record)
            decision_idx += 1

    # 5. Save Results
    if records:
        with open(out_dir / "records.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        
        summary = {
            "run_id": out_dir.name,
            "total_trades": len(records),
            "win_rate": len([r for r in records if r['oco_results']['modular']['pnl_dollars'] > 0]) / len(records),
            "config": config
        }
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
            
        print(f"\nSaved {len(records)} records to {out_dir}")
    else:
        print("\nNo trades triggered.")


if __name__ == "__main__":
    main()

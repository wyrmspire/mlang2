#!/usr/bin/env python
"""
Puller Strategy Backtest
Scans for measured move failure patterns and generates viz-compatible artifacts.

Usage:
    python scripts/backtest_puller.py --weeks 4 --out results/viz/puller_v6_4w
"""

import sys
import json
import argparse
import uuid
import hashlib
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
from src.policy.library.puller import PullerScanner
from src.labels.counterfactual import compute_smart_stop_counterfactual
from src.config import RESULTS_DIR


def make_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
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
    elif pd.isna(obj):
        return None
    return obj


def get_raw_ohlcv_window(stepper: MarketStepper, lookback: int = 60, lookahead: int = 60):
    """
    Get raw OHLCV window around current position for visualization.
    """
    df = stepper.df
    current_idx = stepper.current_idx
    
    start_idx = max(0, current_idx - lookback)
    end_idx = min(len(df), current_idx + lookahead + 1)
    
    combined = df.iloc[start_idx:end_idx]
    
    return [
        {
            "time": r['time'].isoformat() if hasattr(r['time'], 'isoformat') else str(r['time']),
            "open": float(r['open']),
            "high": float(r['high']),
            "low": float(r['low']),
            "close": float(r['close']),
            "volume": float(r['volume'])
        }
        for _, r in combined.iterrows()
    ]


def main():
    parser = argparse.ArgumentParser(description="Run Puller strategy backtest")
    parser.add_argument("--weeks", type=int, default=4, help="Number of weeks to backtest")
    parser.add_argument("--out", type=str, default=None, help="Output directory")
    parser.add_argument("--variation", type=str, default="v6_scalp", help="Variation to run")
    
    args = parser.parse_args()
    
    # Date range: use last N weeks of available data
    end_date = pd.Timestamp("2025-09-17", tz="America/New_York")
    start_date = end_date - timedelta(weeks=args.weeks)
    
    # Setup output
    run_id = f"puller_{args.variation}_{args.weeks}w"
    out_dir = Path(args.out) if args.out else RESULTS_DIR / "viz" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # V6 Scalp parameters (best win rate from prior testing)
    scanner = PullerScanner(
        variation_id=args.variation,
        entry_unit=0.5,
        stop_unit=1.0,
        tp_unit=-2.0,
        min_move_unit=1.5,
        max_move_unit=2.5,
        max_duration_bars=45
    )
    
    print("=" * 60)
    print(f"Puller Strategy Backtest ({args.variation})")
    print(f"Period: {start_date.date()} to {end_date.date()} ({args.weeks} weeks)")
    print(f"Output: {out_dir}")
    print("=" * 60)
    
    # 1. Load data
    print("\n[1] Loading data...")
    df = load_continuous_contract()
    df = df[(df['time'] >= str(start_date)) & (df['time'] < str(end_date))].reset_index(drop=True)
    print(f"Loaded {len(df)} 1m bars")
    
    if len(df) == 0:
        print("ERROR: No data in range")
        return
    
    # 2. Resample to higher timeframes
    print("\n[2] Resampling to higher timeframes...")
    htf_data = resample_all_timeframes(df)
    df_5m = htf_data.get('5m')
    df_15m = htf_data.get('15m')
    print(f"5m bars: {len(df_5m)}, 15m bars: {len(df_15m)}")
    
    # 3. Precompute ATR on 5m
    print("\n[3] Computing ATR...")
    df_5m_copy = df_5m.copy()
    df_5m_copy['atr'] = calculate_atr(df_5m_copy, 14)
    avg_atr = df_5m_copy['atr'].dropna().mean()
    print(f"Average 5m ATR: {avg_atr:.2f} points")
    
    # 4. Setup stepper
    print("\n[4] Running simulation...")
    stepper = MarketStepper(df, start_idx=100, end_idx=len(df) - 100)
    feature_config = FeatureConfig(lookback_1m=60)
    
    decisions = []
    trades = []
    decision_idx = 0
    
    while True:
        step = stepper.step()
        if step.is_done:
            break
        
        # Progress
        if step.bar_idx % 5000 == 0:
            print(f"  Processing bar {step.bar_idx}/{len(df)}...")
        
        # Compute features
        features = compute_features(
            stepper,
            feature_config,
            df_5m=df_5m,
            df_15m=df_15m,
        )
        
        # Check scanner
        scan_result = scanner.scan(features.market_state, features)
        if not scan_result.triggered:
            continue
        
        ctx = scan_result.context
        direction = ctx.get('direction', 'SHORT')
        entry_price = ctx.get('entry_price', features.current_price)
        stop_price = ctx.get('stop_loss', entry_price)
        tp_price = ctx.get('take_profit', entry_price)
        atr = ctx.get('atr', avg_atr)
        
        # Compute counterfactual outcome
        cf = compute_smart_stop_counterfactual(
            df=df,
            entry_idx=step.bar_idx,
            direction=direction,
            stop_price=stop_price,
            tp_multiple=None,  # Use explicit TP
            atr=atr,
            oco_name=f"puller_{args.variation}",
            tp_price=tp_price
        )
        
        decision_id = str(uuid.uuid4())[:8]
        trade_id = str(uuid.uuid4())[:8]
        
        # Get raw OHLCV for visualization
        raw_ohlcv = get_raw_ohlcv_window(stepper, lookback=60, lookahead=60)
        
        # Build decision record (VizDecision-compatible)
        decision = {
            'decision_id': decision_id,
            'timestamp': features.timestamp.isoformat() if features.timestamp else None,
            'bar_idx': step.bar_idx,
            'action': direction,
            
            # Scanner identification
            'scanner_id': scanner.scanner_id,
            'scanner_context': ctx,
            
            # Market state
            'current_price': features.current_price,
            'atr': atr,
            
            # Window (VizWindow-compatible structure)
            'window': {
                'raw_ohlcv_1m': raw_ohlcv,
                'x_context': [],
            },
            
            # OCO (VizOCO-compatible structure)
            'oco': {
                'entry_price': entry_price,
                'stop_price': stop_price,
                'tp_price': tp_price,
                'direction': direction,
                'atr_at_creation': atr,
                'max_bars': 200,
            },
            
            'oco_results': {
                f"puller_{args.variation}": {
                    'outcome': cf.outcome,
                    'pnl_dollars': cf.pnl_dollars,
                    'bars_held': cf.bars_held,
                }
            }
        }
        decisions.append(decision)
        
        # Build trade record
        trade = {
            'trade_id': trade_id,
            'decision_id': decision_id,
            'index': decision_idx,
            'direction': direction,
            'scanner_id': scanner.scanner_id,
            'entry_time': features.timestamp.isoformat() if features.timestamp else None,
            'entry_price': entry_price,
            'stop_price': stop_price,
            'tp_price': tp_price,
            'exit_price': cf.exit_price if hasattr(cf, 'exit_price') else entry_price,
            'pnl_dollars': cf.pnl_dollars,
            'outcome': cf.outcome,
            'exit_reason': cf.outcome,
            'bars_held': cf.bars_held,
            'mae': cf.mae,
            'mfe': cf.mfe,
        }
        trades.append(trade)
        
        decision_idx += 1
    
    print(f"\n[5] Completed! Total triggers: {len(decisions)}")
    
    # 5. Analyze results
    if trades:
        wins = sum(1 for t in trades if t['pnl_dollars'] > 0)
        losses = len(trades) - wins
        total_pnl = sum(t['pnl_dollars'] for t in trades)
        longs = sum(1 for t in trades if t['direction'] == 'LONG')
        shorts = len(trades) - longs
        
        print("\n[6] Results:")
        print(f"  Trades: {len(trades)} (LONG: {longs}, SHORT: {shorts})")
        print(f"  Wins: {wins}, Losses: {losses}, Win Rate: {wins/len(trades)*100:.1f}%")
        print(f"  Total PnL: ${total_pnl:,.2f}")
    
    # 6. Write output files
    print(f"\n[7] Writing output to {out_dir}...")
    
    # decisions.jsonl
    decisions_path = out_dir / "decisions.jsonl"
    with open(decisions_path, 'w') as f:
        for d in decisions:
            f.write(json.dumps(make_serializable(d)) + '\n')
    
    # trades.jsonl
    trades_path = out_dir / "trades.jsonl"
    with open(trades_path, 'w') as f:
        for t in trades:
            f.write(json.dumps(make_serializable(t)) + '\n')
    
    # manifest.json
    now = pd.Timestamp.now(tz="America/New_York")
    manifest = {
        "run_id": run_id,
        "fingerprint": hashlib.md5(run_id.encode()).hexdigest()[:16],
        "config": {
            "description": f"Puller {args.variation} backtest",
            "weeks": args.weeks,
            "variation": args.variation,
        },
        "created_at": now.isoformat(),
        "strategy": "puller",
        "files": {
            "decisions": "decisions.jsonl",
            "trades": "trades.jsonl"
        },
        "counts": {
            "decisions": len(decisions),
            "trades": len(trades),
        }
    }
    
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"  Wrote {len(decisions)} decisions, {len(trades)} trades")
    print(f"\n✅ Results at: {out_dir}")
    print(f"Validate with: python golden/validate_run.py {out_dir}")


if __name__ == "__main__":
    main()

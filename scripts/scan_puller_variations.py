"""
Scan Puller Variations
Runs 10 variations of the Puller Strategy and outputs valid viz artifacts.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_continuous_contract
from src.data.resample import resample_all_timeframes
from src.features.indicators import calculate_atr
from src.features.state import get_market_state
from src.features.pipeline import FeatureBundle, compute_features, FeatureConfig
from src.sim.stepper import MarketStepper
from src.policy.library.puller import PullerScanner
from src.config import RESULTS_DIR
from src.labels.future_window import FutureWindowProvider
from src.labels.trade_outcome import compute_trade_outcome
from src.sim.costs import DEFAULT_COSTS
from src.sim.bar_fill_model import BarFillConfig

def simulate_trade(df, start_idx, entry_price, stop, tp, direction):
    # Setup
    future_provider = FutureWindowProvider(df, start_idx)
    costs = DEFAULT_COSTS
    fill_config = BarFillConfig()
    
    outcome = compute_trade_outcome(
        future_provider=future_provider,
        entry_price=entry_price,
        direction=direction,
        stop_loss=stop,
        take_profit=tp,
        max_bars=500,
        fill_config=fill_config
    )
    
    pnl_dollars = costs.calculate_pnl(
        entry_price,
        outcome.exit_price,
        direction,
        contracts=1,
        include_commission=True
    )
    
    return {
        "entry_time": df.iloc[start_idx]['time'], # Approx, actually fill time might be later
        "exit_time": df.iloc[start_idx + outcome.bars_held]['time'] if start_idx + outcome.bars_held < len(df) else df.iloc[-1]['time'],
        "entry_price": entry_price,
        "exit_price": outcome.exit_price,
        "pnl_raw": outcome.pnl,
        "pnl_dollars": pnl_dollars,
        "outcome": outcome.outcome,
        "bars_held": outcome.bars_held
    }

def run_scan():
    variations = []
    
    # Base: min 1.5, max 2.5, entry 0.75, stop 2.0, tp -4.0
    # Variation ideas: 
    # - Different entry depth (0.5, 1.0)
    # - Different Stop/TP ratios
    # - Different Max Duration
    
    # 10 Variations for multi-OCO training
    # V1: Default
    variations.append(PullerScanner(variation_id="v1_default"))
    # V2: Tighter Entry
    variations.append(PullerScanner(variation_id="v2_tight_entry", entry_unit=0.5))
    # V3: Deeper Entry
    variations.append(PullerScanner(variation_id="v3_deep_entry", entry_unit=1.0))
    # V4: Wider Stop
    variations.append(PullerScanner(variation_id="v4_wide_stop", stop_unit=3.0))
    # V5: Big Target
    variations.append(PullerScanner(variation_id="v5_big_target", tp_unit=-6.0))
    # V6: Scalp
    variations.append(PullerScanner(variation_id="v6_scalp", tp_unit=-2.0, stop_unit=1.0))
    # V7: Longer Duration
    variations.append(PullerScanner(variation_id="v7_long_fuse", max_duration_bars=60))
    # V8: Short Duration
    variations.append(PullerScanner(variation_id="v8_short_fuse", max_duration_bars=30))
    # V9: Big Move Required
    variations.append(PullerScanner(variation_id="v9_big_move", min_move_unit=2.0))
    # V10: Huge Range Allowed
    variations.append(PullerScanner(variation_id="v10_huge_range", max_move_unit=3.5))

    # 2. Load Data (6 Weeks for training)
    print("\n[2] Loading Data (6 Weeks)...")
    # End date: 2025-09-17 (Max in data)
    # Start date: 6 weeks prior
    end_date = pd.Timestamp("2025-09-17", tz="America/New_York")
    start_date = end_date - pd.Timedelta(weeks=6)
    
    print(f"  Range: {start_date.date()} to {end_date.date()}")
    
    df = load_continuous_contract()
    df = df[(df['time'] >= str(start_date))].reset_index(drop=True)
    
    if len(df) == 0:
        print("Error: No data found.")
        return

    # Compute Indicators
    print("  Computing Indicators...")
    # We need ATR for the scanner
    htf = resample_all_timeframes(df)
    df_5m = htf['5m']
    df_5m['atr_14'] = calculate_atr(df_5m, 14)
    df_5m_indexed = df_5m.set_index('time').sort_index()


    # 3. Running Scan
    print("\n[3] Running Scan...")
    
    # Initialize Stepper
    stepper = MarketStepper(df)
    feature_config = FeatureConfig(lookback_1m=60) # Need lookback for pattern scanner
    
    all_decisions = []
    all_trades = []
    
    run_id = "puller_v6_scalp_4w"
    out_dir = RESULTS_DIR / "viz" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    decision_counter = 0

    # Main Loop
    while True:
        step = stepper.step()
        if step.is_done:
            break
            
        if step.bar_idx % 1000 == 0:
            print(f"  Processing bar {step.bar_idx}/{len(df)}...", end='\r')
            
        # Get State
        # Optimization: Only compute full features if we have enough bars
        if step.bar_idx < 60:
            continue
            
        features = compute_features(stepper, feature_config, df_5m=df_5m)
        state = features.market_state
        
        # We need to manually inject 5m ATR into features.indicators if it's not mapped automatically
        # Features pipeline usually maps specific columns. 
        # Let's ensure 'atr_5m_14' is available.
        # compute_features usually pulls from df_5m if aligned.
        # If not, we can grab it manually.
        current_time = state.current_time
        
        # Find 5m row
        # (This is slow in a tight loop, usually handled by pipeline, but fine for script)
        # Hack: features.indicators is a SimpleNamespace.
        if features.indicators is None:
             from types import SimpleNamespace
             features.indicators = SimpleNamespace()
        
        # Get latest 5m ATR
        # Find row <= current_time
        # Optimization: use pre-calculated index map or just resampling
        # For now, let's rely on valid pipeline state or simple approximation
        # features.atr is usually available if computed?
        
        # Let's grab specific 5m ATR from the resampled definition
        # We can look it up by time
        row_5m = df_5m_indexed.asof(current_time)
        atr_val = row_5m.get('atr_14', 10.0)
        if pd.isna(atr_val): atr_val = 10.0
        
        features.indicators.atr_5m_14 = atr_val
        
        # Iterate Variations
        found_trigger = False
        
        for v in variations:
            res = v.scan(state, features)
            
            if res.triggered:
                # Deduplicate? Or record all?
                # User wants "10 different variations", presumably we can have multiple triggers on same bar.
                
                decision_id = f"{run_id}_{decision_counter:05d}"
                decision_counter += 1
                
                ctx = res.context
                
                # Create Decision Record
                decision = {
                    "decision_id": decision_id,
                    "timestamp": state.current_time.isoformat(),
                    "scanner_id": res.scanner_id,
                    "symbol": "MNQ", 
                    "timeframe": "1m",
                    "action": ctx['direction'], # Mapped from direction
                    "bar_idx": step.bar_idx, # Added
                    "direction": ctx['direction'],
                    "sugg_entry": ctx['entry_price'],
                    "sugg_stop": ctx['stop_loss'],
                    "sugg_tp": ctx['take_profit'],
                    "scanner_context": ctx,
                    "x_price_1m": state.ohlcv_1m[:, 3].tolist(), 
                    "window": {
                        "raw_ohlcv_1m": state.ohlcv_1m.tolist(),
                        "x_context": [atr_val] * len(state.ohlcv_1m) 
                    }
                }
                
                all_decisions.append(decision)
                
                # Simulate Trade (Counterfactual)
                # "when price closes below 0 place a limit order"
                # Scan happens at close of bar. So limit order placed for NEXT bars.
                
                trade_res = simulate_trade(
                    df=df,
                    start_idx=step.bar_idx, 
                    entry_price=ctx['entry_price'],
                    stop=ctx['stop_loss'],
                    tp=ctx['take_profit'],
                    direction=ctx['direction']
                )
                
                if trade_res:
                     # ... map to dict ...
                     trade = {
                        "trade_id": f"trade_{decision_id}",
                        "decision_id": decision_id,
                        "scanner_id": res.scanner_id,
                        # ...
                        "entry_time": str(trade_res['entry_time']),
                        "exit_time": str(trade_res['exit_time']),
                        "entry_price": trade_res['entry_price'],
                        "exit_price": trade_res['exit_price'],
                        "pnl_raw": trade_res['pnl_raw'],
                        "pnl_dollars": trade_res['pnl_dollars'],
                        "outcome": trade_res['outcome'],
                        "bars_held": trade_res['bars_held']
                    }
                     all_trades.append(trade)

                # Optional: limit hit rate to avoid flood?
                # For now let it run.
                
    # 4. Save Artifacts
    print(f"\n[4] Saving {len(all_decisions)} decisions and {len(all_trades)} trades...")
    
    # Decisions.jsonl
    with open(out_dir / "decisions.jsonl", "w") as f:
        for d in all_decisions:
            f.write(json.dumps(d, cls=NumpyEncoder) + "\n")
            
    # Trades.jsonl
    with open(out_dir / "trades.jsonl", "w") as f:
        for t in all_trades:
            f.write(json.dumps(t, cls=NumpyEncoder) + "\n")
            
    # Manifest.json
    manifest = {
        "run_id": run_id,
        "strategy_name": "Puller Variations",
        "timestamp": pd.Timestamp.now(tz="America/New_York").isoformat(),
        "created_at": pd.Timestamp.now(tz="America/New_York").isoformat(), # Required
        "fingerprint": "v1_scan", # Required
        "stats": {
            "count": len(all_trades),
            "pnl": sum(t['pnl_dollars'] for t in all_trades)
        }
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"✅ Success! Results at: {out_dir}")
    print("Run validation with: python golden/validate_run.py results/viz/puller_variations_9w")

if __name__ == "__main__":
    run_scan()

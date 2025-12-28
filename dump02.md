
Thin wrapper around run_strategy_scan.

Strategy: When trending and price pulls back to 200 EMA and rejects,
take the rejection. Especially if 20 EMA is still angled and near a level.

Usage:
    python scripts/scan_ema200_rejection.py --weeks 4 --start-date 2025-08-18
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.strategy.scan import run_strategy_scan, RTHFilter, MinATRFilter
from src.policy.triggers.ema_rejection import EMA200RejectionTrigger
from src.policy.brackets import ATRBracket
from src.features.indicators import add_ema
from src.config import NY_TZ
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="EMA 200 Rejection Strategy Scanner")
    parser.add_argument("--start-date", type=str, default="2025-08-18")
    parser.add_argument("--weeks", type=int, default=4)
    parser.add_argument("--stop-atr", type=float, default=1.0, help="Stop loss in ATR")
    parser.add_argument("--tp-r", type=float, default=2.0, help="Take profit R-multiple")
    parser.add_argument("--timeframe", type=str, default="5m")
    parser.add_argument("--out", type=str, default=None)
    
    args = parser.parse_args()
    
    # Create trigger and bracket
    trigger = EMA200RejectionTrigger(
        ema_fast_period=20,
        ema_slow_period=200,
        slope_threshold=0.02
    )
    bracket = ATRBracket(
        stop_atr=args.stop_atr,
        tp_multiple=args.tp_r
    )
    
    # Precompute EMAs for the data range
    from src.data.loader import load_continuous_contract
    from src.data.resample import resample_all_timeframes
    from src.features.indicators import calculate_atr
    
    start = pd.Timestamp(args.start_date)
    end = start + pd.Timedelta(weeks=args.weeks, unit='W')
    
    df_1m = load_continuous_contract()
    # Need extra lookback for 200 EMA
    extended_start = start - pd.Timedelta(days=10)
    df_1m = df_1m[(df_1m['time'] >= str(extended_start)) & (df_1m['time'] < str(end))].reset_index(drop=True)
    
    htf_data = resample_all_timeframes(df_1m)
    df_5m = htf_data.get('5m')
    
    if df_5m is not None and len(df_5m) > 200:
        # Add EMAs
        df_5m['ema_20'] = df_5m['close'].ewm(span=20, adjust=False).mean()
        df_5m['ema_200'] = df_5m['close'].ewm(span=200, adjust=False).mean()
        df_5m['atr'] = calculate_atr(df_5m, 14)
        
        # Compute 20 EMA slope
        df_5m['ema_20_slope'] = (df_5m['ema_20'] - df_5m['ema_20'].shift(5)) / 5
        
        # Normalize slope by ATR for threshold comparison
        df_5m['ema_20_slope'] = df_5m['ema_20_slope'] / df_5m['atr'].replace(0, 1)
    
    # Create context function to pass EMAs to trigger
    def add_ema_context(bar, features):
        """Add EMA values to features for trigger."""
        # Find matching bar in df_5m by time
        bar_time = pd.Timestamp(bar['time'])
        
        # Get values from dataframe
        idx = bar.name if hasattr(bar, 'name') else 0
        
        return {
            'ema_20': df_5m.iloc[idx]['ema_20'] if idx < len(df_5m) else 0,
            'ema_200': df_5m.iloc[idx]['ema_200'] if idx < len(df_5m) else 0,
            'ema_20_slope': df_5m.iloc[idx]['ema_20_slope'] if idx < len(df_5m) else 0
        }
    
    # Run the scan
    run_name = args.out or f"NEW_ema200_rejection_{args.start_date.replace('-', '')}"
    
    result = run_strategy_scan(
        trigger=trigger,
        bracket=bracket,
        start_date=args.start_date,
        weeks=args.weeks,
        filters=[RTHFilter(), MinATRFilter(threshold=2.0)],
        run_name=run_name,
        timeframe=args.timeframe,
        extra_context_fn=add_ema_context
    )
    
    print(f"\nRun ID for UI: {result.run_name}")


if __name__ == "__main__":
    main()

```

### scripts/scan_fakeout_fade.py

```python
"""
Fakeout Fade Strategy Scanner

Thin wrapper around run_strategy_scan - ALL output handling is built-in.

Usage:
    python scripts/scan_fakeout_fade.py --level pdh --weeks 4 --start-date 2025-08-18
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.strategy.scan import run_strategy_scan, RTHFilter, MinATRFilter
from src.policy.triggers.fakeout import FakeoutTrigger
from src.policy.brackets import FixedBracket
from src.features.levels import get_previous_day_levels
from src.config import NY_TZ
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Fakeout Fade Strategy Scanner")
    parser.add_argument("--level", type=str, default="pdh", choices=["pdh", "pdl"])
    parser.add_argument("--start-date", type=str, default="2025-08-18")
    parser.add_argument("--weeks", type=int, default=4)
    parser.add_argument("--stop-points", type=float, default=2.0)
    parser.add_argument("--tp-r", type=float, default=2.0)
    parser.add_argument("--timeframe", type=str, default="5m")
    parser.add_argument("--out", type=str, default=None)
    
    args = parser.parse_args()
    
    # Create trigger and bracket
    trigger = FakeoutTrigger(level=args.level)
    bracket = FixedBracket(
        stop_points=args.stop_points, 
        tp_points=args.stop_points * args.tp_r
    )
    
    # Create custom context function to add PDH/PDL to features
    # (Precompute daily levels once)
    from src.data.loader import load_continuous_contract
    from src.data.resample import resample_all_timeframes
    
    start = pd.Timestamp(args.start_date)
    end = start + pd.Timedelta(weeks=args.weeks, unit='W')
    
    df_1m = load_continuous_contract()
    df_1m = df_1m[(df_1m['time'] >= str(start)) & (df_1m['time'] < str(end))].reset_index(drop=True)
    
    # Precompute PDH/PDL for all dates
    df = df_1m.copy()
    df['time'] = pd.to_datetime(df['time'])
    if df['time'].dt.tz is None:
        df['time'] = df['time'].dt.tz_localize(NY_TZ)
    df['date'] = df['time'].dt.date
    
    daily_levels = {}
    unique_dates = sorted(df['date'].unique())
    for i, current_date in enumerate(unique_dates):
        if i == 0:
            continue
        prev_date = unique_dates[i - 1]
        prev_day_data = df[df['date'] == prev_date]
        if not prev_day_data.empty:
            daily_levels[current_date] = {
                'pdh': prev_day_data['high'].max(),
                'pdl': prev_day_data['low'].min()
            }
    
    def add_pdh_pdl(bar, features):
        """Add PDH/PDL to features for trigger to use."""
        bar_time = pd.Timestamp(bar['time'])
        if bar_time.tzinfo is None:
            bar_time = bar_time.tz_localize(NY_TZ)
        bar_date = bar_time.date()
        
        levels = daily_levels.get(bar_date, {})
        return {
            'pdh': levels.get('pdh', 0),
            'pdl': levels.get('pdl', 0)
        }
    
    # Run the scan - ALL outputs are handled automatically
    run_name = args.out or f"fakeout_{args.level}_{args.start_date.replace('-', '')}"
    
    result = run_strategy_scan(
        trigger=trigger,
        bracket=bracket,
        start_date=args.start_date,
        weeks=args.weeks,
        filters=[RTHFilter(), MinATRFilter(threshold=2.0)],
        run_name=run_name,
        timeframe=args.timeframe,
        extra_context_fn=add_pdh_pdl
    )
    
    print(f"\nRun ID for UI: {result.run_name}")


if __name__ == "__main__":
    main()

```

### scripts/scan_or_false_break.py

```python
"""
NY Opening Range False Break Strategy Scanner

Strategy: If we break the OR early and come back inside within the first hour,
that feels like a trap → fade it back to the other side of the range.

Usage:
    python scripts/scan_or_false_break.py --weeks 4 --start-date 2025-08-18
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.strategy.scan import run_strategy_scan, RTHFilter, MinATRFilter
from src.policy.triggers.or_false_break import ORFalseBreakTrigger
from src.policy.brackets import RangeBracket
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="NY Opening Range False Break Scanner")
    parser.add_argument("--session", type=str, default="NY", 
                        choices=["NY", "LONDON", "ASIA"],
                        help="Which session's OR to use")
    parser.add_argument("--or-minutes", type=int, default=15,
                        help="Minutes to establish OR (15 = 9:30-9:45)")
    parser.add_argument("--max-return", type=int, default=30,
                        help="Max minutes for price to return inside OR")
    parser.add_argument("--tp-multiple", type=float, default=2.0,
                        help="Target as multiple of risk")
    parser.add_argument("--start-date", type=str, default="2025-08-18")
    parser.add_argument("--weeks", type=int, default=4)
    parser.add_argument("--timeframe", type=str, default="5m")
    parser.add_argument("--out", type=str, default=None)
    
    args = parser.parse_args()
    
    # Create trigger for OR false break
    trigger = ORFalseBreakTrigger(
        or_minutes=args.or_minutes,
        max_return_minutes=args.max_return,
        session=args.session
    )
    
    # Range-based bracket: SL = range size, TP = R-multiple of risk
    bracket = RangeBracket(tp_multiple=args.tp_multiple)
    
    # Run the scan
    run_name = args.out or f"NEW_or_false_break_{args.session}_{args.start_date.replace('-', '')}"
    
    result = run_strategy_scan(
        trigger=trigger,
        bracket=bracket,
        start_date=args.start_date,
        weeks=args.weeks,
        filters=[RTHFilter()],  # Only RTH filter - we're trading the NY session
        run_name=run_name,
        timeframe=args.timeframe
    )
    
    print(f"\nRun ID for UI: {result.run_name}")


if __name__ == "__main__":
    main()

```

### scripts/scan_pdh_sweep.py

```python
"""
PDH Sweep Fade Strategy Scanner

Strategy: Fade PDH sweeps - when price barely takes out PDH like a stop run,
then immediately stalls, short back under PDH with tight stop above sweep.

Usage:
    python scripts/scan_pdh_sweep.py --weeks 4 --start-date 2025-08-18
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.strategy.scan import run_strategy_scan, RTHFilter, MinATRFilter
from src.policy.triggers.sweep import SweepTrigger
from src.policy.brackets import FixedBracket
from src.features.indicators import calculate_atr
from src.config import NY_TZ
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="PDH Sweep Fade Strategy Scanner")
    parser.add_argument("--level", type=str, default="pdh", choices=["pdh", "pdl"],
                        help="Level to fade sweeps at")
    parser.add_argument("--max-sweep", type=float, default=3.0,
                        help="Max points beyond level to count as sweep")
    parser.add_argument("--start-date", type=str, default="2025-08-18")
    parser.add_argument("--weeks", type=int, default=4)
    parser.add_argument("--stop-above-sweep", type=float, default=1.0,
                        help="Points above sweep high for stop")
    parser.add_argument("--tp-r", type=float, default=2.0)
    parser.add_argument("--timeframe", type=str, default="5m")
    parser.add_argument("--out", type=str, default=None)
    
    args = parser.parse_args()
    
    # Create sweep trigger
    trigger = SweepTrigger(
        level=args.level,
        sweep_type="fade",
        max_sweep_pts=args.max_sweep,
        min_sweep_pts=0.25
    )
    
    # For sweep fades, stop is above the sweep high (tight stop)
    # We'll compute this dynamically in the context function
    
    # Use a placeholder bracket - actual stop will be sweep_high + buffer
    bracket = FixedBracket(
        stop_points=args.stop_above_sweep + args.max_sweep,  # Conservative estimate
        tp_points=(args.stop_above_sweep + args.max_sweep) * args.tp_r
    )
    
    # Precompute daily levels
    from src.data.loader import load_continuous_contract
    from src.data.resample import resample_all_timeframes
    
    start = pd.Timestamp(args.start_date)
    end = start + pd.Timedelta(weeks=args.weeks, unit='W')
    
    df_1m = load_continuous_contract()
    df_1m = df_1m[(df_1m['time'] >= str(start)) & (df_1m['time'] < str(end))].reset_index(drop=True)
    
    htf_data = resample_all_timeframes(df_1m)
    df_5m = htf_data.get('5m')
    
    if df_5m is not None:
        df_5m['time_dt'] = pd.to_datetime(df_5m['time'])
        if df_5m['time_dt'].dt.tz is None:
            df_5m['time_dt'] = df_5m['time_dt'].dt.tz_localize(NY_TZ)
        df_5m['date'] = df_5m['time_dt'].dt.date
        
        # Compute PDH/PDL per date
        daily = df_5m.groupby('date').agg({'high': 'max', 'low': 'min'}).reset_index()
        daily['pdh'] = daily['high'].shift(1)
        daily['pdl'] = daily['low'].shift(1)
        daily_dict = {row['date']: {'pdh': row['pdh'], 'pdl': row['pdl']} 
                      for _, row in daily.iterrows() if pd.notna(row['pdh'])}
        
        df_5m['pdh'] = df_5m['date'].map(lambda d: daily_dict.get(d, {}).get('pdh', 0))
        df_5m['pdl'] = df_5m['date'].map(lambda d: daily_dict.get(d, {}).get('pdl', 0))
    
    def add_levels(bar, features):
        """Add PDH/PDL to features for trigger."""
        idx = bar.name if hasattr(bar, 'name') else 0
        if idx >= len(df_5m):
            return {}
        
        row = df_5m.iloc[idx]
        return {
            'pdh': row.get('pdh', 0),
            'pdl': row.get('pdl', 0),
        }
    
    # Run the scan
    run_name = args.out or f"NEW_sweep_{args.level}_{args.start_date.replace('-', '')}"
    
    result = run_strategy_scan(
        trigger=trigger,
        bracket=bracket,
        start_date=args.start_date,
        weeks=args.weeks,
        filters=[RTHFilter(), MinATRFilter(threshold=2.0)],
        run_name=run_name,
        timeframe=args.timeframe,
        extra_context_fn=add_levels
    )
    
    print(f"\nRun ID for UI: {result.run_name}")


if __name__ == "__main__":
    main()

```

### scripts/scan_power_hour.py

```python
"""
Power Hour VWAP Reclaim Strategy Scanner

Strategy: After 2:30 PM, if price was under VWAP all morning,
then reclaims and holds for 10 minutes → long to PDH/day high.

Usage:
    python scripts/scan_power_hour.py --weeks 4 --start-date 2025-08-18
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.strategy.scan import run_strategy_scan, RTHFilter, MinATRFilter
from src.policy.triggers.vwap_reclaim import VWAPReclaimTrigger
from src.policy.brackets import ATRBracket


def main():
    parser = argparse.ArgumentParser(description="Power Hour VWAP Reclaim Scanner")
    parser.add_argument("--min-time", type=str, default="14:30",
                        help="Earliest trigger time (HH:MM ET)")
    parser.add_argument("--hold-minutes", type=int, default=10,
                        help="Minutes to hold above VWAP")
    parser.add_argument("--min-below", type=int, default=30,
                        help="Minimum minutes below VWAP before reclaim")
    parser.add_argument("--tp-atr", type=float, default=3.0,
                        help="Take profit as ATR multiple")
    parser.add_argument("--start-date", type=str, default="2025-08-18")
    parser.add_argument("--weeks", type=int, default=4)
    parser.add_argument("--timeframe", type=str, default="1m")
    parser.add_argument("--out", type=str, default=None)
    
    args = parser.parse_args()
    
    # Create trigger for VWAP reclaim
    trigger = VWAPReclaimTrigger(
        min_time=args.min_time,
        hold_minutes=args.hold_minutes,
        min_below_minutes=args.min_below
    )
    
    # Use ATR bracket - stop at 2 ATR, target PDH or 3 ATR
    bracket = ATRBracket(stop_atr=2.0, tp_atr=args.tp_atr)
    
    # Add VWAP to features via extra_context_fn
    # Signature: (bar: pd.Series, features: MockFeatures) -> dict of extra attributes
    def add_vwap(bar, features):
        """Add VWAP to feature bundle for trigger."""
        result = {}
        # VWAP should already be in the bar if pipeline computed it
        if 'vwap_session' in bar:
            result['vwap_session'] = bar['vwap_session']
        return result
    
    # Run the scan
    run_name = args.out or f"NEW_power_hour_{args.start_date.replace('-', '')}"
    
    result = run_strategy_scan(
        trigger=trigger,
        bracket=bracket,
        start_date=args.start_date,
        weeks=args.weeks,
        filters=[RTHFilter()],
        run_name=run_name,
        timeframe=args.timeframe,
        extra_context_fn=add_vwap
    )
    
    print(f"\nRun ID for UI: {result.run_name}")


if __name__ == "__main__":
    main()

```

### scripts/scan_puller_variations.py

```python
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

```

### scripts/scan_test_simple.py

```python
"""
Midday Entry Strategy - For Testing

Enters LONG at 12:00 PM ET every trading day.
Holds for 1 hour (counterfactual calculates outcome).

Uses 1-minute timeframe so we hit exact times.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.strategy.scan import run_strategy_scan, RTHFilter
from src.policy.triggers.time_trigger import TimeTrigger
from src.policy.brackets import FixedBracket


def main():
    # Trigger at noon ET every day
    trigger = TimeTrigger(
        hour=12,  # Noon
        minute=0,
        direction="LONG"
    )
    
    # Fixed bracket: 3 point stop, 5 point target
    bracket = FixedBracket(stop_points=3.0, tp_points=5.0)
    
    # Run scan on August data (1 month)
    result = run_strategy_scan(
        trigger=trigger,
        bracket=bracket,
        start_date="2025-08-01",
        weeks=4,  # 1 month
        filters=[RTHFilter()],
        run_name="midday_test",
        timeframe="1m",  # 1-minute to hit exact times
    )
    
    print(f"\nScan complete! Run ID: {result.run_name}")
    print(f"Decisions: {result.total_decisions}")
    print(f"Trades: {result.total_trades}")


if __name__ == "__main__":
    main()

```

### scripts/session_ifvg_replay.py

```python
#!/usr/bin/env python3
"""
IFVG CNN Replay Runner

Run simulation/replay mode using the trained IFVG CNN.
When the CNN detects a pattern with high confidence, it triggers a trade
using the IFVG entry rules (limit order at FVG midpoint).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import time
import torch
import numpy as np
import pandas as pd
from datetime import timedelta

from src.config import MODELS_DIR, NY_TZ
from src.data.loader import load_continuous_contract
from src.data.resample import resample_all_timeframes
from src.features.fvg import find_fvg
from src.features.swings import find_swings, count_levels_swept
from src.sim.stepper import MarketStepper


# ============================================================================
# MODEL
# ============================================================================

class IFVGPatternCNN(torch.nn.Module):
    """CNN for IFVG pattern detection (must match training architecture)."""
    
    def __init__(self, input_channels: int = 5, seq_length: int = 30, num_classes: int = 2):
        super().__init__()
        
        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(input_channels, 16, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            
            torch.nn.Conv1d(16, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            
            torch.nn.Conv1d(32, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),
        )
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(32, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
    
    def predict_proba(self, x):
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)


def load_ifvg_model(model_path: Path):
    """Load trained IFVG CNN."""
    model = IFVGPatternCNN()
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ============================================================================
# HELPERS
# ============================================================================

def emit_event(event_type: str, data: dict):
    """Emit event as JSON line to stdout."""
    event = {'type': event_type, **data}
    print(json.dumps(event), flush=True)


def normalize_window(ohlcv: np.ndarray) -> np.ndarray:
    """Normalize OHLCV window for CNN input."""
    ohlcv = ohlcv.copy().astype(np.float32)
    
    # Normalize price columns by first close
    first_close = ohlcv[3, 0]
    if first_close > 0:
        ohlcv[0:4] = (ohlcv[0:4] - first_close) / first_close * 100
    
    # Normalize volume by max
    max_vol = ohlcv[4].max() if ohlcv[4].max() > 0 else 1
    ohlcv[4] = ohlcv[4] / max_vol
    
    return ohlcv


def get_price_window(df_1m: pd.DataFrame, bar_idx: int, lookback: int = 30) -> np.ndarray:
    """Extract (5, lookback) price window from 1m data."""
    start_idx = max(0, bar_idx - lookback)
    window = df_1m.iloc[start_idx:bar_idx]
    
    if len(window) < lookback:
        return None
    
    ohlcv = np.array([
        window['open'].values,
        window['high'].values,
        window['low'].values,
        window['close'].values,
        window['volume'].values
    ], dtype=np.float32)
    
    return ohlcv


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run IFVG CNN Replay")
    parser.add_argument("--model", type=str, default="models/ifvg_cnn.pth",
                        help="Path to trained IFVG model")
    parser.add_argument("--start-date", type=str, default="2025-03-18",
                        help="Start date for replay")
    parser.add_argument("--days", type=int, default=1,
                        help="Number of days to replay")
    parser.add_argument("--speed", type=float, default=10.0,
                        help="Speed multiplier")
    parser.add_argument("--threshold", type=float, default=0.65,
                        help="Confidence threshold for triggering")
    
    args = parser.parse_args()
    
    model_path = Path(args.model)
    if not model_path.exists():
        emit_event('ERROR', {'message': f'Model not found: {model_path}'})
        sys.exit(1)
    
    # Load model
    emit_event('STATUS', {'message': 'Loading IFVG CNN model...'})
    model = load_ifvg_model(model_path)
    
    # Load data
    emit_event('STATUS', {'message': 'Loading market data...'})
    df = load_continuous_contract()
    
    start_date = pd.Timestamp(args.start_date, tz=NY_TZ)
    end_date = start_date + timedelta(days=args.days)
    
    df = df[(df['time'] >= start_date) & (df['time'] < end_date)].reset_index(drop=True)
    if len(df) < 60:
        emit_event('ERROR', {'message': f'Not enough data: {len(df)} bars'})
        sys.exit(1)
    
    emit_event('STATUS', {'message': f'Loaded {len(df)} bars'})
    
    # Resample
    htf_data = resample_all_timeframes(df)
    df_5m = htf_data.get('5m')
    
    # Initialize stepper
    stepper = MarketStepper(df, start_idx=30, end_idx=len(df) - 10)
    
    # Emit replay start
    emit_event('REPLAY_START', {
        'start_date': start_date.isoformat(),
        'end_date': end_date.isoformat(),
        'total_bars': len(df),
        'model': str(model_path),
        'strategy': 'IFVG CNN'
    })
    
    bar_delay = 1.0 / args.speed
    decision_count = 0
    trigger_count = 0
    
    # Track recent FVGs for IFVG detection
    recent_fvgs = []
    
    while True:
        step = stepper.step()
        if step.is_done:
            break
        
        bar_idx = step.bar_idx
        current_bar = step.bar
        timestamp = current_bar['time']
        
        # Emit bar update
        emit_event('BAR', {
            'bar_idx': bar_idx,
            'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
            'open': float(current_bar['open']),
            'high': float(current_bar['high']),
            'low': float(current_bar['low']),
            'close': float(current_bar['close']),
            'volume': float(current_bar['volume'])
        })
        
        # Only check every 5 bars for efficiency
        if bar_idx % 5 != 0:
            time.sleep(bar_delay)
            continue
        
        # Get price window
        price_window = get_price_window(df, bar_idx, lookback=30)
        if price_window is None:
            time.sleep(bar_delay)
            continue
        
        # Normalize and convert to tensor
        x = normalize_window(price_window)
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # (1, 5, 30)
        
        # Run CNN inference
        with torch.no_grad():
            probs = model.predict_proba(x_tensor)
            long_prob = float(probs[0, 1])  # P(LONG)
            short_prob = float(probs[0, 0])  # P(SHORT)
        
        # Determine prediction and confidence
        if long_prob > short_prob:
            direction = "LONG"
            confidence = long_prob
        else:
            direction = "SHORT"
            confidence = short_prob
        
        decision_count += 1
        triggered = confidence >= args.threshold
        
        emit_event('DECISION', {
            'decision_id': f'ifvg_cnn_{decision_count:04d}',
            'bar_idx': bar_idx,
            'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
            'direction': direction,
            'confidence': round(confidence, 4),
            'long_prob': round(long_prob, 4),
            'short_prob': round(short_prob, 4),
            'threshold': args.threshold,
            'triggered': triggered,
            'price': float(current_bar['close'])
        })
        
        if triggered:
            trigger_count += 1
            
            # Calculate entry levels using current price and ATR
            atr = (df.iloc[max(0, bar_idx-14):bar_idx]['high'].max() - 
                   df.iloc[max(0, bar_idx-14):bar_idx]['low'].min()) / 3
            
            entry_price = float(current_bar['close'])
            if direction == "LONG":
                stop_price = entry_price - atr
                tp_price = entry_price + (2 * atr)
            else:
                stop_price = entry_price + atr
                tp_price = entry_price - (2 * atr)
            
            emit_event('OCO_OPEN', {
                'decision_id': f'ifvg_cnn_{decision_count:04d}',
                'direction': direction,
                'entry_price': round(entry_price, 2),
                'stop_price': round(stop_price, 2),
                'tp_price': round(tp_price, 2),
                'confidence': round(confidence, 4)
            })
        
        time.sleep(bar_delay)
    
    emit_event('REPLAY_END', {
        'total_bars_processed': decision_count * 5,
        'total_decisions': decision_count,
        'total_triggers': trigger_count,
        'trigger_rate': f'{trigger_count/max(1,decision_count)*100:.1f}%'
    })


if __name__ == "__main__":
    main()

```

### scripts/session_ifvg_simulation.py

```python
#!/usr/bin/env python3
"""
IFVG CNN Simulation Runner

Uses the trained 4-class CNN with IFVG Scanner for live simulation.
Pipeline:
1. IFVG Scanner detects setup (FVG inversion + liquidity)
2. CNN predicts class probabilities [LONG_WIN, LONG_LOSS, SHORT_WIN, SHORT_LOSS]
3. If quality threshold met, take the trade with predicted direction
4. Execute with limit order OCO (entry at FVG midpoint, SL at invalidation, TP at 3R)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import timedelta, time as dt_time
from zoneinfo import ZoneInfo

from src.config import MODELS_DIR, NY_TZ
from src.data.loader import load_continuous_contract
from src.data.resample import resample_all_timeframes
from src.sim.stepper import MarketStepper

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = MODELS_DIR / "ifvg_4class_cnn.pth"
MIN_FVG_POINTS = 2.0
INVERSION_WINDOW = 12  # 1 hour on 5m
SL_PADDING = 1.0
RISK_REWARD = 3.0
MIN_WIN_PROB = 0.40  # Minimum P(WIN) to take trade
POINT_VALUE = 50


# ============================================================================
# MODEL
# ============================================================================

class IFVG4ClassCNN(nn.Module):
    def __init__(self, input_channels=5, seq_length=30, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )
    
    def forward(self, x):
        return self.classifier(self.features(x))
    
    def predict_proba(self, x):
        return torch.softmax(self.forward(x), dim=-1)


def load_model(path: Path):
    model = IFVG4ClassCNN()
    state_dict = torch.load(path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ============================================================================
# FVG DETECTION (simplified from debug scanner)
# ============================================================================

def find_fvgs(df_5m: pd.DataFrame, min_gap: float = 2.0):
    fvgs = []
    for i in range(1, len(df_5m) - 1):
        prev = df_5m.iloc[i-1]
        curr = df_5m.iloc[i]
        next_ = df_5m.iloc[i+1]
        bar_time = curr['time'] if 'time' in curr else df_5m.index[i]
        
        # Bullish FVG
        bullish_gap = next_['low'] - prev['high']
        if bullish_gap >= min_gap:
            fvgs.append({
                'type': 'BULLISH', 'time': bar_time, 'bar_idx': i,
                'high': next_['low'], 'low': prev['high'],
                'midpoint': (next_['low'] + prev['high']) / 2,
                'gap': bullish_gap
            })
        
        # Bearish FVG
        bearish_gap = prev['low'] - next_['high']
        if bearish_gap >= min_gap:
            fvgs.append({
                'type': 'BEARISH', 'time': bar_time, 'bar_idx': i,
                'high': prev['low'], 'low': next_['high'],
                'midpoint': (prev['low'] + next_['high']) / 2,
                'gap': bearish_gap
            })
    return fvgs


def check_inversion(fvgs, new_idx, window=12):
    """Check if there's an opposite FVG within window."""
    if new_idx >= len(fvgs):
        return None
    new_fvg = fvgs[new_idx]
    opposite = 'BULLISH' if new_fvg['type'] == 'BEARISH' else 'BEARISH'
    
    for i in range(new_idx - 1, -1, -1):
        old_fvg = fvgs[i]
        if old_fvg['type'] == opposite:
            bar_diff = new_fvg['bar_idx'] - old_fvg['bar_idx']
            if bar_diff <= window:
                return old_fvg
            break
    return None


# ============================================================================
# HELPERS
# ============================================================================

def emit_event(event_type: str, data: dict):
    print(json.dumps({'type': event_type, **data}), flush=True)


def normalize_window(ohlcv: np.ndarray) -> np.ndarray:
    ohlcv = ohlcv.copy().astype(np.float32)
    first_close = ohlcv[3, 0]
    if first_close > 0:
        ohlcv[0:4] = (ohlcv[0:4] - first_close) / first_close * 100
    max_vol = ohlcv[4].max() if ohlcv[4].max() > 0 else 1
    ohlcv[4] = ohlcv[4] / max_vol
    return ohlcv


def get_price_window(df_1m: pd.DataFrame, bar_idx: int, lookback: int = 30):
    start_idx = max(0, bar_idx - lookback)
    window = df_1m.iloc[start_idx:bar_idx]
    if len(window) < lookback:
        return None
    return np.array([
        window['open'].values, window['high'].values,
        window['low'].values, window['close'].values,
        window['volume'].values
    ], dtype=np.float32)


def decide_trade(probs):
    """
    Decide whether to trade based on 4-class probabilities.
    
    probs: [P(LONG_WIN), P(LONG_LOSS), P(SHORT_WIN), P(SHORT_LOSS)]
    
    Returns: (take_trade, direction, confidence)
    """
    long_win = probs[0]
    long_loss = probs[1]
    short_win = probs[2]
    short_loss = probs[3]
    
    # Quality = P(WIN | direction)
    long_quality = long_win / (long_win + long_loss + 1e-6)
    short_quality = short_win / (short_win + short_loss + 1e-6)
    
    # Which direction is better?
    if long_win > short_win and long_quality >= MIN_WIN_PROB:
        return True, "LONG", float(long_win), float(long_quality)
    elif short_win > long_win and short_quality >= MIN_WIN_PROB:
        return True, "SHORT", float(short_win), float(short_quality)
    else:
        return False, None, 0, 0


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="IFVG CNN Simulation")
    parser.add_argument("--model", type=str, default=str(MODEL_PATH))
    parser.add_argument("--start-date", type=str, default="2025-03-24")
    parser.add_argument("--days", type=int, default=1)
    parser.add_argument("--speed", type=float, default=10.0)
    parser.add_argument("--min-quality", type=float, default=0.40)
    args = parser.parse_args()
    
    global MIN_WIN_PROB
    MIN_WIN_PROB = args.min_quality
    
    model_path = Path(args.model)
    if not model_path.exists():
        emit_event('ERROR', {'message': f'Model not found: {model_path}'})
        sys.exit(1)
    
    emit_event('STATUS', {'message': 'Loading model...'})
    model = load_model(model_path)
    
    emit_event('STATUS', {'message': 'Loading data...'})
    df_1m = load_continuous_contract()
    
    start_date = pd.Timestamp(args.start_date, tz=NY_TZ)
    end_date = start_date + timedelta(days=args.days)
    
    df_1m = df_1m[(df_1m['time'] >= start_date) & (df_1m['time'] < end_date)].reset_index(drop=True)
    if len(df_1m) < 60:
        emit_event('ERROR', {'message': f'Not enough data: {len(df_1m)} bars'})
        sys.exit(1)
    
    htf = resample_all_timeframes(df_1m)
    df_5m = htf.get('5m')
    if 'time' not in df_5m.columns:
        df_5m = df_5m.reset_index()
    
    emit_event('REPLAY_START', {
        'start_date': start_date.isoformat(),
        'end_date': end_date.isoformat(),
        'total_bars': len(df_1m),
        'model': str(model_path),
        'strategy': 'IFVG 4-Class CNN'
    })
    
    bar_delay = 1.0 / args.speed
    stepper = MarketStepper(df_1m, start_idx=30, end_idx=len(df_1m) - 10)
    
    # Track state
    recent_fvgs = []
    last_fvg_check_5m_idx = -1
    triggers = 0
    cooldown_until = -1
    
    while True:
        step = stepper.step()
        if step.is_done:
            break
        
        bar_idx = step.bar_idx
        current_bar = step.bar
        timestamp = current_bar['time']
        
        emit_event('BAR', {
            'bar_idx': bar_idx,
            'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
            'open': float(current_bar['open']),
            'high': float(current_bar['high']),
            'low': float(current_bar['low']),
            'close': float(current_bar['close']),
            'volume': float(current_bar['volume'])
        })
        
        # Only check on 5m boundaries
        if bar_idx % 5 != 0:
            time.sleep(bar_delay)
            continue
        
        # Cooldown
        if bar_idx < cooldown_until:
            time.sleep(bar_delay)
            continue
        
        # Find 5m index
        current_5m_idx = bar_idx // 5
        if current_5m_idx >= len(df_5m):
            time.sleep(bar_delay)
            continue
        
        # Scan for new FVGs
        df_5m_up_to = df_5m.iloc[:current_5m_idx + 1]
        fvgs = find_fvgs(df_5m_up_to, MIN_FVG_POINTS)
        
        # Check for new FVG
        if len(fvgs) > len(recent_fvgs):
            new_fvg = fvgs[-1]
            recent_fvgs = fvgs[-20:]  # Keep last 20
            
            # Check for inversion
            old_fvg = check_inversion(fvgs, len(fvgs) - 1, INVERSION_WINDOW)
            
            if old_fvg:
                # IFVG detected! Run CNN
                price_window = get_price_window(df_1m, bar_idx, lookback=30)
                
                if price_window is not None:
                    x = normalize_window(price_window)
                    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
                    
                    with torch.no_grad():
                        probs = model.predict_proba(x_tensor)[0].numpy()
                    
                    take_trade, direction, confidence, quality = decide_trade(probs)
                    
                    emit_event('IFVG_DETECTED', {
                        'fvg_type': new_fvg['type'],
                        'fvg_gap': round(new_fvg['gap'], 2),
                        'probs': {
                            'LONG_WIN': round(float(probs[0]), 4),
                            'LONG_LOSS': round(float(probs[1]), 4),
                            'SHORT_WIN': round(float(probs[2]), 4),
                            'SHORT_LOSS': round(float(probs[3]), 4)
                        },
                        'take_trade': take_trade,
                        'direction': direction,
                        'quality': round(quality, 4)
                    })
                    
                    if take_trade:
                        triggers += 1
                        cooldown_until = bar_idx + 30  # 30 min cooldown
                        
                        # Calculate levels
                        entry = new_fvg['midpoint']
                        if direction == "SHORT":
                            stop = new_fvg['high'] + SL_PADDING
                            risk = stop - entry
                            tp = entry - (RISK_REWARD * risk)
                        else:
                            stop = new_fvg['low'] - SL_PADDING
                            risk = entry - stop
                            tp = entry + (RISK_REWARD * risk)
                        
                        # Emit DECISION for UI compatibility (includes OCO levels)
                        emit_event('DECISION', {
                            'decision_id': f'ifvg_sim_{triggers:04d}',
                            'bar_idx': bar_idx,
                            'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
                            'win_probability': quality,
                            'threshold': MIN_WIN_PROB,
                            'triggered': True,
                            'price': entry,
                            'stop_price': round(stop, 2),
                            'tp_price': round(tp, 2),
                            'direction': direction,
                            'atr': risk
                        })
                        
                        emit_event('OCO_OPEN', {
                            'decision_id': f'ifvg_sim_{triggers:04d}',
                            'direction': direction,
                            'entry_price': round(entry, 2),
                            'stop_price': round(stop, 2),
                            'tp_price': round(tp, 2),
                            'confidence': round(confidence, 4),
                            'quality': round(quality, 4),
                            'fvg_gap': round(new_fvg['gap'], 2)
                        })
        
        time.sleep(bar_delay)
    
    emit_event('REPLAY_END', {
        'total_triggers': triggers
    })


if __name__ == "__main__":
    main()

```

### scripts/session_live.py

```python
#!/usr/bin/env python3
"""
Live Mode Simulator

Runs a strategy in "Real-Time" (Simulated execution on real data).
- Loads 7 days history from YFinance.
- Simulates past days to build equity curve.
- Enters LIVE mode and waits for new bars to trade in real-time.
- Emits JSON events for the frontend UI.

Usage:
    python scripts/session_live.py --ticker MES=F --strategy ema_cross
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime

from src.sim.yfinance_stepper import YFinanceStepper
from src.features.indicators import calculate_atr, calculate_ema
from src.policy.library.ict_ifvg import ICTIFVGScanner
from src.policy.entry_scans import EntryOrder, EntryConfig, apply_entry_scans

# Global scanner instance (stateful)
_ifvg_scanner = None

# =============================================================================
# Strategy Logic
# =============================================================================

def check_ema_cross(df_history: pd.DataFrame) -> dict:
    """Check for 9/21 EMA cross."""
    if len(df_history) < 30:
        return None
        
    df = df_history.copy()
    df['ema_fast'] = calculate_ema(df['close'], 9)
    df['ema_slow'] = calculate_ema(df['close'], 21)
    
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Cross UP
    if prev['ema_fast'] <= prev['ema_slow'] and curr['ema_fast'] > curr['ema_slow']:
        return {'direction': 'LONG', 'confidence': 0.8}
    
    # Cross DOWN
    if prev['ema_fast'] >= prev['ema_slow'] and curr['ema_fast'] < curr['ema_slow']:
        return {'direction': 'SHORT', 'confidence': 0.8}
        
    return None

def check_orb(df_history: pd.DataFrame) -> dict:
    """Simple ORB (9:30-10:00 range) check."""
    # Need access to full day session. This is harder with just a tail df.
    # Placeholder for now.
    return None


def check_ifvg(df_history: pd.DataFrame, bar_idx: int) -> dict:
    """
    Check for IFVG setup using the real ICTIFVGScanner.
    Returns signal dict with direction, entry, stop, tp or None.
    """
    global _ifvg_scanner
    if _ifvg_scanner is None:
        _ifvg_scanner = ICTIFVGScanner(
            min_liquidity_score=2,  # Relaxed for more signals
            inversion_window_bars=6,
            swing_lookback=5,
            min_gap_atr=0.15,
            risk_reward=2.0,
            cooldown_bars=6
        )
    
    if len(df_history) < 30:
        return None
    
    # Calculate ATR
    atr_series = calculate_atr(df_history, 14)
    atr = float(atr_series.iloc[-1]) if len(atr_series) > 0 else 5.0
    
    # Check for setup
    setup = _ifvg_scanner.check(df_history, bar_idx, atr=atr)
    
    if setup:
        return {
            'direction': setup.direction,  # 'LONG' or 'SHORT'
            'confidence': 0.7 + (setup.liquidity_score * 0.05),  # Score-based confidence
            'entry_price': setup.entry_price,
            'stop_price': setup.stop_price,
            'tp_price': setup.tp_price
        }
    return None

# =============================================================================
# Helper
# =============================================================================

def emit(event_type: str, data: dict):
    """Emit JSON event."""
    msg = {'type': event_type, **data}
    print(json.dumps(msg), flush=True)

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Live Trading Simulator")
    parser.add_argument("--ticker", type=str, default="MES=F", help="Ticker symbol")
    parser.add_argument("--strategy", type=str, default="ema_cross", help="Strategy to run")
    parser.add_argument("--days", type=int, default=7, help="History days")
    parser.add_argument("--speed", type=float, default=10.0, help="Historical playback speed")
    
    # Entry scan configuration
    parser.add_argument("--entry-params", type=str, default="{}", help="JSON entry params")
    
    args = parser.parse_args()
    
    # Initialize OCO Engine
    from src.sim.oco_engine import OCOEngine, OCOConfig, StopConfig
    from src.sim.costs import DEFAULT_COSTS
    
    oco_engine = OCOEngine(costs=DEFAULT_COSTS)
    entry_params = json.loads(args.entry_params)
    
    emit('STATUS', {'message': f'Initializing Live Mode for {args.ticker}...'})
    
    try:
        stepper = YFinanceStepper(ticker=args.ticker, days_back=args.days)
    except Exception as e:
        emit('ERROR', {'message': f'Failed to init stepper: {str(e)}'})
        return

    emit('REPLAY_START', {
        'start_date': str(stepper.df['time'].iloc[0]),
        'total_bars': len(stepper.df),
        'strategy': args.strategy,
        'mode': 'LIVE_SIMULATION'
    })

    # EMIT INITIAL HISTORY BATCH
    # Convert entire history DataFrame to list of dicts for frontend
    history_bars = []
    for _, row in stepper.df.iterrows():
        history_bars.append({
            'timestamp': str(row['time']),
            'close': float(row['close']),
            'high': float(row['high']),
            'low': float(row['low']),
            'open': float(row['open']),
            'volume': float(row['volume'])
        })
    emit('HISTORY', {'bars': history_bars})
    
    decision_count = 0
    # bar_delay = 1.0 / args.speed # No delay needed for history batch
    
    print(f"Processing history...", file=sys.stderr)
    
    live_mode_notified = False

    while True:
        # Step
        step = stepper.step()
        
        # If None, it means we are waiting for live data
        if step is None and stepper.live_mode:
            if not live_mode_notified:
                print(">>> ENTERING LIVE MODE - Waiting for market updates <<<", file=sys.stderr)
                emit('STATUS', {'message': 'History complete. Entered LIVE mode.'})
                live_mode_notified = True
            time.sleep(1)
            continue
            
        bar = step.bar
        
        # Determine if this is a "New Live Bar" or "History Bar"
        if stepper.live_mode:
            emit('BAR', {
                'bar_idx': step.bar_idx,
                'timestamp': str(bar['time']),
                'close': float(bar['close']),
                'high': float(bar['high']),
                'low': float(bar['low']),
                'open': float(bar['open']),
                'volume': float(bar['volume'])
            })
        
        # Run Strategy
        history = stepper.get_history(lookback=60)
        signal = None
        
        if args.strategy == "ema_cross":
            signal = check_ema_cross(history)
        elif args.strategy == "ifvg":
            signal = check_ifvg(history, step.bar_idx)
            
        if signal:
            decision_count += 1
            emit('DECISION', {
                'decision_id': f'live_{decision_count:04d}',
                'bar_idx': step.bar_idx,
                'timestamp': str(bar['time']),
                'direction': signal['direction'],
                'confidence': signal['confidence'],
                'price': float(bar['close']),
                'triggered': True
            })
            
            # Use OCOEngine to create bracket
            # If signal provides explicit prices (IFVG), favor those?
            # Or use dynamic entry parameters if override not set?
            
            # For this modular update, we prioritize the USER SELECTED entry strategy
            # UNLESS the signal is "High Precision" (like IFVG providing exact levels).
            # But the user specifically asked for modular entry tools.
            # So we will use the OCO Engine's calculation.
            
            # Construct OCO Config
            config = OCOConfig(
                direction=signal['direction'],
                entry_type=args.entry_type.upper(), # 'MARKET', 'LIMIT', 'RETRACE_SIGNAL'
                entry_params=entry_params,
                stop_atr=args.stop_atr,
                tp_multiple=args.tp_r,
                entry_offset_atr=0.0 # Legacy
            )
            
            # Create Bracket (Calculates prices)
            atr_series = calculate_atr(history, 14)
            atr = float(atr_series.iloc[-1]) if len(atr_series) > 0 else 5.0
            
            bracket = oco_engine.create_bracket(
                config=config,
                base_price=float(bar['close']),
                atr=atr,
                df_1m=history, # Pass history as 1m context
                df_htf=history, # Pass history as htf (simplification for now)
                current_idx=len(history)-1
            )
            
            emit('OCO_OPEN', {
                'decision_id': f'live_{decision_count:04d}',
                'direction': bracket.config.direction,
                'entry_price': round(bracket.entry_price, 2),
                'stop_price': round(bracket.stop_price, 2),
                'tp_price': round(bracket.tp_price, 2),
                'entry_type': bracket.config.entry_type
            })
            
        # No artificial delay needed in history since we sent batch

if __name__ == "__main__":
    main()

```

### scripts/session_replay.py

```python
"""
Replay Mode Runner

Run trained CNN model on historical data bar-by-bar, emitting events.
Usage: python scripts/session_replay.py --model models/best_model.pth --start-date 2025-03-17 --days 1
"""

import argparse
import json
import sys
import time
import torch
import pandas as pd
from pathlib import Path
from datetime import timedelta

from src.config import RESULTS_DIR, NY_TZ
from src.data.loader import load_continuous_contract
from src.data.resample import resample_all_timeframes
from src.features.indicators import calculate_atr
from src.features.pipeline import compute_features, FeatureConfig
from src.sim.stepper import MarketStepper
from src.models.fusion import FusionModel, SimpleCNN
from src.core.enums import RunMode


def normalize_window(x, method='zscore'):
    """Simple z-score normalization for price windows."""
    import numpy as np
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0) + 1e-8
    return (x - mean) / std



def load_model(model_path: Path):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Determine model type from checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Determine model architecture and num_classes
    num_classes = 2
    # Check for FusionModel classifier (layer 6)
    if 'classifier.6.weight' in state_dict:
        num_classes = state_dict['classifier.6.weight'].shape[0]
    elif 'classifier.6.bias' in state_dict:
        num_classes = state_dict['classifier.6.bias'].shape[0]
    # Check for SimpleCNN classifier (layer 4)
    elif 'classifier.4.weight' in state_dict:
        num_classes = state_dict['classifier.4.weight'].shape[0]
    elif 'classifier.4.bias' in state_dict:
        num_classes = state_dict['classifier.4.bias'].shape[0]
        
    emit_event('STATUS', {'message': f'Detected {num_classes} output classes in model'})

    if any('price_encoder' in k for k in state_dict.keys()):
        model = FusionModel(num_classes=num_classes)
    else:
        model = SimpleCNN(num_classes=num_classes)
    
    model.load_state_dict(state_dict)
    model.eval()
    return model


def emit_event(event_type: str, data: dict):
    """Emit event as JSON line to stdout."""
    event = {
        'type': event_type,
        **data
    }
    print(json.dumps(event), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Run CNN Model Replay")
    parser.add_argument("--model", type=str, default="models/best_model.pth",
                        help="Path to trained model")
    parser.add_argument("--start-date", type=str, default="2025-03-17",
                        help="Start date for replay")
    parser.add_argument("--days", type=int, default=1,
                        help="Number of days to replay")
    parser.add_argument("--speed", type=float, default=10.0,
                        help="Speed multiplier (1.0 = real-time, 10.0 = 10x)")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="Confidence threshold for triggering")
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory for decisions")
    
    args = parser.parse_args()
    
    model_path = Path(args.model)
    if not model_path.exists():
        emit_event('ERROR', {'message': f'Model not found: {model_path}'})
        sys.exit(1)
    
    # Load model
    emit_event('STATUS', {'message': 'Loading model...'})
    model = load_model(model_path)
    
    # Load data
    emit_event('STATUS', {'message': 'Loading market data...'})
    df = load_continuous_contract()
    
    start_date = pd.Timestamp(args.start_date, tz=NY_TZ)
    end_date = start_date + timedelta(days=args.days)
    
    df = df[(df['time'] >= start_date) & (df['time'] < end_date)].reset_index(drop=True)
    if len(df) < 200:
        emit_event('ERROR', {'message': f'Not enough data: {len(df)} bars'})
        sys.exit(1)
    
    emit_event('STATUS', {'message': f'Loaded {len(df)} bars'})
    
    # Resample for higher timeframes
    htf_data = resample_all_timeframes(df)
    df_5m = htf_data.get('5m')
    df_15m = htf_data.get('15m')
    df_5m['atr'] = calculate_atr(df_5m, 14)
    
    # Initialize stepper
    stepper = MarketStepper(df, start_idx=120, end_idx=len(df) - 30)
    feature_config = FeatureConfig(lookback_1m=120)
    
    # Emit replay start
    emit_event('REPLAY_START', {
        'start_date': start_date.isoformat(),
        'end_date': end_date.isoformat(),
        'total_bars': len(df),
        'model': str(model_path)
    })
    
    # Delay between bars based on speed
    bar_delay = 1.0 / args.speed  # seconds per bar
    
    decision_count = 0
    trigger_count = 0
    decisions = []
    
    while True:
        step = stepper.step()
        if step.is_done:
            break
        
        bar_idx = step.bar_idx
        current_bar = step.bar
        timestamp = current_bar['time']
        
        # Emit bar update
        emit_event('BAR', {
            'bar_idx': bar_idx,
            'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
            'open': float(current_bar['open']),
            'high': float(current_bar['high']),
            'low': float(current_bar['low']),
            'close': float(current_bar['close']),
            'volume': float(current_bar['volume'])
        })
        
        # Only run model every 5 bars (reduce noise) 
        if bar_idx % 5 != 0:
            time.sleep(bar_delay)
            continue
        
        # Compute features
        try:
            features = compute_features(stepper, feature_config, df_5m=df_5m, df_15m=df_15m)
        except Exception as e:
            time.sleep(bar_delay)
            continue
        
        # Prepare model inputs
        x_1m = features.x_price_1m
        x_5m = features.x_price_5m
        x_15m = features.x_price_15m
        
        if x_1m is None or len(x_1m) < 60:
            time.sleep(bar_delay)
            continue
        
        # Normalize all timeframes
        x_1m_norm = normalize_window(x_1m, method='zscore')
        x_5m_norm = normalize_window(x_5m, method='zscore') if x_5m is not None and len(x_5m) > 0 else np.zeros((24, 5))
        x_15m_norm = normalize_window(x_15m, method='zscore') if x_15m is not None and len(x_15m) > 0 else np.zeros((8, 5))
        
        # Convert to tensors: (1, channels, length)
        import numpy as np
        x_1m_t = torch.tensor(x_1m_norm.T, dtype=torch.float32).unsqueeze(0)
        x_5m_t = torch.tensor(x_5m_norm.T, dtype=torch.float32).unsqueeze(0)
        x_15m_t = torch.tensor(x_15m_norm.T, dtype=torch.float32).unsqueeze(0)
        
        # Context vector (use indicators if available)
        context_dim = 20
        context = np.zeros(context_dim)
        if features.indicators:
            ind = features.indicators
            context[0] = getattr(ind, 'rsi_1m_14', 50) / 100 if hasattr(ind, 'rsi_1m_14') else 0.5
            context[1] = getattr(ind, 'rsi_5m_14', 50) / 100 if hasattr(ind, 'rsi_5m_14') else 0.5
            context[2] = features.atr / 20 if features.atr else 0
        x_context = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            try:
                if hasattr(model, 'price_encoder'):
                    # FusionModel
                    probs = model.predict_proba(x_1m_t, x_5m_t, x_15m_t, x_context)
                    win_prob = float(probs[0])
                elif hasattr(model, 'features'):
                    # SimpleCNN
                    logits = model(x_1m_t)
                    probs = torch.softmax(logits, dim=-1)
                    win_prob = float(probs[0, 1]) if probs.shape[-1] > 1 else float(probs[0, 0])
                else:
                    win_prob = 0.5
            except Exception as e:
                emit_event('DEBUG', {'error': str(e)})
                win_prob = 0.5
        
        decision_count += 1
        
        # Emit decision event
        triggered = win_prob >= args.threshold
        
        if triggered:
            # Determine direction from class or from heuristic if scalar
            # 4-class: 0=NoSignal, 1=Long, 2=Short, 3=Wait? (Hypothetical)
            # Binary: 0=No, 1=Yes
            
            # For this patch, assume default simple Binary or assume win_prob > thresh means action.
            # Ideally we check model logic. 
            
            # Simple heuristic:
            # If we don't know direction, default to LONG for now or alternating?
            # Actually, IFVG strategy usually implies direction from the pattern.
            # Pure CNN Replay: The model output SHOULD imply direction.
            # If output is single float (win_prob), it usually implies one specific setups (e.g. Long-only model?)
            # Ref: ifvg_4class usually has [Long prob, Short prob, ...]
            
            # Let's inspect probs if available
            direction = 'LONG' 
            if 'probs' in locals():
                if probs.shape[-1] == 3: # [No, Long, Short]
                    if probs[0, 2] > probs[0, 1]: direction = 'SHORT'
                elif probs.shape[-1] == 4: # [No, Long, Short, Other]
                     if probs[0, 2] > probs[0, 1]: direction = 'SHORT'
                # If binary, maybe >0.5 is Long? Or model is Long-Only.
            
            atr = float(features.atr) if features.atr else 0
            if atr == 0: atr = current_bar['close'] * 0.001
            
            entry_price = float(current_bar['close'])
            stop_price = entry_price - (2 * atr) if direction == 'LONG' else entry_price + (2 * atr)
            tp_price = entry_price + (4 * atr) if direction == 'LONG' else entry_price - (4 * atr)

            emit_event('DECISION', {
                'decision_id': f'replay_{decision_count:04d}',
                'bar_idx': bar_idx,
                'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
                'win_probability': round(win_prob, 4),
                'threshold': args.threshold,
                'triggered': True,
                'price': entry_price,
                'atr': atr,
                'direction': direction,
                'stop_price': stop_price,
                'tp_price': tp_price
            })
        else:
            emit_event('DECISION', {
                'decision_id': f'replay_{decision_count:04d}',
                'bar_idx': bar_idx,
                'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
                'win_probability': round(win_prob, 4),
                'threshold': args.threshold,
                'triggered': False,
                'price': float(current_bar['close']),
                'atr': float(features.atr) if features.atr else 0
            })
        
        if triggered:
            trigger_count += 1
            decisions.append({
                'decision_id': f'replay_{decision_count:04d}',
                'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
                'bar_idx': bar_idx,
                'win_probability': win_prob,
                'price': float(current_bar['close'])
            })
        
        # Delay for visualization
        time.sleep(bar_delay)
    
    # Emit replay end
    emit_event('REPLAY_END', {
        'total_bars_processed': decision_count * 5,
        'total_decisions': decision_count,
        'total_triggers': trigger_count,
        'trigger_rate': f'{trigger_count/max(1,decision_count)*100:.1f}%'
    })
    
    # Save decisions if output specified
    if args.out and decisions:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / 'replay_decisions.jsonl', 'w') as f:
            for d in decisions:
                f.write(json.dumps(d) + '\n')
        emit_event('STATUS', {'message': f'Saved {len(decisions)} decisions to {out_dir}'})


if __name__ == "__main__":
    main()

```

### scripts/smart_cnn.py

```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from dataclasses import dataclass
from typing import List

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import ONE_MIN_PARQUET_DIR, PROCESSED_DIR, MODELS_DIR, LOCAL_TZ
from src.utils.logging_utils import get_logger

logger = get_logger("smart_cnn")

# --- Architecture (Must match training!) ---
class TradeCNN(nn.Module):
    def __init__(self, input_len=20, input_channels=4):
        super(TradeCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(64 * 5, 32) 
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = x.permute(0, 2, 1) 
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    direction: str 
    exit_time: pd.Timestamp = None
    exit_price: float = None
    pnl: float = 0.0
    outcome: str = None 

class SmartCNNStrategy:
    def __init__(self, 
                 model_path: Path = MODELS_DIR / "setup_cnn_v1.pth",
                 tp_ticks: int = 20, 
                 sl_ticks: int = 10,
                 threshold: float = 0.6): # Confidence threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TradeCNN().to(self.device)
        
        if model_path.exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.error(f"Model not found at {model_path}")
            
        self.tp_dist = tp_ticks * 0.25
        self.sl_dist = sl_ticks * 0.25
        self.threshold = threshold
        self.trades = []

    def get_prediction(self, df_window):
        # Prepare Input
        # Needs 20 bars. 
        if len(df_window) < 20: 
            return 0.0, 0.0 # Prob Long, Prob Short
            
        # Normalize
        base_price = df_window.iloc[0]['open']
        feats = df_window[['open', 'high', 'low', 'close']].values
        feats_norm = (feats / base_price) - 1.0
        
        # Ensure exact 20
        feats_norm = feats_norm[-20:]
        
        # Create Batch (1, 20, 4) -> (1, 4, 20) handled by model
        # Input Long
        input_long = torch.FloatTensor(feats_norm).unsqueeze(0).to(self.device)
        # Input Short (Inverted)
        input_short = torch.FloatTensor(-feats_norm).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prob_long = self.model(input_long).item()
            prob_short = self.model(input_short).item()
            
        return prob_long, prob_short

    def run_simulation(self, start_date_str: str = "2025-07-07 16:40:00"):
        input_path = ONE_MIN_PARQUET_DIR / "mes_1min_all.parquet"
        if not input_path.exists(): return
        
        logger.info(f"Simulating Smart Strategy (Test Set starting {start_date_str})...")
        df_1m = pd.read_parquet(input_path)
        df_1m['time'] = pd.to_datetime(df_1m['time'])
        df_1m = df_1m.sort_values('time').set_index('time')
        
        # Filter for Test Period
        start_ts = pd.Timestamp(start_date_str).tz_localize('UTC') if 'UTC' not in start_date_str else pd.Timestamp(start_date_str)
        # Check tz awareness of df
        if df_1m.index.tz is None:
            # Assume UTC if data is UTC
            pass 
        else:
            if start_ts.tz is None: start_ts = start_ts.tz_localize('UTC')
        
        # We need context Before start_ts, so slice generously then filter triggers
        df_1m_test = df_1m.loc[start_ts - pd.Timedelta(hours=1):]
        
        # Resample for 20m triggers
        # We need "Last 5m candle" reference.
        df_5m = df_1m_test.resample('5min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
        }).dropna()
        
        triggers = df_5m[df_5m.index.minute % 20 == 0]
        triggers = triggers[triggers.index >= start_ts]
        
        logger.info(f"Found {len(triggers)} test opportunities.")
        
        count = 0
        for current_time, row in triggers.iterrows():
            trigger_time = current_time
            
            # Context for Model: 20m before trigger
            context_end = trigger_time
            context_start = context_end - pd.Timedelta(minutes=20)
            
            # Fetch 1m context
            context_window = df_1m.loc[context_start:context_end]
            # Precise slice: strictly < trigger_time?
            context_window = context_window[context_window.index < trigger_time]
            
            if len(context_window) < 15: continue # Skip if missing data
            
            p_long, p_short = self.get_prediction(context_window)
            
            if count < 20: 
                 logger.info(f"Pred: L={p_long:.4f} S={p_short:.4f}")
            
            # Decision
            direction = None
            # Relaxed logic: If both meet threshold, pick higher or random tie-break
            if p_long > self.threshold and p_long >= p_short:
                direction = 'LONG'
                confidence = p_long
            elif p_short > self.threshold and p_short > p_long:
                direction = 'SHORT'
                confidence = p_short
                
            if not direction:
                continue
                
            # EXECUTION (Dynamic Sizing)
            # Need previous 5m candle for sizing
            prev_time = trigger_time - pd.Timedelta(minutes=5)
            if prev_time not in df_5m.index: continue
            prev_bar = df_5m.loc[prev_time]
            candle_range = prev_bar['high'] - prev_bar['low']
            if candle_range == 0: candle_range = 0.25
            
            sl_dist = 2.0 * candle_range
            tp_dist = 3.0 * candle_range
            
            entry_price = prev_bar['close'] # Approx fill at close of prev bar (Open of current)
            if trigger_time in df_1m.index:
                 entry_price = df_1m.loc[trigger_time]['open']
            
            if direction == 'LONG':
                sl_price = entry_price - sl_dist
                tp_price = entry_price + tp_dist
            else:
                sl_price = entry_price + sl_dist
                tp_price = entry_price - tp_dist
                
            # Simulate Outcome
            future = df_1m.loc[trigger_time:]
            outcome = 'TIMEOUT'
            exit_px = entry_price
            exit_t = trigger_time
            
            # Vectorized Check (Subset 2000 bars)
            subset = future.iloc[:2000]
            if subset.empty: continue
            
            times = subset.index.values
            highs = subset['high'].values
            lows = subset['low'].values
            closes = subset['close'].values
            
            if direction == 'LONG':
                 mask_win = highs >= tp_price
                 mask_loss = lows <= sl_price
            else:
                 mask_win = lows <= tp_price
                 mask_loss = highs >= sl_price
                 
            idx_win = np.argmax(mask_win) if mask_win.any() else 999999
            idx_loss = np.argmax(mask_loss) if mask_loss.any() else 999999
            
            if idx_win == 999999 and idx_loss == 999999:
                outcome = 'TIMEOUT'
                exit_px = closes[-1]
                exit_t = times[-1]
            elif idx_win < idx_loss:
                outcome = 'WIN'
                exit_px = tp_price
                exit_t = times[idx_win]
            else:
                outcome = 'LOSS'
                exit_px = sl_price
                exit_t = times[idx_loss]
            
            pnl = (exit_px - entry_price) * (1 if direction == 'LONG' else -1)
            
            self.trades.append({
                'entry_time': trigger_time,
                'direction': direction,
                'pnl': pnl,
                'outcome': outcome,
                'confidence': confidence
            })
            
            count += 1
            if count % 100 == 0:
                logger.info(f"Simulated {count} trades... Last PnL: {pnl:.2f}")

        logger.info(f"Smart Simulation Complete. Trades: {len(self.trades)}")
        if self.trades:
            df_res = pd.DataFrame(self.trades)
            wins = df_res[df_res['outcome'] == 'WIN']
            wr = len(wins) / len(df_res)
            logger.info(f"Win Rate: {wr:.2f} | Avg PnL: {df_res['pnl'].mean():.2f} | Total PnL: {df_res['pnl'].sum():.2f}")
            out_path = PROCESSED_DIR / "smart_verification_trades.parquet"
            df_res.to_parquet(out_path)

if __name__ == "__main__":
    # Threshold 0.38 since model output is around 0.40
    strat = SmartCNNStrategy(threshold=0.38) 
    strat.run_simulation()

```

### scripts/stress_test_tools.py

```python

import sys
import os
import json
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.skills.indicator_skills import GetATRTool, GetVWAPTool, DetectSupportResistanceTool, GetVolumeProfileTool
from src.skills.data_skills import GetTimeOfDayStatsTool

def stress_test():
    print("🚀 Starting Tool Stress Test\n")
    
    # 1. Test Time-of-Day Stats
    tod_tool = GetTimeOfDayStatsTool()
    tod_results = tod_tool.execute(lookback_days=10)
    print(f"✅ Time-of-Day Stats: {len(tod_results['hourly_stats'])} hours analyzed")
    
    # 2. Test ATR
    atr_tool = GetATRTool()
    atr_results = atr_tool.execute(lookback_bars=50)
    print(f"✅ ATR: Current ATR is {atr_results['current_atr']:.2f}")
    
    # 3. Test VWAP
    vwap_tool = GetVWAPTool()
    vwap_results = vwap_tool.execute(lookback_bars=5)
    print(f"✅ VWAP: Current VWAP is {vwap_results['current_vwap']:.2f}")
    
    # 4. Test S&R Detection
    sr_tool = DetectSupportResistanceTool()
    sr_results = sr_tool.execute(lookback_bars=500)
    print(f"✅ Support & Resistance: {len(sr_results['levels'])} levels detected")
    for level in sr_results['levels'][:3]:
        print(f"   - {level['type']}: {level['price']} (Strength: {level['strength']})")
        
    # 5. Test Volume Profile
    vp_tool = GetVolumeProfileTool()
    vp_results = vp_tool.execute(lookback_bars=500)
    print(f"✅ Volume Profile: POC at {vp_results['poc_price']:.2f}")

    print("\n🎉 All Research Tools Verified!")

if __name__ == "__main__":
    stress_test()

```

### scripts/sweep/__init__.py

```python
"""
Sweep Module - Parameter sweep tools for trading strategies.

Integrated with mlang2 architecture.
"""

# Core configurations
from .config import (
    PatternSweepConfig,
    CandleComposition,
    OCOBracketConfig,
    ModelSweepConfig,
    CANDLE_COMPOSITIONS,
    OCO_SWEEP_VALUES,
)

__all__ = [
    'PatternSweepConfig',
    'CandleComposition', 
    'OCOBracketConfig',
    'ModelSweepConfig',
    'CANDLE_COMPOSITIONS',
    'OCO_SWEEP_VALUES',
]

```

### scripts/sweep/config.py

```python
"""
Sweep Configuration Dataclasses
Defines all tunable parameters for the Shotgun Sweep pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import json


@dataclass
class PatternSweepConfig:
    """Configuration for pattern mining geometry."""
    rise_ratio_min: float = 2.5      # Min (Peak - Start) / (Start - Trigger)
    rise_ratio_max: float = 4.0      # Max ratio (invalidation threshold)
    min_drop: float = 1.0            # Minimum $ drop to qualify
    atr_buffer: float = 0.2          # ATR multiplier for stop placement
    validation_distance: float = 1.0  # Distance for pattern validation
    lookback_bars: int = 120         # How far back to scan for patterns
    
    # Unique identifier for this config
    config_id: str = ""
    
    def to_cli_args(self) -> List[str]:
        """Convert to CLI argument list."""
        return [
            "--rise-min", str(self.rise_ratio_min),
            "--rise-max", str(self.rise_ratio_max),
            "--min-drop", str(self.min_drop),
            "--atr-buffer", str(self.atr_buffer),
            "--validation-dist", str(self.validation_distance),
            "--lookback", str(self.lookback_bars),
        ]
    
    def to_dict(self) -> dict:
        return {
            "config_id": self.config_id,
            "rise_ratio_min": self.rise_ratio_min,
            "rise_ratio_max": self.rise_ratio_max,
            "min_drop": self.min_drop,
            "atr_buffer": self.atr_buffer,
            "validation_distance": self.validation_distance,
            "lookback_bars": self.lookback_bars,
        }


@dataclass
class CandleComposition:
    """Defines the mix of candle timeframes for model input."""
    candles_1m: int = 30    # Number of 1-minute candles
    candles_3m: int = 20    # Number of 3-minute candles  
    candles_5m: int = 10    # Number of 5-minute candles
    candles_15m: int = 0    # Number of 15-minute candles
    
    @property
    def total_features(self) -> int:
        """Total number of candle input features."""
        return (self.candles_1m + self.candles_3m + 
                self.candles_5m + self.candles_15m) * 4  # OHLC
    
    @property
    def label(self) -> str:
        """Human-readable label for this composition."""
        parts = []
        if self.candles_1m: parts.append(f"{self.candles_1m}x1m")
        if self.candles_3m: parts.append(f"{self.candles_3m}x3m")
        if self.candles_5m: parts.append(f"{self.candles_5m}x5m")
        if self.candles_15m: parts.append(f"{self.candles_15m}x15m")
        return "+".join(parts) if parts else "empty"
    
    def to_cli_args(self) -> List[str]:
        return [
            "--candles-1m", str(self.candles_1m),
            "--candles-3m", str(self.candles_3m),
            "--candles-5m", str(self.candles_5m),
            "--candles-15m", str(self.candles_15m),
        ]


@dataclass
class OCOBracketConfig:
    """Configuration for OCO (One-Cancels-Other) bracket testing."""
    direction: str = "SHORT"          # 'LONG', 'SHORT', 'BOTH'
    r_multiple: float = 1.4           # Take profit as multiple of risk
    stop_atr_pct: float = 0.5         # Stop distance as % of ATR
    stop_type: str = "WICK"           # 'WICK', 'OPEN', 'ATR'
    
    # Unique identifier
    config_id: str = ""
    
    @property
    def label(self) -> str:
        return f"{self.direction}_{self.r_multiple}R_{self.stop_type}_{int(self.stop_atr_pct*100)}atr"
    
    def to_dict(self) -> dict:
        return {
            "config_id": self.config_id,
            "direction": self.direction,
            "r_multiple": self.r_multiple,
            "stop_atr_pct": self.stop_atr_pct,
            "stop_type": self.stop_type,
        }


@dataclass
class ModelSweepConfig:
    """Configuration for model architecture and training."""
    architecture: str = "CNN_Classic"  # 'CNN_Classic', 'CNN_Wide', 'LSTM', 'MLP'
    epochs: int = 10
    learning_rate: float = 0.001
    batch_size: int = 32
    dropout: float = 0.3
    
    # Input configuration
    candle_composition: CandleComposition = field(default_factory=CandleComposition)
    
    # Unique identifier
    config_id: str = ""
    
    def to_cli_args(self) -> List[str]:
        args = [
            "--architecture", self.architecture,
            "--epochs", str(self.epochs),
            "--lr", str(self.learning_rate),
            "--batch-size", str(self.batch_size),
            "--dropout", str(self.dropout),
        ]
        args.extend(self.candle_composition.to_cli_args())
        return args
    
    def to_dict(self) -> dict:
        return {
            "config_id": self.config_id,
            "architecture": self.architecture,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "dropout": self.dropout,
            "candle_composition": self.candle_composition.label,
        }


# ============================================================
# Default Sweep Ranges
# ============================================================

PATTERN_SWEEP_RANGES = {
    "rise_ratio_min": (1.5, 3.5),
    "rise_ratio_max": (2.5, 6.0),
    "min_drop": (0.5, 2.0),
    "atr_buffer": (0.1, 0.5),
    "validation_distance": (0.5, 2.0),
    "lookback_bars": (60, 180),
}

OCO_SWEEP_VALUES = {
    "direction": ["LONG", "SHORT"],
    "r_multiple": [1.0, 1.4, 1.8, 2.0, 2.5, 3.0],
    "stop_atr_pct": [0.25, 0.5, 0.75, 1.0],
    "stop_type": ["WICK", "OPEN", "ATR"],
}

MODEL_ARCHITECTURES = ["CNN_Classic", "CNN_Wide", "LSTM_Seq", "Feature_MLP"]

# Pre-defined candle compositions to sweep
CANDLE_COMPOSITIONS = [
    CandleComposition(30, 0, 0, 0),      # Pure 30x1m
    CandleComposition(60, 0, 0, 0),      # Pure 60x1m
    CandleComposition(20, 0, 0, 0),      # Minimal 20x1m
    CandleComposition(30, 20, 10, 0),    # Mixed: 30x1m + 20x3m + 10x5m
    CandleComposition(20, 10, 5, 0),     # Light mixed
    CandleComposition(40, 10, 0, 0),     # Heavy 1m + some 3m
]

```

### scripts/sweep/oco_tester.py

```python
"""
OCO Bracket Tester for Sweep Pipeline
Tests multiple OCO configurations on labeled pattern data.

Usage:
    python src/sweep/oco_tester.py \
        --pattern-data labeled_sweep_001.parquet \
        --model-path models/cnn_sweep.pth \
        --output results/oco_results.csv
"""

import argparse
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import sys
import json
from typing import List, Dict

# Add project root
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import PROCESSED_DIR, ONE_MIN_PARQUET_DIR, MODELS_DIR
from src.utils.logging_utils import get_logger
from src.sweep.config import OCOBracketConfig
from src.sweep.param_grid import get_default_oco_scenarios

logger = get_logger("oco_tester")


def parse_args():
    parser = argparse.ArgumentParser(description="OCO Bracket Tester")
    
    parser.add_argument("--pattern-data", type=str, required=True,
                        help="Path to labeled pattern data (parquet)")
    parser.add_argument("--model-path", type=str, default="",
                        help="Path to trained model (optional, for filtering)")
    parser.add_argument("--output", type=str, default="",
                        help="Output CSV path")
    
    # OCO override params (optional - uses defaults if not specified)
    parser.add_argument("--direction", type=str, default="",
                        choices=["", "LONG", "SHORT"])
    parser.add_argument("--r-mult", type=float, default=0)
    parser.add_argument("--stop-atr-pct", type=float, default=0)
    parser.add_argument("--stop-type", type=str, default="",
                        choices=["", "WICK", "OPEN", "ATR"])
    
    parser.add_argument("--use-defaults", action="store_true",
                        help="Use 10 default OCO scenarios")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only output stats")
    
    # Money management
    parser.add_argument("--risk-per-trade", type=float, default=75.0)
    parser.add_argument("--starting-balance", type=float, default=2000.0)
    
    return parser.parse_args()


def load_model(model_path: str):
    """Load trained model for signal filtering (optional)."""
    if not model_path or not Path(model_path).exists():
        return None
    
    # Import model architecture
    from src.models.cnn_model import TradeCNN
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TradeCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    logger.info(f"Loaded model from {model_path}")
    return model


def run_oco_backtest(
    patterns: pd.DataFrame,
    df_1m: pd.DataFrame,
    oco_config: OCOBracketConfig,
    risk_per_trade: float = 75.0,
    starting_balance: float = 2000.0,
) -> Dict:
    """
    Run backtest with specific OCO configuration.
    
    Returns:
        Dict with results: trades, win_rate, pnl, etc.
    """
    results = []
    balance = starting_balance
    
    for idx, pattern in patterns.iterrows():
        # Skip inconclusive
        original_outcome = pattern.get('outcome', '')
        if original_outcome == 'Inconclusive':
            continue
        
        trigger_time = pattern['trigger_time']
        
        # Ensure proper timezone for comparison
        if pd.Timestamp(trigger_time).tz is None:
             trigger_time = pd.Timestamp(trigger_time).tz_localize('UTC')
        else:
             trigger_time = pd.Timestamp(trigger_time).tz_convert('UTC')
             
        entry_price = pattern['entry']
        atr = pattern.get('atr', 1.0)
        
        # Determine direction based on config
        direction = oco_config.direction
        if direction == "BOTH":
            # Use original pattern direction if available
            direction = pattern.get('direction', 'SHORT')
        
        # Calculate stop based on stop_type
        if oco_config.stop_type == "ATR":
            stop_dist = atr * oco_config.stop_atr_pct
        elif oco_config.stop_type == "WICK":
            # Use pattern's stop (wick-based) if available
            if 'stop' in pattern and pd.notna(pattern.get('stop')):
                stop_dist = abs(pattern['stop'] - entry_price)
            else:
                # Fallback to ATR
                stop_dist = atr * oco_config.stop_atr_pct
        else:  # OPEN
            stop_dist = atr * 0.5  # Default to 0.5 ATR
        
        if stop_dist <= 0:
            stop_dist = atr * 0.5
        
        # Calculate TP based on R-multiple
        tp_dist = stop_dist * oco_config.r_multiple
        
        if direction == "SHORT":
            stop_price = entry_price + stop_dist
            tp_price = entry_price - tp_dist
        else:  # LONG
            stop_price = entry_price - stop_dist
            tp_price = entry_price + tp_dist
        
        # Simulate outcome using 1m data
        future = df_1m[df_1m.index > trigger_time]
        if len(future) == 0:
            continue
        
        # Limit search window
        future = future.iloc[:2000]
        
        highs = future['high'].values
        lows = future['low'].values
        times = future.index.values
        
        if direction == "SHORT":
            mask_win = lows <= tp_price
            mask_loss = highs >= stop_price
        else:
            mask_win = highs >= tp_price
            mask_loss = lows <= stop_price
        
        idx_win = np.argmax(mask_win) if mask_win.any() else 999999
        idx_loss = np.argmax(mask_loss) if mask_loss.any() else 999999
        
        if idx_win == 999999 and idx_loss == 999999:
            outcome = 'TIMEOUT'
            pnl = 0
        elif idx_win < idx_loss:
            outcome = 'WIN'
            pnl = risk_per_trade * oco_config.r_multiple
        else:
            outcome = 'LOSS'
            pnl = -risk_per_trade
        
        balance += pnl
        
        results.append({
            'trigger_time': trigger_time,
            'direction': direction,
            'entry': entry_price,
            'stop': stop_price,
            'tp': tp_price,
            'outcome': outcome,
            'pnl': pnl,
            'balance': balance,
            'oco_config': oco_config.label,
        })
    
    # Calculate summary stats
    if results:
        df_results = pd.DataFrame(results)
        valid_trades = df_results[df_results['outcome'].isin(['WIN', 'LOSS'])]
        wins = len(valid_trades[valid_trades['outcome'] == 'WIN'])
        losses = len(valid_trades[valid_trades['outcome'] == 'LOSS'])
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
        total_pnl = df_results['pnl'].sum()
        
        # Expected value per trade
        avg_win = risk_per_trade * oco_config.r_multiple
        avg_loss = risk_per_trade
        ev = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        summary = {
            "oco_config": oco_config.label,
            "direction": oco_config.direction,
            "r_multiple": oco_config.r_multiple,
            "stop_type": oco_config.stop_type,
            "stop_atr_pct": oco_config.stop_atr_pct,
            "total_trades": len(valid_trades),
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 4),
            "total_pnl": round(total_pnl, 2),
            "final_balance": round(balance, 2),
            "expected_value": round(ev, 2),
        }
    else:
        df_results = pd.DataFrame()
        summary = {"oco_config": oco_config.label, "error": "No trades"}
    
    return {
        "trades": df_results,
        "summary": summary,
    }


def main():
    args = parse_args()
    
    # Ensure CUDA is available
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("No GPU detected, running on CPU")
    
    # Load pattern data
    pattern_path = Path(args.pattern_data)
    if not pattern_path.is_absolute():
        pattern_path = PROCESSED_DIR / args.pattern_data
    
    if not pattern_path.exists():
        logger.error(f"Pattern data not found: {pattern_path}")
        return
    
    patterns = pd.read_parquet(pattern_path)
    logger.info(f"Loaded {len(patterns)} patterns from {pattern_path}")
    
    # Load 1m data for simulation
    data_path = PROCESSED_DIR / "continuous_1m.parquet"
    if not data_path.exists():
        data_path = ONE_MIN_PARQUET_DIR / "mes_1min_all.parquet"
    
    df_1m = pd.read_parquet(data_path)
    if 'time' in df_1m.columns:
        df_1m['time'] = pd.to_datetime(df_1m['time'], utc=True)
        df_1m = df_1m.set_index('time')
    df_1m = df_1m.sort_index()
    
    # Determine OCO configs to test
    if args.use_defaults:
        oco_configs = get_default_oco_scenarios()
        logger.info(f"Using {len(oco_configs)} default OCO scenarios")
    elif args.direction and args.r_mult > 0:
        # Single custom config
        oco_configs = [OCOBracketConfig(
            direction=args.direction,
            r_multiple=args.r_mult,
            stop_atr_pct=args.stop_atr_pct or 0.5,
            stop_type=args.stop_type or "ATR",
            config_id="custom",
        )]
    else:
        # Default 10 scenarios
        oco_configs = get_default_oco_scenarios()
    
    # Run backtests for each OCO config
    all_summaries = []
    all_trades = []
    
    for oco_config in oco_configs:
        logger.info(f"Testing OCO: {oco_config.label}")
        
        result = run_oco_backtest(
            patterns=patterns,
            df_1m=df_1m,
            oco_config=oco_config,
            risk_per_trade=args.risk_per_trade,
            starting_balance=args.starting_balance,
        )
        
        summary = result["summary"]
        trades = result["trades"]
        
        all_summaries.append(summary)
        if not trades.empty:
            all_trades.append(trades)
        
        logger.info(f"  -> Trades: {summary.get('total_trades', 0)}, "
                    f"Win Rate: {summary.get('win_rate', 0)*100:.1f}%, "
                    f"PnL: ${summary.get('total_pnl', 0):.2f}")
    
    # Output results
    logger.info("=" * 60)
    logger.info("OCO SWEEP RESULTS")
    logger.info("=" * 60)
    
    df_summary = pd.DataFrame(all_summaries)
    print(df_summary.to_string(index=False))
    
    if not args.dry_run and args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_summary.to_csv(out_path, index=False)
        logger.info(f"Results saved to {out_path}")
        
        # Also save detailed trades
        if all_trades:
            trades_path = out_path.parent / f"{out_path.stem}_trades.parquet"
            pd.concat(all_trades).to_parquet(trades_path)
            logger.info(f"Detailed trades saved to {trades_path}")
    
    # Print JSON for orchestrator
    print(json.dumps(all_summaries, indent=2))
    return all_summaries


if __name__ == "__main__":
    main()

```

### scripts/sweep/param_grid.py

```python
"""
Parameter Grid Generator
Creates sweep configurations for pattern mining, OCO brackets, and models.
"""

import numpy as np
from typing import List
import itertools

from .config import (
    PatternSweepConfig, 
    OCOBracketConfig, 
    ModelSweepConfig,
    CandleComposition,
    PATTERN_SWEEP_RANGES,
    OCO_SWEEP_VALUES,
    MODEL_ARCHITECTURES,
    CANDLE_COMPOSITIONS,
)


def generate_pattern_grid(n: int = 33, seed: int = 42) -> List[PatternSweepConfig]:
    """
    Generate N pattern configurations via Latin Hypercube Sampling.
    Default 33 configs × 30 triggers = ~1000 pattern evaluations.
    
    Args:
        n: Number of configurations to generate
        seed: Random seed for reproducibility
        
    Returns:
        List of PatternSweepConfig objects
    """
    np.random.seed(seed)
    configs = []
    
    ranges = PATTERN_SWEEP_RANGES
    
    for i in range(n):
        # Sample each parameter uniformly within range
        rise_min = np.random.uniform(*ranges["rise_ratio_min"])
        rise_max = np.random.uniform(rise_min + 0.5, ranges["rise_ratio_max"][1])  # Ensure max > min
        
        config = PatternSweepConfig(
            rise_ratio_min=round(rise_min, 2),
            rise_ratio_max=round(rise_max, 2),
            min_drop=round(np.random.uniform(*ranges["min_drop"]), 2),
            atr_buffer=round(np.random.uniform(*ranges["atr_buffer"]), 2),
            validation_distance=round(np.random.uniform(*ranges["validation_distance"]), 2),
            lookback_bars=int(np.random.uniform(*ranges["lookback_bars"])),
            config_id=f"pattern_{i:03d}",
        )
        configs.append(config)
    
    return configs


def generate_oco_grid(n: int = 33, seed: int = 42) -> List[OCOBracketConfig]:
    """
    Generate N OCO bracket configurations.
    Uses combination of grid + random for diversity.
    
    Args:
        n: Number of configurations to generate
        seed: Random seed for reproducibility
        
    Returns:
        List of OCOBracketConfig objects
    """
    np.random.seed(seed)
    configs = []
    
    vals = OCO_SWEEP_VALUES
    
    # Generate all combinations
    all_combos = list(itertools.product(
        vals["direction"],
        vals["r_multiple"],
        vals["stop_atr_pct"],
        vals["stop_type"],
    ))
    
    # If we need fewer than total combos, sample randomly
    if n < len(all_combos):
        indices = np.random.choice(len(all_combos), size=n, replace=False)
        selected = [all_combos[i] for i in indices]
    else:
        selected = all_combos[:n]
    
    for i, (direction, r_mult, stop_pct, stop_type) in enumerate(selected):
        config = OCOBracketConfig(
            direction=direction,
            r_multiple=r_mult,
            stop_atr_pct=stop_pct,
            stop_type=stop_type,
            config_id=f"oco_{i:03d}",
        )
        configs.append(config)
    
    return configs


def generate_model_grid(
    architectures: List[str] = None,
    candle_compositions: List[CandleComposition] = None,
) -> List[ModelSweepConfig]:
    """
    Generate model configurations for each architecture × candle composition.
    
    Returns:
        List of ModelSweepConfig objects
    """
    if architectures is None:
        architectures = MODEL_ARCHITECTURES
    if candle_compositions is None:
        candle_compositions = CANDLE_COMPOSITIONS
    
    configs = []
    idx = 0
    
    for arch in architectures:
        for candle_comp in candle_compositions:
            # Adjust seq_len based on architecture
            if arch == "CNN_Wide" and candle_comp.candles_1m < 60:
                # Skip wide CNN for small inputs
                continue
                
            config = ModelSweepConfig(
                architecture=arch,
                epochs=10,
                learning_rate=0.001,
                batch_size=32,
                dropout=0.3,
                candle_composition=candle_comp,
                config_id=f"model_{idx:03d}",
            )
            configs.append(config)
            idx += 1
    
    return configs


def get_default_oco_scenarios() -> List[OCOBracketConfig]:
    """
    Get the 10 default OCO scenarios for test phase evaluation.
    Every model test gets these 10 results.
    """
    return [
        # Long scenarios
        OCOBracketConfig("LONG", 1.0, 0.50, "ATR", "default_01"),
        OCOBracketConfig("LONG", 1.4, 0.50, "ATR", "default_02"),
        OCOBracketConfig("LONG", 2.0, 0.50, "WICK", "default_03"),
        OCOBracketConfig("LONG", 1.4, 0.25, "WICK", "default_04"),
        # Short scenarios
        OCOBracketConfig("SHORT", 1.0, 0.50, "ATR", "default_05"),
        OCOBracketConfig("SHORT", 1.4, 0.50, "ATR", "default_06"),
        OCOBracketConfig("SHORT", 2.0, 0.50, "WICK", "default_07"),
        OCOBracketConfig("SHORT", 1.4, 0.25, "WICK", "default_08"),
        # Hybrid scenarios
        OCOBracketConfig("LONG", 1.8, 0.75, "ATR", "default_09"),
        OCOBracketConfig("SHORT", 1.8, 0.75, "ATR", "default_10"),
    ]


if __name__ == "__main__":
    # Quick test
    print("Pattern Configs:", len(generate_pattern_grid(33)))
    print("OCO Configs:", len(generate_oco_grid(33)))
    print("Model Configs:", len(generate_model_grid()))
    print("Default OCO Scenarios:", len(get_default_oco_scenarios()))

```

### scripts/sweep/pattern_miner_v2.py

```python
"""
Pattern Miner V2 - Proportional Detection
Finds patterns where price rises X times a unit, then returns back.
All measurements are RATIOS, not dollar amounts.

Usage:
    python src/sweep/pattern_miner_v2.py \
        --rise-ratio 1.5 --return-ratio 1.0 --invalid-ratio 2.5 \
        --max-triggers 30 --output-suffix "config_001"
"""

import argparse
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import sys
import json
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import PROCESSED_DIR, ONE_MIN_PARQUET_DIR
from src.utils.logging_utils import get_logger

logger = get_logger("pattern_miner_v2")

# Enforce GPU
if not torch.cuda.is_available():
    logger.error("GPU NOT DETECTED! This script requires CUDA.")
    sys.exit(1)
device = torch.device("cuda")
logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

# 10 OCO configs: 5 LONG + 5 SHORT
OCO_CONFIGS = [
    {"name": "LONG_1.0R", "direction": "LONG", "r_mult": 1.0},
    {"name": "LONG_1.4R", "direction": "LONG", "r_mult": 1.4},
    {"name": "LONG_2.0R", "direction": "LONG", "r_mult": 2.0},
    {"name": "LONG_1.8R", "direction": "LONG", "r_mult": 1.8},
    {"name": "LONG_2.5R", "direction": "LONG", "r_mult": 2.5},
    {"name": "SHORT_1.0R", "direction": "SHORT", "r_mult": 1.0},
    {"name": "SHORT_1.4R", "direction": "SHORT", "r_mult": 1.4},
    {"name": "SHORT_2.0R", "direction": "SHORT", "r_mult": 2.0},
    {"name": "SHORT_1.8R", "direction": "SHORT", "r_mult": 1.8},
    {"name": "SHORT_2.5R", "direction": "SHORT", "r_mult": 2.5},
]


def parse_args():
    parser = argparse.ArgumentParser(description="Proportional Pattern Miner V2")
    
    # Pattern ratios (proportional, not dollars)
    parser.add_argument("--rise-ratio", type=float, default=1.5,
                        help="Rise as multiple of unit move (e.g., 1.5x)")
    parser.add_argument("--return-ratio", type=float, default=1.0,
                        help="Return as multiple of unit (trigger at -1x)")
    parser.add_argument("--invalid-ratio", type=float, default=2.5,
                        help="Invalidation level (if hit before return)")
    parser.add_argument("--lookback", type=int, default=60,
                        help="Bars to look back for pattern start")
    parser.add_argument("--min-unit", type=float, default=0.5,
                        help="Minimum unit size in points (filter noise)")
    
    # Output
    parser.add_argument("--output-suffix", type=str, default="")
    parser.add_argument("--max-triggers", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    
    return parser.parse_args()


def simulate_oco(df_1m, trigger_idx, entry_price, stop_price, oco_config):
    """Simulate single OCO outcome. Stop is candle BEFORE move."""
    direction = oco_config["direction"]
    r_mult = oco_config["r_mult"]
    
    risk = abs(entry_price - stop_price)
    if risk <= 0:
        return "INVALID", 0
    
    if direction == "LONG":
        tp_price = entry_price + (risk * r_mult)
        future = df_1m.iloc[trigger_idx+1:trigger_idx+2001]
        if len(future) == 0:
            return "TIMEOUT", 0
        
        sl_hit = future[future['low'] <= stop_price]
        tp_hit = future[future['high'] >= tp_price]
    else:  # SHORT
        tp_price = entry_price - (risk * r_mult)
        future = df_1m.iloc[trigger_idx+1:trigger_idx+2001]
        if len(future) == 0:
            return "TIMEOUT", 0
        
        sl_hit = future[future['high'] >= stop_price]
        tp_hit = future[future['low'] <= tp_price]
    
    sl_idx = sl_hit.index[0] if not sl_hit.empty else 999999999
    tp_idx = tp_hit.index[0] if not tp_hit.empty else 999999999
    
    if sl_idx == 999999999 and tp_idx == 999999999:
        return "TIMEOUT", 0
    elif tp_idx < sl_idx:
        return "WIN", r_mult
    else:
        return "LOSS", -1.0


def mine_proportional_patterns(args):
    """
    Mine patterns using proportional ratios.
    Pattern: price rises X times unit, returns to -1x (before hitting invalid level).
    Stop: close of candle BEFORE the move started.
    """
    logger.info(f"Mining: rise={args.rise_ratio}x, return={args.return_ratio}x, "
                f"invalid={args.invalid_ratio}x")
    
    # Load data
    data_path = PROCESSED_DIR / "continuous_1m.parquet"
    if not data_path.exists():
        data_path = ONE_MIN_PARQUET_DIR / "mes_1min_all.parquet"
    
    if not data_path.exists():
        return {"error": "No data found"}
    
    df_1m = pd.read_parquet(data_path)
    
    if isinstance(df_1m.index, pd.DatetimeIndex) or 'time' not in df_1m.columns:
        df_1m = df_1m.reset_index()
    
    time_cols = [c for c in df_1m.columns if 'time' in c.lower() or c == 'index']
    if time_cols:
        df_1m = df_1m.rename(columns={time_cols[0]: 'time'})
    
    df_1m = df_1m.sort_values('time').reset_index(drop=True)
    df_1m['time'] = pd.to_datetime(df_1m['time'], utc=True)
    
    # Arrays for speed
    closes = df_1m['close'].values
    highs = df_1m['high'].values
    lows = df_1m['low'].values
    times = df_1m['time'].values
    n = len(df_1m)
    
    # Track results
    oco_results = {cfg["name"]: {"wins": 0, "losses": 0, "pnl": 0.0} 
                   for cfg in OCO_CONFIGS}
    
    patterns_data = []
    pattern_count = 0
    last_trigger = times[0] - np.timedelta64(1, 'D')
    max_triggers = args.max_triggers if args.max_triggers > 0 else float('inf')
    
    logger.info(f"Scanning {n} bars...")
    
    for i in range(100, n - 100):
        if pattern_count >= max_triggers:
            break
        
        curr_time = times[i]
        curr_price = closes[i]
        
        # 15min cooldown
        if (curr_time - last_trigger) < np.timedelta64(15, 'm'):
            continue
        
        # Look for pattern start
        for j in range(i - 1, max(0, i - args.lookback), -1):
            start_price = closes[j]
            
            # ========== SHORT PATTERN: Price UP then returns ==========
            peak_price = np.max(highs[j:i+1])
            peak_idx = j + np.argmax(highs[j:i+1])
            
            drop = peak_price - curr_price
            rise = peak_price - start_price
            
            if drop >= args.min_unit and rise >= args.min_unit:
                ratio = rise / drop
                if args.rise_ratio <= ratio <= args.invalid_ratio:
                    if j >= 1:
                        stop_price = closes[j - 1]
                        entry_price = curr_price
                        
                        pattern_info = {
                            'trigger_idx': i,
                            'trigger_time': curr_time,
                            'start_idx': j,
                            'peak_idx': peak_idx,
                            'entry': entry_price,
                            'stop': stop_price,
                            'unit': drop,
                            'rise': rise,
                            'ratio': ratio,
                            'peak': peak_price,
                            'direction': 'SHORT',  # This is a SHORT setup
                        }
                        
                        for oco_cfg in OCO_CONFIGS:
                            outcome, pnl_r = simulate_oco(df_1m, i, entry_price, stop_price, oco_cfg)
                            if outcome == "WIN":
                                oco_results[oco_cfg["name"]]["wins"] += 1
                                oco_results[oco_cfg["name"]]["pnl"] += pnl_r
                            elif outcome == "LOSS":
                                oco_results[oco_cfg["name"]]["losses"] += 1
                                oco_results[oco_cfg["name"]]["pnl"] += pnl_r
                            pattern_info[f"outcome_{oco_cfg['name']}"] = outcome
                        
                        patterns_data.append(pattern_info)
                        pattern_count += 1
                        last_trigger = curr_time
                        break
            
            # ========== LONG PATTERN: Price DOWN then returns ==========
            trough_price = np.min(lows[j:i+1])
            trough_idx = j + np.argmin(lows[j:i+1])
            
            rise_back = curr_price - trough_price
            fall = start_price - trough_price
            
            if rise_back >= args.min_unit and fall >= args.min_unit:
                ratio = fall / rise_back
                if args.rise_ratio <= ratio <= args.invalid_ratio:
                    if j >= 1:
                        stop_price = closes[j - 1]
                        entry_price = curr_price
                        
                        pattern_info = {
                            'trigger_idx': i,
                            'trigger_time': curr_time,
                            'start_idx': j,
                            'peak_idx': trough_idx,  # Actually trough for LONG
                            'entry': entry_price,
                            'stop': stop_price,
                            'unit': rise_back,
                            'rise': fall,
                            'ratio': ratio,
                            'peak': trough_price,  # Actually trough for LONG
                            'direction': 'LONG',  # This is a LONG setup
                        }
                        
                        for oco_cfg in OCO_CONFIGS:
                            outcome, pnl_r = simulate_oco(df_1m, i, entry_price, stop_price, oco_cfg)
                            if outcome == "WIN":
                                oco_results[oco_cfg["name"]]["wins"] += 1
                                oco_results[oco_cfg["name"]]["pnl"] += pnl_r
                            elif outcome == "LOSS":
                                oco_results[oco_cfg["name"]]["losses"] += 1
                                oco_results[oco_cfg["name"]]["pnl"] += pnl_r
                            pattern_info[f"outcome_{oco_cfg['name']}"] = outcome
                        
                        patterns_data.append(pattern_info)
                        pattern_count += 1
                        last_trigger = curr_time
                        break
    
    # Calculate stats
    oco_stats = []
    for cfg in OCO_CONFIGS:
        name = cfg["name"]
        data = oco_results[name]
        total = data["wins"] + data["losses"]
        win_rate = data["wins"] / total if total > 0 else 0
        
        oco_stats.append({
            "oco_config": name,
            "direction": cfg["direction"],
            "r_mult": cfg["r_mult"],
            "wins": data["wins"],
            "losses": data["losses"],
            "total": total,
            "win_rate": round(win_rate, 4),
            "total_pnl_r": round(data["pnl"], 2),
            "ev_per_trade": round(data["pnl"] / total, 3) if total > 0 else 0,
        })
    
    oco_stats = sorted(oco_stats, key=lambda x: x["ev_per_trade"], reverse=True)
    
    summary = {
        "pattern_config": {
            "rise_ratio": args.rise_ratio,
            "return_ratio": args.return_ratio,
            "invalid_ratio": args.invalid_ratio,
            "min_unit": args.min_unit,
        },
        "total_patterns": pattern_count,
        "oco_results": oco_stats,
        "best_oco": oco_stats[0] if oco_stats else None,
    }
    
    return {
        "summary": summary,
        "patterns": pd.DataFrame(patterns_data) if patterns_data else pd.DataFrame(),
    }


def main():
    args = parse_args()
    result = mine_proportional_patterns(args)
    
    if "error" in result:
        print(json.dumps(result))
        return
    
    summary = result["summary"]
    patterns_df = result["patterns"]
    
    logger.info("=" * 60)
    logger.info(f"Mining Complete! Found {summary['total_patterns']} patterns")
    logger.info("Top 5 OCO configs by EV:")
    for oco in summary["oco_results"][:5]:
        logger.info(f"  {oco['oco_config']}: WR={oco['win_rate']*100:.1f}%, "
                    f"EV={oco['ev_per_trade']:.3f}R")
    logger.info("=" * 60)
    
    if not args.dry_run and len(patterns_df) > 0:
        suffix = args.output_suffix or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        out_path = PROCESSED_DIR / f"patterns_v2_{suffix}.parquet"
        patterns_df.to_parquet(out_path)
        
        stats_path = PROCESSED_DIR / f"patterns_v2_{suffix}_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved to {out_path}")
    
    print(json.dumps(summary))
    return summary


if __name__ == "__main__":
    main()

```

### scripts/sweep/run_sweep_integrated.py

```python
"""
Sweep Runner - Integrated with mlang2 Architecture

This wrapper connects the sweep tools from mlang to:
- ExperimentDB for storing results
- FeatureEngine for consistent normalization
- ModelRegistry for model management

Usage:
    python scripts/sweep/run_sweep_integrated.py --help
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
from datetime import datetime
from typing import Dict, Any, List

# Sweep configs
from scripts.sweep.config import (
    PatternSweepConfig, 
    CandleComposition, 
    OCOBracketConfig,
    ModelSweepConfig,
    CANDLE_COMPOSITIONS,
    OCO_SWEEP_VALUES,
)

# mlang2 integrations
from src.storage import ExperimentDB
from src.features.engine import normalize_ohlcv_window, FeatureConfig


def run_sweep_variant(
    pattern_config: PatternSweepConfig,
    oco_config: OCOBracketConfig,
    model_config: ModelSweepConfig,
    data_path: str = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run a single sweep variant and return results.
    
    Args:
        pattern_config: Pattern mining geometry
        oco_config: OCO bracket configuration
        model_config: Model architecture settings
        data_path: Path to market data (uses default if None)
        verbose: Print progress
    
    Returns:
        Dict with trades, win_rate, pnl, config
    """
    import numpy as np
    import yfinance as yf
    from datetime import datetime, timedelta
    
    # Load data via yfinance (max 7 days for 1m data)
    end = datetime.now()
    start = end - timedelta(days=7)
    
    try:
        ticker = yf.Ticker("ES=F")  # MES proxy
        df = ticker.history(start=start, end=end, interval="1m")
    except Exception as e:
        print(f"Failed to load data: {e}")
        return {'total_trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0, 'total_pnl': 0, 'config': {}}
    
    if df is None or len(df) == 0:
        print("No market data available - market may be closed!")
        return {'total_trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0, 'total_pnl': 0, 'config': {}}
    
    # Standardize columns
    df = df.rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low', 
        'Close': 'close', 'Volume': 'volume'
    })
    df = df.reset_index()
    
    if verbose:
        print(f"Loaded {len(df)} bars via yfinance")
    
    # Create feature config from candle composition
    lookback = model_config.candle_composition.candles_1m
    feature_config = FeatureConfig(lookback=lookback)
    
    # Simulate trades based on pattern config
    trades = []
    wins = 0
    total_pnl = 0.0
    
    # Simplified trading simulation
    atr_period = 14
    
    for i in range(lookback + atr_period, len(df) - 10):
        # Calculate ATR
        highs = df['high'].iloc[i-atr_period:i].values
        lows = df['low'].iloc[i-atr_period:i].values
        closes = df['close'].iloc[i-atr_period:i].values
        
        tr = np.maximum(highs - lows, 
                        np.maximum(np.abs(highs - np.roll(closes, 1)),
                                  np.abs(lows - np.roll(closes, 1))))
        atr = np.mean(tr[1:])
        
        if atr < 0.5:
            continue
        
        # Get normalized window
        window_data = df.iloc[i-lookback:i][['open', 'high', 'low', 'close', 'volume']].values
        x_norm = normalize_ohlcv_window(window_data, feature_config)
        
        # Simple trigger: large move followed by pullback
        recent_range = df['high'].iloc[i-5:i].max() - df['low'].iloc[i-5:i].min()
        if recent_range > atr * pattern_config.rise_ratio_min:
            # Trigger trade based on OCO config
            entry = df['close'].iloc[i]
            
            if oco_config.direction == "LONG":
                stop = entry - atr * oco_config.stop_atr_pct
                tp = entry + atr * oco_config.r_multiple * oco_config.stop_atr_pct
            else:
                stop = entry + atr * oco_config.stop_atr_pct
                tp = entry - atr * oco_config.r_multiple * oco_config.stop_atr_pct
            
            # Check outcome in next bars
            for j in range(i+1, min(i+50, len(df))):
                if oco_config.direction == "LONG":
                    if df['low'].iloc[j] <= stop:
                        pnl = -abs(entry - stop) * 50  # MES $50/pt
                        trades.append({'win': False, 'pnl': pnl})
                        total_pnl += pnl
                        break
                    elif df['high'].iloc[j] >= tp:
                        pnl = abs(tp - entry) * 50
                        wins += 1
                        trades.append({'win': True, 'pnl': pnl})
                        total_pnl += pnl
                        break
                else:
                    if df['high'].iloc[j] >= stop:
                        pnl = -abs(stop - entry) * 50
                        trades.append({'win': False, 'pnl': pnl})
                        total_pnl += pnl
                        break
                    elif df['low'].iloc[j] <= tp:
                        pnl = abs(entry - tp) * 50
                        wins += 1
                        trades.append({'win': True, 'pnl': pnl})
                        total_pnl += pnl
                        break
    
    win_rate = wins / len(trades) if trades else 0
    
    return {
        'total_trades': len(trades),
        'wins': wins,
        'losses': len(trades) - wins,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_pnl': total_pnl / len(trades) if trades else 0,
        'config': {
            'pattern': pattern_config.to_dict(),
            'oco': oco_config.to_dict(),
            'model': model_config.to_dict(),
        }
    }


def run_mini_sweep(directions: List[str] = None, r_multiples: List[float] = None) -> List[Dict]:
    """
    Run a small sweep over key parameters.
    
    Args:
        directions: List of directions to test ["LONG", "SHORT"]
        r_multiples: List of R multiples to test [1.0, 1.4, 2.0]
    
    Returns:
        List of results sorted by win_rate
    """
    if directions is None:
        directions = ["LONG", "SHORT"]
    if r_multiples is None:
        r_multiples = [1.0, 1.4, 2.0]
    
    db = ExperimentDB()
    results = []
    
    pattern_config = PatternSweepConfig(
        rise_ratio_min=1.5,
        lookback_bars=30,
    )
    
    model_config = ModelSweepConfig(
        candle_composition=CandleComposition(candles_1m=30)
    )
    
    print("=" * 60)
    print("MINI SWEEP - Testing configurations")
    print("=" * 60)
    
    for direction in directions:
        for r_mult in r_multiples:
            oco_config = OCOBracketConfig(
                direction=direction,
                r_multiple=r_mult,
                stop_atr_pct=0.5,
                config_id=f"{direction}_{r_mult}R"
            )
            
            print(f"\nTesting: {oco_config.config_id}...")
            
            result = run_sweep_variant(
                pattern_config=pattern_config,
                oco_config=oco_config,
                model_config=model_config,
                verbose=False
            )
            
            print(f"  Trades: {result['total_trades']} | "
                  f"WR: {result['win_rate']:.1%} | "
                  f"PnL: ${result['total_pnl']:.2f}")
            
            # Store to DB
            run_id = f"sweep_{oco_config.config_id}_{datetime.now().strftime('%H%M%S')}"
            db.store_run(
                run_id=run_id,
                strategy="sweep_test",
                config=result['config'],
                metrics={
                    'total_trades': result['total_trades'],
                    'wins': result['wins'],
                    'losses': result['losses'],
                    'win_rate': result['win_rate'],
                    'total_pnl': result['total_pnl'],
                }
            )
            
            results.append({
                'config_id': oco_config.config_id,
                **result
            })
    
    # Sort by win_rate
    results.sort(key=lambda x: x['win_rate'], reverse=True)
    
    print("\n" + "=" * 60)
    print("RESULTS (sorted by win rate)")
    print("=" * 60)
    for r in results:
        print(f"  {r['config_id']}: {r['win_rate']:.1%} WR, ${r['total_pnl']:.2f}")
    
    print(f"\nStored {len(results)} experiments to DB")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sweep with mlang2 integration")
    parser.add_argument("--directions", nargs="+", default=["LONG", "SHORT"])
    parser.add_argument("--r-multiples", nargs="+", type=float, default=[1.0, 1.4, 2.0])
    parser.add_argument("--quick", action="store_true", help="Run minimal sweep")
    
    args = parser.parse_args()
    
    if args.quick:
        results = run_mini_sweep(["LONG"], [1.0])
    else:
        results = run_mini_sweep(args.directions, args.r_multiples)
    
    print(f"\nBest config: {results[0]['config_id']} with {results[0]['win_rate']:.1%}")

```

### scripts/sweep/supersweep.py

```python
"""
SUPERSWEEP - Comprehensive Strategy Testing

Tests 30 OCO/limit/ATR configurations across all MES data with market context filters:
- Time of day, Day of week
- Above/below weekly VWAP
- 200 EMA on 5m, 15m
- PDH/PDL (previous day high/low)
- ONH/ONL (overnight high/low)
- Previous day close

Usage:
    python src/sweep/supersweep.py --output results/supersweep_results.parquet
"""

import argparse
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import sys
from itertools import product

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.variants import CNN_Classic
from src.config import PROCESSED_DIR, ONE_MIN_PARQUET_DIR, MODELS_DIR
from src.utils.logging_utils import get_logger

logger = get_logger("supersweep")

# GPU check
if not torch.cuda.is_available():
    logger.error("GPU NOT DETECTED!")
    sys.exit(1)
device = torch.device("cuda")
logger.info(f"GPU: {torch.cuda.get_device_name(0)}")


# ============ CONFIGURATION GRID ============

# Entry offsets (multiple of ATR)
ENTRY_OFFSETS = [0, 0.25, 0.5, 0.75, 1.0]

# ATR timeframes for stop calculation
ATR_TIMEFRAMES = ['5m', '15m']

# TP multiples
TP_MULTS = [1.0, 1.4, 2.0]

# 30 configurations: 5 offsets × 2 ATR × 3 TP = 30
def generate_configs():
    configs = []
    for offset, atr_tf, tp in product(ENTRY_OFFSETS, ATR_TIMEFRAMES, TP_MULTS):
        configs.append({
            'name': f'LONG_off{offset}_atr{atr_tf}_tp{tp}',
            'direction': 'LONG',
            'entry_offset': offset,
            'atr_tf': atr_tf,
            'tp_mult': tp,
        })
    return configs

CONFIGS = generate_configs()
logger.info(f"Generated {len(CONFIGS)} configurations")


# ============ HELPER FUNCTIONS ============

TICK = 0.25
PV = 5.0  # MES point value

def round_tick(p, d='n'):
    if d == 'u':
        return np.ceil(p / TICK) * TICK
    elif d == 'd':
        return np.floor(p / TICK) * TICK
    return round(p / TICK) * TICK


def calculate_vwap(df, period='W'):
    """Calculate VWAP for given period."""
    df = df.copy()
    df['typical'] = (df['high'] + df['low'] + df['close']) / 3
    df['tp_vol'] = df['typical'] * df['volume']
    
    if period == 'W':
        df['period'] = df['time_dt'].dt.isocalendar().week
    else:
        df['period'] = df['time_dt'].dt.date
    
    vwap = df.groupby('period').apply(
        lambda x: x['tp_vol'].cumsum() / x['volume'].cumsum()
    ).reset_index(level=0, drop=True)
    return vwap


def calculate_ema(series, period):
    """Calculate EMA."""
    return series.ewm(span=period, adjust=False).mean()


def get_session_levels(df, trigger_time):
    """Get PDH, PDL, PDC, ONH, ONL for given trigger time."""
    try:
        trigger_date = trigger_time.date()
        
        # Previous day
        prev_day = trigger_date - pd.Timedelta(days=1)
        while prev_day.weekday() >= 5:  # Skip weekends
            prev_day -= pd.Timedelta(days=1)
        
        # Convert to string for date comparison
        prev_day_str = str(prev_day)
        prev_day_data = df[df['time_dt'].dt.strftime('%Y-%m-%d') == prev_day_str]
        
        if len(prev_day_data) == 0:
            return None
        
        pdh = prev_day_data['high'].max()
        pdl = prev_day_data['low'].min()
        pdc = prev_day_data['close'].iloc[-1]
        
        # Overnight - simplified: just use prev day data
        onh = pdh
        onl = pdl
        
        return {
            'pdh': pdh, 'pdl': pdl, 'pdc': pdc,
            'onh': onh, 'onl': onl
        }
    except:
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="Supersweep Analysis")
    parser.add_argument("--output", type=str, default="results/supersweep_results.parquet")
    parser.add_argument("--risk", type=float, default=300.0)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--model", type=str, default="models/sweep_CNN_Classic_v3_bidirectional.pth")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load model
    model = CNN_Classic(input_dim=4, seq_len=20).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    logger.info(f"Model loaded: {args.model}")
    
    # Load data
    data_path = PROCESSED_DIR / "continuous_1m.parquet"
    if not data_path.exists():
        data_path = ONE_MIN_PARQUET_DIR / "mes_1min_all.parquet"
    
    df = pd.read_parquet(data_path)
    if 'time' in df.columns:
        df['time_dt'] = pd.to_datetime(df['time'], utc=True)
    elif 'time_dt' not in df.columns:
        df['time_dt'] = df.index
    df = df.sort_values('time_dt').reset_index(drop=True)
    logger.info(f"Loaded {len(df)} bars")
    
    # Resample for different ATR timeframes
    df_5m = df.set_index('time_dt').resample('5min').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()
    df_5m['tr'] = df_5m['high'] - df_5m['low']
    df_5m['atr'] = df_5m['tr'].rolling(14).mean()
    df_5m['ema200'] = calculate_ema(df_5m['close'], 200)
    
    df_15m = df.set_index('time_dt').resample('15min').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()
    df_15m['tr'] = df_15m['high'] - df_15m['low']
    df_15m['atr'] = df_15m['tr'].rolling(14).mean()
    df_15m['ema200'] = calculate_ema(df_15m['close'], 200)
    
    # Weekly VWAP on 1m data
    df['volume'] = df.get('volume', 1)  # Default volume if missing
    df['vwap'] = calculate_vwap(df, 'W')
    
    # Test portion (last 30%)
    n = len(df)
    test_start = int(n * 0.7)
    
    logger.info(f"Testing on {n - test_start} bars (last 30%)")
    
    all_trades = []
    last_i = 0
    trade_count = 0
    
    for i in range(test_start + 20, n - 200, 5):
        if i - last_i < 15:
            continue
        
        # CNN detection
        window = df.iloc[i-20:i][['open', 'high', 'low', 'close']].values
        mean, std = np.mean(window), np.std(window)
        if std == 0:
            std = 1.0
        feats = (window - mean) / std
        
        x = torch.FloatTensor(feats).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = model(x).item()
        
        if prob < args.threshold:
            continue
        
        last_i = i
        trigger_time = df.iloc[i]['time_dt']
        base_price = df.iloc[i]['close']
        
        # Get market context
        hour = trigger_time.hour
        day_of_week = trigger_time.dayofweek
        
        # VWAP
        vwap = df.iloc[i].get('vwap', base_price)
        above_vwap = base_price > vwap
        
        # EMA 200
        try:
            ema200_5m = df_5m.loc[:trigger_time]['ema200'].iloc[-1]
            above_ema200_5m = base_price > ema200_5m
        except:
            above_ema200_5m = None
        
        try:
            ema200_15m = df_15m.loc[:trigger_time]['ema200'].iloc[-1]
            above_ema200_15m = base_price > ema200_15m
        except:
            above_ema200_15m = None
        
        # Session levels
        levels = get_session_levels(df, trigger_time)
        if levels:
            above_pdh = base_price > levels['pdh']
            below_pdl = base_price < levels['pdl']
            above_pdc = base_price > levels['pdc']
            above_onh = base_price > levels['onh']
            below_onl = base_price < levels['onl']
        else:
            above_pdh = below_pdl = above_pdc = above_onh = below_onl = None
        
        # Get ATRs
        try:
            atr_5m = df_5m.loc[:trigger_time]['atr'].iloc[-1]
        except:
            continue
        try:
            atr_15m = df_15m.loc[:trigger_time]['atr'].iloc[-1]
        except:
            continue
        
        if pd.isna(atr_5m) or pd.isna(atr_15m):
            continue
        
        future = df.iloc[i+1:i+200]
        
        # Test each configuration
        for cfg in CONFIGS:
            atr = atr_5m if cfg['atr_tf'] == '5m' else atr_15m
            
            # Entry
            if cfg['entry_offset'] == 0:
                entry = base_price
                fill_bar = i
            else:
                limit = round_tick(base_price + cfg['entry_offset'] * atr, 'u')
                fills = future[future['high'] >= limit]
                if fills.empty:
                    continue
                entry = limit
                fill_bar = fills.index[0]
            
            # Stop and TP
            stop = round_tick(entry - atr, 'd')
            risk_dist = entry - stop
            if risk_dist <= 0:
                continue
            tp = round_tick(entry + risk_dist * cfg['tp_mult'], 'u')
            
            contracts = max(1, int(args.risk / (risk_dist * PV)))
            actual_risk = contracts * risk_dist * PV
            
            # Simulate
            tf = df.iloc[fill_bar+1:fill_bar+150]
            if len(tf) == 0:
                continue
            
            sl = tf[tf['low'] <= stop]
            tph = tf[tf['high'] >= tp]
            si = sl.index[0] if not sl.empty else 999999
            ti = tph.index[0] if not tph.empty else 999999
            
            if ti < si:
                outcome = 'WIN'
                pnl = contracts * risk_dist * cfg['tp_mult'] * PV
                exit_idx = ti
            elif si < 999999:
                outcome = 'LOSS'
                pnl = -actual_risk
                exit_idx = si
            else:
                outcome = 'TIMEOUT'
                pnl = 0
                exit_idx = tf.index[-1]
            
            duration = (df.iloc[exit_idx]['time_dt'] - df.iloc[fill_bar]['time_dt']).total_seconds() / 60
            mae = entry - tf['low'].min()
            
            trade = {
                'trigger_time': trigger_time,
                'config': cfg['name'],
                'entry_offset': cfg['entry_offset'],
                'atr_tf': cfg['atr_tf'],
                'tp_mult': cfg['tp_mult'],
                'entry': entry,
                'stop': stop,
                'tp': tp,
                'atr': atr,
                'contracts': contracts,
                'outcome': outcome,
                'pnl': pnl,
                'duration_mins': duration,
                'mae': mae,
                'hour': hour,
                'day_of_week': day_of_week,
                'above_vwap': above_vwap,
                'above_ema200_5m': above_ema200_5m,
                'above_ema200_15m': above_ema200_15m,
                'above_pdh': above_pdh,
                'below_pdl': below_pdl,
                'above_pdc': above_pdc,
                'above_onh': above_onh,
                'below_onl': below_onl,
            }
            all_trades.append(trade)
        
        trade_count += 1
        if trade_count % 100 == 0:
            logger.info(f"Processed {trade_count} triggers, {len(all_trades)} trade records...")
    
    # Save results
    results_df = pd.DataFrame(all_trades)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(output_path)
    
    logger.info(f"Saved {len(results_df)} trade records to {output_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUPERSWEEP SUMMARY")
    print("=" * 60)
    print(f"Total triggers: {trade_count}")
    print(f"Total trade records: {len(results_df)}")
    
    # Best configs
    print("\n=== TOP 10 CONFIGS BY WIN RATE ===")
    cfg_stats = results_df.groupby('config').agg({
        'outcome': lambda x: (x == 'WIN').sum(),
        'pnl': ['count', 'sum']
    })
    cfg_stats.columns = ['wins', 'total', 'pnl']
    cfg_stats['wr'] = cfg_stats['wins'] / cfg_stats['total']
    cfg_stats = cfg_stats[cfg_stats['total'] >= 50].sort_values('wr', ascending=False)
    
    for cfg in cfg_stats.head(10).itertuples():
        print(f"  {cfg.Index}: {cfg.wins}/{cfg.total} = {cfg.wr*100:.1f}% WR, ${cfg.pnl:+,.0f}")
    
    # Best filters
    print("\n=== FILTER ANALYSIS ===")
    for filter_col in ['above_vwap', 'above_ema200_5m', 'above_ema200_15m', 'above_pdc']:
        filtered = results_df[results_df[filter_col] == True]
        if len(filtered) > 50:
            wins = (filtered['outcome'] == 'WIN').sum()
            total = len(filtered[filtered['outcome'].isin(['WIN', 'LOSS'])])
            if total > 0:
                print(f"  {filter_col}=True: {wins}/{total} = {wins/total*100:.1f}% WR")


if __name__ == "__main__":
    main()

```

### scripts/sweep/train_sweep.py

```python
"""
CLI Training Script for Sweep Pipeline
Trains models with different architectures and candle compositions.

Usage:
    python src/sweep/train_sweep.py \
        --architecture CNN_Classic \
        --input-data labeled_sweep_001.parquet \
        --candles-1m 30 --candles-3m 20 --candles-5m 10 \
        --epochs 10 --lr 0.001 \
        --output-suffix "cnn_001"
"""

import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys
import json
from datetime import datetime

# Add project root
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import PROCESSED_DIR, ONE_MIN_PARQUET_DIR, MODELS_DIR
from src.utils.logging_utils import get_logger
from src.models.variants import CNN_Classic, CNN_Wide, LSTM_Seq, Feature_MLP

logger = get_logger("train_sweep")

# Enforce GPU
if not torch.cuda.is_available():
    logger.error("GPU NOT DETECTED! This script requires CUDA.")
    sys.exit(1)

device = torch.device("cuda")
logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")


class TradeDataset(Dataset):
    """Dataset for trade pattern training."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).reshape(-1, 1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def parse_args():
    parser = argparse.ArgumentParser(description="Model Training with CLI parameters")
    
    # Model architecture
    parser.add_argument("--architecture", type=str, default="CNN_Classic",
                        choices=["CNN_Classic", "CNN_Wide", "LSTM_Seq", "Feature_MLP"],
                        help="Model architecture to use")
    
    # Input data
    parser.add_argument("--input-data", type=str, required=True,
                        help="Path to labeled pattern data (parquet)")
    
    # Candle composition
    parser.add_argument("--candles-1m", type=int, default=30)
    parser.add_argument("--candles-3m", type=int, default=0)
    parser.add_argument("--candles-5m", type=int, default=0)
    parser.add_argument("--candles-15m", type=int, default=0)
    
    # Training params
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    
    # Split ratios
    parser.add_argument("--train-ratio", type=float, default=0.56)
    parser.add_argument("--val-ratio", type=float, default=0.14)
    
    # OCO config for labeling (use outcome column from sweep)
    parser.add_argument("--oco-config", type=str, default="SHORT_2.0R_50ATR",
                        help="OCO config to use for outcome labels (e.g. SHORT_2.0R_50ATR)")
    
    # Output
    parser.add_argument("--output-suffix", type=str, default="")
    parser.add_argument("--dry-run", action="store_true")
    
    return parser.parse_args()


def prepare_data(
    patterns: pd.DataFrame,
    df_1m: pd.DataFrame,
    window_size: int = 30,
    oco_config: str = "SHORT_2.0R_50ATR",
) -> tuple:
    """
    Prepare training data from patterns.
    Uses specified OCO config for outcome labels.
    
    Returns:
        X, y, timestamps
    """
    X = []
    y = []
    timestamps = []
    
    # Determine outcome column
    outcome_col = f"outcome_{oco_config}"
    if outcome_col not in patterns.columns:
        # Fallback to legacy 'outcome' column
        if 'outcome' in patterns.columns:
            outcome_col = 'outcome'
        else:
            logger.error(f"Outcome column not found: {outcome_col}")
            return np.array([]), np.array([]), []
    
    valid_patterns = patterns[patterns[outcome_col].isin(['WIN', 'LOSS'])].copy()
    valid_patterns = valid_patterns.sort_values('trigger_time')
    
    logger.info(f"Processing {len(valid_patterns)} valid patterns using {outcome_col}...")
    
    # Ensure index is UTC
    if df_1m.index.tz is None:
        df_1m.index = df_1m.index.tz_localize('UTC')
    else:
        df_1m.index = df_1m.index.tz_convert('UTC')
        
    for idx, pattern in valid_patterns.iterrows():
        trigger_time = pattern['trigger_time']
        
        # Ensure trigger_time is UTC
        if pd.Timestamp(trigger_time).tz is None:
            trigger_time = pd.Timestamp(trigger_time).tz_localize('UTC')
        else:
            trigger_time = pd.Timestamp(trigger_time).tz_convert('UTC')
        
        # Get window before trigger
        end_time = trigger_time
        start_time = end_time - pd.Timedelta(minutes=window_size)
        
        try:
            window = df_1m.loc[start_time:end_time]
        except KeyError:
            continue
            
        window = window[window.index < end_time]
        
        if len(window) < window_size:
            continue
        
        # Z-Score Normalization per window (per success_study.md)
        feats = window[['open', 'high', 'low', 'close']].values
        mean = np.mean(feats)
        std = np.std(feats)
        if std == 0:
            std = 1.0  # Prevent div/0
        
        feats_norm = (feats - mean) / std
        
        # Take last window_size bars
        if len(feats_norm) > window_size:
            feats_norm = feats_norm[-window_size:]
        elif len(feats_norm) < window_size:
            continue
        
        # Invert for SHORT direction to unify dataset (per success_study.md)
        pattern_direction = pattern.get('direction', 'SHORT')
        if pattern_direction == "SHORT":
            feats_norm = -feats_norm
        
        # Label using OCO-specific outcome
        label = 1 if pattern[outcome_col] == 'WIN' else 0
        
        X.append(feats_norm)
        y.append(label)
        timestamps.append(trigger_time)
    
    X = np.array(X)
    y = np.array(y)
    
    logger.info(f"Prepared {len(X)} samples. Win rate: {np.mean(y):.2f}")
    
    return X, y, timestamps


def get_model(architecture: str, seq_len: int, input_dim: int = 4):
    """Get model instance by architecture name."""
    
    if architecture == "CNN_Classic":
        return CNN_Classic(input_dim=input_dim, seq_len=seq_len)
    elif architecture == "CNN_Wide":
        return CNN_Wide(input_dim=input_dim, seq_len=seq_len)
    elif architecture == "LSTM_Seq":
        return LSTM_Seq(input_dim=input_dim)
    elif architecture == "Feature_MLP":
        # MLP expects flattened features
        return Feature_MLP(input_dim=seq_len * input_dim)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
) -> dict:
    """
    Train model and return metrics.
    """
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_acc = 0
    history = {"train_loss": [], "train_acc": [], "val_acc": []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                predicted = (outputs > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = val_correct / val_total if val_total > 0 else 0
        
        history["train_loss"].append(train_loss / len(train_loader))
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        logger.info(f"Epoch {epoch+1}/{epochs} - "
                    f"Loss: {train_loss/len(train_loader):.4f}, "
                    f"Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}")
    
    return {
        "best_val_acc": best_val_acc,
        "final_train_acc": train_acc,
        "final_val_acc": val_acc,
        "history": history,
    }


def main():
    args = parse_args()
    
    # Calculate window size from candle composition
    window_size = args.candles_1m  # Primary window (we'll handle multi-TF later)
    
    logger.info(f"Training {args.architecture} with {window_size} bar window")
    
    # Load pattern data
    pattern_path = Path(args.input_data)
    if not pattern_path.is_absolute():
        pattern_path = PROCESSED_DIR / args.input_data
    
    if not pattern_path.exists():
        logger.error(f"Pattern data not found: {pattern_path}")
        return {"error": "No data"}
    
    patterns = pd.read_parquet(pattern_path)
    logger.info(f"Loaded {len(patterns)} patterns")
    
    # Load 1m data
    data_path = PROCESSED_DIR / "continuous_1m.parquet"
    if not data_path.exists():
        data_path = ONE_MIN_PARQUET_DIR / "mes_1min_all.parquet"
    
    df_1m = pd.read_parquet(data_path)
    if 'time' in df_1m.columns:
        df_1m['time'] = pd.to_datetime(df_1m['time'], utc=True)
        df_1m = df_1m.set_index('time')
    df_1m = df_1m.sort_index()
    
    # Prepare data
    X, y, timestamps = prepare_data(patterns, df_1m, window_size, args.oco_config)
    
    if len(X) < 50:
        logger.error(f"Not enough samples: {len(X)}")
        return {"error": "Insufficient data"}
    
    # Split chronologically
    n = len(X)
    train_end = int(n * args.train_ratio)
    val_end = int(n * (args.train_ratio + args.val_ratio))
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    logger.info(f"Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Create dataloaders
    train_ds = TradeDataset(X_train, y_train)
    val_ds = TradeDataset(X_val, y_val)
    test_ds = TradeDataset(X_test, y_test)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)
    
    # Get model
    model = get_model(args.architecture, window_size)
    logger.info(f"Model: {model.__class__.__name__}")
    
    # Train
    train_result = train_model(model, train_loader, val_loader, args.epochs, args.lr)
    
    # Test evaluation
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    test_acc = test_correct / test_total if test_total > 0 else 0
    
    # Summary
    summary = {
        "architecture": args.architecture,
        "window_size": window_size,
        "candle_composition": f"{args.candles_1m}x1m+{args.candles_3m}x3m+"
                              f"{args.candles_5m}x5m+{args.candles_15m}x15m",
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "best_val_acc": train_result["best_val_acc"],
        "test_acc": test_acc,
        "train_win_rate": float(np.mean(y_train)),
        "test_win_rate": float(np.mean(y_test)),
    }
    
    logger.info("=" * 50)
    logger.info(f"Training Complete!")
    logger.info(f"Best Val Acc: {train_result['best_val_acc']:.3f}")
    logger.info(f"Test Acc: {test_acc:.3f}")
    logger.info("=" * 50)
    
    # Save model
    if not args.dry_run:
        suffix = args.output_suffix or datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = MODELS_DIR / f"sweep_{args.architecture}_{suffix}.pth"
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
        summary["model_path"] = str(model_path)
        
        # Save summary
        summary_path = MODELS_DIR / f"sweep_{args.architecture}_{suffix}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()

```

### scripts/train_from_shards.py

```python
"""
Train From Shards
Train a FusionModel on an existing sharded dataset.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader, random_split

from src.datasets.reader import DecisionDataset
from src.models.fusion import FusionModel
from src.models.train import train_model, TrainConfig, ImbalanceStrategy
from src.config import MODELS_DIR

def main():
    parser = argparse.ArgumentParser(description="Train Model from Shards")
    parser.add_argument("--data", type=str, required=True, help="Path to shard directory (e.g. shards/swing_breakout_v1)")
    parser.add_argument("--out", type=str, default="swing_breakout_model.pth", help="Output model filename")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    
    args = parser.parse_args()
    
    shard_dir = Path(args.data)
    model_out = MODELS_DIR / args.out
    
    print("=" * 60)
    print(f"Training on: {shard_dir}")
    print(f"Output to:   {model_out}")
    print("=" * 60)
    
    # 1. Load Data
    print("\n[1] Loading dataset...")
    full_dataset = DecisionDataset(shard_dir)
    print(f"Total records: {len(full_dataset)}")
    
    if len(full_dataset) < 10:
        print("Error: Not enough data to train.")
        sys.exit(1)
    
    # 2. Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch)
    
    print(f"Train size: {len(train_ds)}")
    print(f"Val size:   {len(val_ds)}")
    
    # 3. Configure
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=0.001,
        save_path=model_out,
        imbalance_strategy=ImbalanceStrategy.WEIGHTED_LOSS
    )
    
    # 4. Create Model
    # Note: We need to know the context dim. Default is 20 in schema code.
    # Ideally we read this from schema/manifest.
    model = FusionModel(
        context_dim=20, 
        num_classes=2,
        dropout=0.3
    )
    
    # 5. Train
    print("\n[2] Training...")
    result = train_model(model, train_loader, val_loader, config)
    
    print("\n[3] Results")
    print(f"Best Val Loss: {result.best_val_loss:.4f} (Epoch {result.best_epoch})")
    print(f"Model saved to: {model_out}")

if __name__ == "__main__":
    main()

```

### scripts/train_fusion_mtf.py

```python
#!/usr/bin/env python3
"""
Multi-Timeframe Fusion Model

Theory: Only trade if higher timeframe (1H) agrees with entry direction.
Input: 30 bars of 1m data + 5 bars of 1H data
Filter: If 1H is bullish (close > open over window), only take LONGS.

Usage:
    python scripts/train_fusion_mtf.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
from torch.utils.data import Dataset, DataLoader, random_split
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple

from src.storage import ExperimentDB


# =============================================================================
# Multi-Timeframe Fusion Model
# =============================================================================

class MTFFusionModel(nn.Module):
    """
    Fusion model that processes 1m and 1H data separately, then combines.
    
    Architecture:
    - 1m Branch: CNN for short-term patterns (30 bars × 5 features)
    - 1H Branch: MLP for trend context (5 bars × 5 features)
    - Fusion: Concatenate + FC layers
    """
    
    def __init__(
        self,
        bars_1m: int = 30,
        bars_1h: int = 5,
        num_features: int = 5,  # OHLCV
        num_classes: int = 2,   # LONG or SHORT
    ):
        super().__init__()
        
        self.bars_1m = bars_1m
        self.bars_1h = bars_1h
        
        # 1-Minute Branch (CNN)
        self.cnn_1m = nn.Sequential(
            nn.Conv1d(num_features, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
        )
        cnn_out_size = 64 * 4  # 256
        
        # 1-Hour Branch (MLP)
        self.mlp_1h = nn.Sequential(
            nn.Flatten(),
            nn.Linear(bars_1h * num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        mlp_out_size = 16
        
        # Fusion Head
        self.fusion = nn.Sequential(
            nn.Linear(cnn_out_size + mlp_out_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )
    
    def forward(self, x_1m, x_1h):
        """
        Args:
            x_1m: (batch, bars_1m, features) - 1-minute data
            x_1h: (batch, bars_1h, features) - 1-hour data
        """
        # CNN expects (batch, channels, seq_len)
        x_1m = x_1m.permute(0, 2, 1)
        
        # Process each branch
        feat_1m = self.cnn_1m(x_1m)      # (batch, 256)
        feat_1h = self.mlp_1h(x_1h)       # (batch, 16)
        
        # Fuse
        fused = torch.cat([feat_1m, feat_1h], dim=1)
        
        return self.fusion(fused)
    
    def get_1h_trend(self, x_1h):
        """Determine if 1H trend is bullish (returns True/False per sample)."""
        # Simple: compare first close to last close
        # x_1h: (batch, bars, features) where features = [O, H, L, C, V]
        first_close = x_1h[:, 0, 3]  # First bar close
        last_close = x_1h[:, -1, 3]  # Last bar close
        return last_close > first_close  # Bullish if rising


# =============================================================================
# Dataset with MTF data
# =============================================================================

class MTFDataset(Dataset):
    """Generate multi-timeframe samples from market data."""
    
    def __init__(
        self,
        df_1m: pd.DataFrame,
        df_1h: pd.DataFrame,
        bars_1m: int = 30,
        bars_1h: int = 5,
    ):
        self.samples_1m = []
        self.samples_1h = []
        self.labels = []
        self.bars_1m = bars_1m
        self.bars_1h = bars_1h
        
        # Align 1m and 1h data
        df_1m = df_1m.copy()
        df_1h = df_1h.copy()
        
        df_1m['hour'] = df_1m['time'].dt.floor('h')
        
        # For each valid point, create a sample
        for i in range(bars_1m + 60, len(df_1m) - 10):  # Leave room for outcome
            current_time = df_1m.iloc[i]['time']
            current_hour = current_time.floor('h')
            
            # Get 1m window
            window_1m = df_1m.iloc[i-bars_1m:i][['open', 'high', 'low', 'close', 'volume']].values
            
            # Get 1h window (last 5 hours before current)
            hourly_mask = df_1h['time'] < current_hour
            hourly_data = df_1h[hourly_mask].tail(bars_1h)
            
            if len(hourly_data) < bars_1h:
                continue
            
            window_1h = hourly_data[['open', 'high', 'low', 'close', 'volume']].values
            
            # Normalize each window separately (Z-score)
            window_1m = self._normalize(window_1m)
            window_1h = self._normalize(window_1h)
            
            # Simple label: next 10 bars go up = LONG(0), down = SHORT(1)
            future_close = df_1m.iloc[i + 10]['close']
            current_close = df_1m.iloc[i]['close']
            label = 0 if future_close > current_close else 1
            
            self.samples_1m.append(window_1m)
            self.samples_1h.append(window_1h)
            self.labels.append(label)
        
        print(f"MTF Dataset: {len(self.samples_1m)} samples")
        print(f"  LONG (up): {sum(1 for l in self.labels if l == 0)}")
        print(f"  SHORT (down): {sum(1 for l in self.labels if l == 1)}")
    
    def _normalize(self, data):
        """Z-score normalize."""
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0) + 1e-8
        return (data - mean) / std
    
    def __len__(self):
        return len(self.samples_1m)
    
    def __getitem__(self, idx):
        x_1m = torch.tensor(self.samples_1m[idx], dtype=torch.float32)
        x_1h = torch.tensor(self.samples_1h[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x_1m, x_1h, y


# =============================================================================
# Training with Trend Filter
# =============================================================================

def train_fusion_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 30,
    lr: float = 0.001,
    device: str = 'cuda',
    use_trend_filter: bool = True,
) -> Tuple[dict, float, Dict]:
    """
    Train fusion model with optional 1H trend filter.
    
    If use_trend_filter=True:
    - Only count LONG predictions as correct if 1H is bullish
    - Only count SHORT predictions as correct if 1H is bearish
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    best_state = None
    stats = {'filtered_trades': 0, 'total_trades': 0}
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for x_1m, x_1h, y in train_loader:
            x_1m, x_1h, y = x_1m.to(device), x_1h.to(device), y.to(device)
            
            optimizer.zero_grad()
            out = model(x_1m, x_1h)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validate with trend filter
        model.eval()
        correct_unfiltered = 0
        correct_filtered = 0
        total = 0
        filtered_trades = 0
        
        with torch.no_grad():
            for x_1m, x_1h, y in val_loader:
                x_1m, x_1h, y = x_1m.to(device), x_1h.to(device), y.to(device)
                
                out = model(x_1m, x_1h)
                _, pred = out.max(1)
                
                # Get 1H trend
                trend_bullish = model.get_1h_trend(x_1h)
                
                for i in range(len(pred)):
                    total += 1
                    
                    # Unfiltered accuracy
                    if pred[i] == y[i]:
                        correct_unfiltered += 1
                    
                    # Filtered: only count if direction matches trend
                    if use_trend_filter:
                        is_long = pred[i] == 0
                        should_take = (is_long and trend_bullish[i]) or (not is_long and not trend_bullish[i])
                        
                        if should_take:
                            filtered_trades += 1
                            if pred[i] == y[i]:
                                correct_filtered += 1
                    else:
                        filtered_trades += 1
                        if pred[i] == y[i]:
                            correct_filtered += 1
        
        acc_unfiltered = correct_unfiltered / total if total > 0 else 0
        acc_filtered = correct_filtered / filtered_trades if filtered_trades > 0 else 0
        
        if acc_filtered > best_acc:
            best_acc = acc_filtered
            best_state = model.state_dict().copy()
            stats = {
                'filtered_trades': filtered_trades,
                'total_trades': total,
                'unfiltered_acc': acc_unfiltered,
                'filtered_acc': acc_filtered,
            }
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f} "
                  f"- Val Acc: {acc_unfiltered:.1%} (unfiltered) / {acc_filtered:.1%} (filtered)")
    
    return best_state, best_acc, stats


# =============================================================================
# Main
# =============================================================================

def run_fusion_comparison(days: int = 7) -> Dict[str, Any]:
    """
    Train fusion model and compare filtered vs unfiltered expectancy.
    """
    print("=" * 60)
    print("MULTI-TIMEFRAME FUSION MODEL")
    print("=" * 60)
    print("Input: 30 bars (1m) + 5 bars (1H)")
    print("Filter: Only trade if 1H trend agrees with direction")
    print("=" * 60)
    
    # Load data
    actual_days = min(days, 7)
    end = datetime.now()
    start = end - timedelta(days=actual_days)
    
    print(f"\n[1] Loading {actual_days} days of ES data...")
    ticker = yf.Ticker("ES=F")
    
    df_1m = ticker.history(start=start, end=end, interval="1m")
    df_1h = ticker.history(start=start - timedelta(days=30), end=end, interval="1h")
    
    if df_1m is None or len(df_1m) == 0:
        print("ERROR: No data!")
        return {}
    
    # Standardize
    for df in [df_1m, df_1h]:
        df.columns = [c.lower() for c in df.columns]
    
    df_1m = df_1m.reset_index()
    df_1h = df_1h.reset_index()
    df_1m['time'] = pd.to_datetime(df_1m['Datetime'] if 'Datetime' in df_1m.columns else df_1m['datetime'])
    df_1h['time'] = pd.to_datetime(df_1h['Datetime'] if 'Datetime' in df_1h.columns else df_1h['datetime'])
    
    print(f"    1m: {len(df_1m)} bars")
    print(f"    1h: {len(df_1h)} bars")
    
    # Create dataset
    print(f"\n[2] Creating MTF dataset...")
    dataset = MTFDataset(df_1m, df_1h, bars_1m=30, bars_1h=5)
    
    if len(dataset) < 50:
        print("ERROR: Not enough samples")
        return {}
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    
    # Train model
    print(f"\n[3] Training Fusion Model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"    Device: {device}")
    
    model = MTFFusionModel(bars_1m=30, bars_1h=5, num_classes=2)
    
    best_state, best_acc, stats = train_fusion_model(
        model, train_loader, val_loader,
        epochs=30, lr=0.001, device=device,
        use_trend_filter=True
    )
    
    # Save
    model_path = Path("models/mtf_fusion.pth")
    model_path.parent.mkdir(exist_ok=True)
    torch.save(best_state, model_path)
    
    # Results
    print("\n" + "=" * 60)
    print("RESULTS: TREND FILTER IMPACT")
    print("=" * 60)
    print(f"  Unfiltered Accuracy: {stats['unfiltered_acc']:.1%}")
    print(f"  Filtered Accuracy:   {stats['filtered_acc']:.1%}")
    print(f"  Trades Taken:        {stats['filtered_trades']}/{stats['total_trades']} "
          f"({100*stats['filtered_trades']/stats['total_trades']:.0f}%)")
    
    improvement = (stats['filtered_acc'] - stats['unfiltered_acc']) * 100
    if improvement > 0:
        print(f"\n  ✓ Filter IMPROVED expectancy by {improvement:.1f} percentage points!")
    else:
        print(f"\n  ✗ Filter did not improve expectancy ({improvement:.1f}pp)")
    
    # Store
    db = ExperimentDB()
    run_id = f"mtf_fusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    db.store_run(
        run_id=run_id,
        strategy="mtf_fusion",
        config={
            'bars_1m': 30,
            'bars_1h': 5,
            'use_trend_filter': True,
        },
        metrics={
            'total_trades': stats['total_trades'],
            'filtered_trades': stats['filtered_trades'],
            'wins': int(stats['filtered_trades'] * stats['filtered_acc']),
            'losses': int(stats['filtered_trades'] * (1 - stats['filtered_acc'])),
            'win_rate': stats['filtered_acc'],
            'unfiltered_win_rate': stats['unfiltered_acc'],
            'total_pnl': 0,
        },
        model_path=str(model_path)
    )
    print(f"\n[+] Saved to ExperimentDB: {run_id}")
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MTF Fusion Model")
    parser.add_argument("--days", type=int, default=7, help="Days to train on")
    
    args = parser.parse_args()
    
    results = run_fusion_comparison(args.days)

```

### scripts/train_ifvg_4class.py

```python
#!/usr/bin/env python3
"""
Train 4-Class IFVG CNN

Train a CNN with 4 classes to predict both direction AND outcome:
- Class 0: LONG_WIN
- Class 1: LONG_LOSS  
- Class 2: SHORT_WIN
- Class 3: SHORT_LOSS

This allows the model to:
1. Skip low-quality setups (neither direction looks like a winner)
2. Pick the direction that has higher win probability
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from typing import List, Dict

from src.config import MODELS_DIR

# ============================================================================
# CONFIGURATION
# ============================================================================

RECORDS_FILE = Path("results/ifvg_debug/records.jsonl")
MODEL_OUT = MODELS_DIR / "ifvg_4class_cnn.pth"
LOOKBACK_BARS = 30
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 0.001


# ============================================================================
# MODEL
# ============================================================================

class IFVG4ClassCNN(nn.Module):
    """
    CNN for 4-class IFVG pattern detection.
    
    Input: (batch, 5, 30) - OHLCV channels
    Output: (batch, 4) - [P(LONG_WIN), P(LONG_LOSS), P(SHORT_WIN), P(SHORT_LOSS)]
    """
    
    def __init__(self, input_channels: int = 5, seq_length: int = 30, num_classes: int = 4):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)


# ============================================================================
# DATASET
# ============================================================================

def get_label(direction: str, outcome: str) -> int:
    """Map direction + outcome to 4-class label."""
    if direction == "LONG" and outcome == "WIN":
        return 0  # LONG_WIN
    elif direction == "LONG" and outcome == "LOSS":
        return 1  # LONG_LOSS
    elif direction == "SHORT" and outcome == "WIN":
        return 2  # SHORT_WIN
    elif direction == "SHORT" and outcome == "LOSS":
        return 3  # SHORT_LOSS
    else:
        return -1  # Skip (timeout, not filled)


class IFVG4ClassDataset(Dataset):
    """Dataset for 4-class IFVG training."""
    
    def __init__(self, records: List[Dict], lookback: int = 30):
        self.samples = []
        self.labels = []
        
        for record in records:
            oco = record.get('oco', {})
            oco_results = record.get('oco_results', {})
            
            direction = oco.get('direction', '')
            outcome = oco_results.get('outcome', '')
            
            # Get label
            label = get_label(direction, outcome)
            if label == -1:
                continue  # Skip timeouts/not filled
            
            # Get price window
            window_data = record.get('window', {})
            raw_ohlcv = window_data.get('raw_ohlcv_1m', [])
            
            if len(raw_ohlcv) < lookback:
                continue
            
            # Take first lookback bars (history before entry)
            pre_trade = raw_ohlcv[:lookback]
            
            # Convert to numpy array: (5, lookback)
            ohlcv = np.array([
                [b['open'] for b in pre_trade],
                [b['high'] for b in pre_trade],
                [b['low'] for b in pre_trade],
                [b['close'] for b in pre_trade],
                [b.get('volume', 0) for b in pre_trade]
            ], dtype=np.float32)
            
            # Normalize prices by first close (percent change)
            first_close = ohlcv[3, 0]
            if first_close > 0:
                ohlcv[0:4] = (ohlcv[0:4] - first_close) / first_close * 100
            
            # Normalize volume
            max_vol = ohlcv[4].max()
            if max_vol > 0:
                ohlcv[4] = ohlcv[4] / max_vol
            
            self.samples.append(ohlcv)
            self.labels.append(label)
        
        # Print class distribution
        from collections import Counter
        counts = Counter(self.labels)
        print(f"Dataset: {len(self.samples)} samples")
        print(f"  LONG_WIN (0): {counts[0]}")
        print(f"  LONG_LOSS (1): {counts[1]}")
        print(f"  SHORT_WIN (2): {counts[2]}")
        print(f"  SHORT_LOSS (3): {counts[3]}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, train_loader, val_loader, epochs, lr, device):
    model = model.to(device)
    
    # Use weighted loss for class imbalance
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_acc = 0
    best_state = None
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = logits.argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        
        val_acc = correct / max(1, total)
        scheduler.step(1 - val_acc)
        
        if epoch % 5 == 0 or val_acc > best_val_acc:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss/len(train_loader):.4f} - Val Acc: {val_acc:.2%}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    return best_state, best_val_acc


def main():
    print("=" * 60)
    print("Train 4-Class IFVG CNN")
    print("=" * 60)
    
    # Load records
    print("\n[1] Loading records...")
    if not RECORDS_FILE.exists():
        print(f"Error: {RECORDS_FILE} not found")
        sys.exit(1)
    
    records = []
    with open(RECORDS_FILE) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} records")
    
    # Create dataset
    print("\n[2] Creating dataset...")
    dataset = IFVG4ClassDataset(records, lookback=LOOKBACK_BARS)
    
    if len(dataset) < 20:
        print("Error: Not enough samples")
        sys.exit(1)
    
    # Split 80/20
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Train
    print("\n[3] Training...")
    model = IFVG4ClassCNN()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    best_state, best_acc = train_model(model, train_loader, val_loader, EPOCHS, LEARNING_RATE, device)
    
    # Save
    print("\n[4] Saving model...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, MODEL_OUT)
    print(f"Saved to: {MODEL_OUT}")
    print(f"Best validation accuracy: {best_acc:.2%}")
    
    # Save model info
    info = {
        "architecture": "IFVG4ClassCNN",
        "input_shape": [5, LOOKBACK_BARS],
        "num_classes": 4,
        "classes": ["LONG_WIN", "LONG_LOSS", "SHORT_WIN", "SHORT_LOSS"],
        "best_val_accuracy": best_acc,
        "training_samples": len(train_ds),
        "epochs": EPOCHS
    }
    with open(MODEL_OUT.with_suffix('.json'), 'w') as f:
        json.dump(info, f, indent=2)


if __name__ == "__main__":
    main()

```

### scripts/train_ifvg_cnn.py

```python
#!/usr/bin/env python3
"""
Train IFVG CNN

Train a CNN to recognize pre-trade patterns from successful IFVG trades.
Uses 30 1m candles before each trade, labeled by direction (LONG/SHORT).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from typing import List, Dict, Any

from src.data.loader import load_continuous_contract
from src.config import MODELS_DIR, NY_TZ

# ============================================================================
# CONFIGURATION
# ============================================================================

RECORDS_FILE = Path("results/ict_ifvg/records.jsonl")
MODEL_OUT = MODELS_DIR / "ifvg_cnn.pth"
LOOKBACK_BARS = 30  # 30 1m candles before entry
EPOCHS = 30
BATCH_SIZE = 8
LEARNING_RATE = 0.001


# ============================================================================
# MODEL
# ============================================================================

class IFVGPatternCNN(nn.Module):
    """
    Simple CNN for IFVG pattern detection.
    
    Input: (batch, 5, 30) - 5 channels (OHLCV), 30 time steps
    Output: (batch, 2) - probability of LONG vs SHORT
    """
    
    def __init__(self, input_channels: int = 5, seq_length: int = 30, num_classes: int = 2):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 30 -> 15
            
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 15 -> 7
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # 7 -> 1
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, length)
        Returns:
            Logits (batch, num_classes)
        """
        x = self.features(x)
        return self.classifier(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability of each class."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)


# ============================================================================
# DATASET
# ============================================================================

class IFVGTradeDataset(Dataset):
    """Dataset of pre-trade price patterns from IFVG records."""
    
    def __init__(self, records: List[Dict], df_1m, lookback: int = 30):
        self.samples = []
        self.labels = []
        
        for record in records:
            # Only use filled winning trades for training
            oco_results = record.get('oco_results', {})
            if not oco_results.get('filled', False):
                continue
            if oco_results.get('outcome') not in ('WIN', 'LOSS'):
                continue
            
            # Get direction label: LONG=1, SHORT=0
            oco = record.get('oco', {})
            direction = oco.get('direction', 'LONG')
            label = 1 if direction == 'LONG' else 0
            
            # Extract price window from raw_ohlcv_1m
            window_data = record.get('window', {})
            raw_ohlcv = window_data.get('raw_ohlcv_1m', [])
            
            if len(raw_ohlcv) < lookback:
                continue
            
            # Take last `lookback` bars before trade (first bars in window are history)
            # The window should have history before entry
            pre_trade = raw_ohlcv[:lookback]
            
            # Convert to numpy array: (lookback, 5) -> (5, lookback)
            ohlcv = np.array([
                [b['open'] for b in pre_trade],
                [b['high'] for b in pre_trade],
                [b['low'] for b in pre_trade],
                [b['close'] for b in pre_trade],
                [b.get('volume', 0) for b in pre_trade]
            ], dtype=np.float32)
            
            # Normalize price columns (0-3) by first close
            first_close = ohlcv[3, 0]
            if first_close > 0:
                ohlcv[0:4] = (ohlcv[0:4] - first_close) / first_close * 100  # Percent change
            
            # Normalize volume by max
            max_vol = ohlcv[4].max()
            if max_vol > 0:
                ohlcv[4] = ohlcv[4] / max_vol
            
            self.samples.append(ohlcv)
            self.labels.append(label)
        
        print(f"Created dataset with {len(self.samples)} samples")
        print(f"  LONG: {sum(self.labels)}, SHORT: {len(self.labels) - sum(self.labels)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, train_loader, val_loader, epochs, lr, device):
    """Train the model and return best weights."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_val_acc = 0
    best_state = None
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = logits.argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        
        val_acc = correct / max(1, total)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss/len(train_loader):.4f} - Val Acc: {val_acc:.2%}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
    
    return best_state, best_val_acc


def main():
    print("=" * 60)
    print("Train IFVG Pattern CNN")
    print("=" * 60)
    
    # Load records
    print("\n[1] Loading IFVG trade records...")
    if not RECORDS_FILE.exists():
        print(f"Error: {RECORDS_FILE} not found. Run run_ict_ifvg.py first.")
        sys.exit(1)
    
    records = []
    with open(RECORDS_FILE) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} records")
    
    # Load full 1m data (for potential augmentation)
    print("\n[2] Loading market data...")
    df_1m = load_continuous_contract()
    print(f"Loaded {len(df_1m)} 1m bars")
    
    # Create dataset
    print("\n[3] Creating dataset...")
    dataset = IFVGTradeDataset(records, df_1m, lookback=LOOKBACK_BARS)
    
    if len(dataset) < 4:
        print("Error: Not enough samples to train. Need more trades.")
        sys.exit(1)
    
    # Split
    train_size = max(1, int(0.7 * len(dataset)))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Create model
    print("\n[4] Training...")
    model = IFVGPatternCNN(input_channels=5, seq_length=LOOKBACK_BARS, num_classes=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    best_state, best_acc = train_model(model, train_loader, val_loader, EPOCHS, LEARNING_RATE, device)
    
    # Save
    print("\n[5] Saving model...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, MODEL_OUT)
    print(f"Saved to: {MODEL_OUT}")
    print(f"Best validation accuracy: {best_acc:.2%}")


if __name__ == "__main__":
    main()

```

### scripts/train_lstm_compare.py

```python
#!/usr/bin/env python3
"""
LSTM vs CNN Comparison

Engineer theory: CNN focuses on shapes, LSTM captures price flow/sequence.
Test: Train an LSTM on 60-bar close price sequences, compare to baseline CNN.

Usage:
    python scripts/train_lstm_compare.py --input results/ict_ifvg/records.jsonl
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from datetime import datetime
from typing import Dict, Any, List, Tuple

from src.storage import ExperimentDB
from src.features.engine import normalize_ohlcv_window, FeatureConfig


# =============================================================================
# LSTM Model Architecture
# =============================================================================

class PriceLSTM(nn.Module):
    """
    LSTM for price sequence classification.
    

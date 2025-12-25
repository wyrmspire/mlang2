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

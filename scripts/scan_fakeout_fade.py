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

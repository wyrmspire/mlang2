"""
EMA Rejection Strategy Scanner - Using Generic RejectionTrigger

Strategy: Rejection at the 200 EMA when trending (20 EMA angled).

Usage:
    python scripts/scan_ema_rejection.py --feature ema_200 --weeks 4 --start-date 2025-08-18
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.strategy.scan import run_strategy_scan, RTHFilter, MinATRFilter
from src.policy.triggers.parametric import RejectionTrigger, ComparisonTrigger
from src.policy.brackets import FixedBracket
from src.features.indicators import calculate_atr
from src.config import NY_TZ
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Generic EMA Rejection Strategy Scanner")
    parser.add_argument("--feature", type=str, default="ema_200", 
                        help="Feature to reject from (ema_200, ema_50, pdh, pdl, vwap)")
    parser.add_argument("--direction", type=str, default="both",
                        choices=["both", "long_only", "short_only"],
                        help="Which rejections to take")
    parser.add_argument("--start-date", type=str, default="2025-08-18")
    parser.add_argument("--weeks", type=int, default=4)
    parser.add_argument("--stop-points", type=float, default=5.0)
    parser.add_argument("--tp-r", type=float, default=2.0)
    parser.add_argument("--timeframe", type=str, default="5m")
    parser.add_argument("--out", type=str, default=None)
    
    args = parser.parse_args()
    
    # Create trigger
    trigger = RejectionTrigger(
        feature=args.feature,
        direction=args.direction
    )
    
    bracket = FixedBracket(
        stop_points=args.stop_points,
        tp_points=args.stop_points * args.tp_r
    )
    
    # Precompute indicators for the data range
    from src.data.loader import load_continuous_contract
    from src.data.resample import resample_all_timeframes
    
    start = pd.Timestamp(args.start_date)
    end = start + pd.Timedelta(weeks=args.weeks, unit='W')
    
    # Need extra lookback for 200 EMA
    extended_start = start - pd.Timedelta(days=10)
    
    df_1m = load_continuous_contract()
    df_1m = df_1m[(df_1m['time'] >= str(extended_start)) & (df_1m['time'] < str(end))].reset_index(drop=True)
    
    htf_data = resample_all_timeframes(df_1m)
    df_5m = htf_data.get('5m')
    
    if df_5m is not None and len(df_5m) > 200:
        # Add EMAs dynamically based on feature
        df_5m['ema_20'] = df_5m['close'].ewm(span=20, adjust=False).mean()
        df_5m['ema_50'] = df_5m['close'].ewm(span=50, adjust=False).mean()
        df_5m['ema_200'] = df_5m['close'].ewm(span=200, adjust=False).mean()
        df_5m['atr'] = calculate_atr(df_5m, 14)
        
        # Add PDH/PDL
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
    
    def add_indicators(bar, features):
        """Add indicator values to features for the trigger."""
        idx = bar.name if hasattr(bar, 'name') else 0
        if idx >= len(df_5m):
            return {}
        
        row = df_5m.iloc[idx]
        return {
            'ema_20': row.get('ema_20', 0),
            'ema_50': row.get('ema_50', 0),
            'ema_200': row.get('ema_200', 0),
            'pdh': row.get('pdh', 0),
            'pdl': row.get('pdl', 0),
        }
    
    # Run the scan
    feature_clean = args.feature.replace('_', '')
    run_name = args.out or f"NEW_rejection_{feature_clean}_{args.start_date.replace('-', '')}"
    
    result = run_strategy_scan(
        trigger=trigger,
        bracket=bracket,
        start_date=args.start_date,
        weeks=args.weeks,
        filters=[RTHFilter(), MinATRFilter(threshold=2.0)],
        run_name=run_name,
        timeframe=args.timeframe,
        extra_context_fn=add_indicators
    )
    
    print(f"\nRun ID for UI: {result.run_name}")


if __name__ == "__main__":
    main()

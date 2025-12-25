"""
EMA 200 Rejection Strategy Scanner

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

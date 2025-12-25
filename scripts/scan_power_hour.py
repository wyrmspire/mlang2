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

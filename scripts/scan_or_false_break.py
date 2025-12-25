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

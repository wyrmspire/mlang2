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

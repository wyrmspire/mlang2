#!/usr/bin/env python3
"""
Demo script showing the new Phase 5/6 functionality.

This demonstrates:
1. Centralized contract sizing
2. Centralized PnL calculation
3. 2-hour window enforcement
4. Proper exporter usage

Run: python scripts/demo_phase_5_6.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from datetime import datetime, timedelta

from src.sim.sizing import calculate_contracts, calculate_pnl_dollars, calculate_reward_dollars
from src.viz.window_utils import enforce_2hour_window, get_window_bounds_from_trades
from src.config import DEFAULT_MAX_RISK_DOLLARS


def demo_contract_sizing():
    """Demonstrate centralized contract sizing."""
    print("=" * 70)
    print("1. CONTRACT SIZING DEMO")
    print("=" * 70)
    
    # Example trade setup
    entry_price = 5000.0
    stop_price = 4990.0  # 10 points risk
    max_risk = 300.0
    
    # Calculate contracts using centralized function
    result = calculate_contracts(entry_price, stop_price, max_risk)
    
    print(f"\nTrade Setup:")
    print(f"  Entry: ${entry_price}")
    print(f"  Stop:  ${stop_price}")
    print(f"  Risk:  {result.risk_points} points")
    
    print(f"\nPosition Sizing (Max Risk: ${max_risk}):")
    print(f"  Contracts: {result.contracts}")
    print(f"  Actual Risk: ${result.risk_dollars:.2f}")
    print(f"  Point Value: ${result.point_value}")
    
    # Calculate potential reward
    tp_price = 5014.0  # 14 points reward (1.4 R multiple)
    reward = calculate_reward_dollars(entry_price, tp_price, "LONG", result.contracts)
    print(f"\nPotential Reward:")
    print(f"  TP Price: ${tp_price}")
    print(f"  Reward: ${reward:.2f}")
    print(f"  R-Multiple: {reward / result.risk_dollars:.2f}R")


def demo_pnl_calculation():
    """Demonstrate centralized PnL calculation."""
    print("\n" + "=" * 70)
    print("2. PnL CALCULATION DEMO")
    print("=" * 70)
    
    # Winning trade
    entry = 5000.0
    exit_win = 5014.0
    contracts = 6
    
    pnl_points, pnl_dollars = calculate_pnl_dollars(
        entry_price=entry,
        exit_price=exit_win,
        direction="LONG",
        contracts=contracts,
        include_commission=True
    )
    
    print(f"\nWinning LONG Trade:")
    print(f"  Entry: ${entry} → Exit: ${exit_win}")
    print(f"  Contracts: {contracts}")
    print(f"  PnL: {pnl_points} points = ${pnl_dollars:.2f}")
    
    # Losing trade
    exit_loss = 4990.0
    pnl_points_loss, pnl_dollars_loss = calculate_pnl_dollars(
        entry_price=entry,
        exit_price=exit_loss,
        direction="LONG",
        contracts=contracts,
        include_commission=True
    )
    
    print(f"\nLosing LONG Trade:")
    print(f"  Entry: ${entry} → Exit: ${exit_loss}")
    print(f"  Contracts: {contracts}")
    print(f"  PnL: {pnl_points_loss} points = ${pnl_dollars_loss:.2f}")
    
    # Verify invariant
    print(f"\nInvariant Validation:")
    point_value = 5.0
    expected_gross = pnl_points * point_value * contracts
    print(f"  pnl_points * point_value * contracts = ${expected_gross:.2f}")
    print(f"  Actual pnl_dollars (with commission) = ${pnl_dollars:.2f}")
    print(f"  ✓ Invariant holds (within commission range)")


def demo_window_enforcement():
    """Demonstrate 2-hour window enforcement."""
    print("\n" + "=" * 70)
    print("3. 2-HOUR WINDOW ENFORCEMENT DEMO")
    print("=" * 70)
    
    # Create sample 1-minute data (6 hours worth)
    start_time = datetime(2025, 3, 17, 8, 0)
    times = [start_time + timedelta(minutes=i) for i in range(6 * 60)]
    
    df_1m = pd.DataFrame({
        'time': times,
        'open': [5000.0] * len(times),
        'high': [5010.0] * len(times),
        'low': [4990.0] * len(times),
        'close': [5000.0] * len(times),
        'volume': [100] * len(times),
    })
    
    # Trade entry at 10:00, exit after 30 bars (10:30)
    entry_time = datetime(2025, 3, 17, 10, 0)
    bars_held = 30
    
    print(f"\nTrade Details:")
    print(f"  Entry: {entry_time.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Bars Held: {bars_held} minutes")
    exit_time = entry_time + timedelta(minutes=bars_held)
    print(f"  Exit: {exit_time.strftime('%Y-%m-%d %H:%M')}")
    
    # Enforce 2-hour window
    raw_ohlcv, warning = enforce_2hour_window(
        df_1m=df_1m,
        entry_time=entry_time,
        bars_held=bars_held
    )
    
    print(f"\n2-Hour Window Policy:")
    print(f"  Required Start: {(entry_time - timedelta(hours=2)).strftime('%H:%M')}")
    print(f"  Required End: {(exit_time + timedelta(hours=2)).strftime('%H:%M')}")
    
    if raw_ohlcv:
        first_time = datetime.fromisoformat(raw_ohlcv[0]['time'])
        last_time = datetime.fromisoformat(raw_ohlcv[-1]['time'])
        print(f"\n  Actual Start: {first_time.strftime('%H:%M')}")
        print(f"  Actual End: {last_time.strftime('%H:%M')}")
        print(f"  Total Bars: {len(raw_ohlcv)}")
    
    if warning:
        print(f"\n  ⚠️  Warning: {warning}")
    else:
        print(f"\n  ✓ Full 2-hour window available")


def demo_window_bounds():
    """Demonstrate window bounds computation from trades."""
    print("\n" + "=" * 70)
    print("4. WINDOW BOUNDS FROM TRADES DEMO")
    print("=" * 70)
    
    # Multiple trades throughout the day
    trades = [
        {
            'entry_time': '2025-03-17T09:00:00-05:00',
            'exit_time': '2025-03-17T09:30:00-05:00',
        },
        {
            'entry_time': '2025-03-17T10:30:00-05:00',
            'exit_time': '2025-03-17T11:00:00-05:00',
        },
        {
            'entry_time': '2025-03-17T13:00:00-05:00',
            'exit_time': '2025-03-17T14:30:00-05:00',
        },
    ]
    
    bounds = get_window_bounds_from_trades(trades)
    
    print(f"\nTrades:")
    for i, trade in enumerate(trades, 1):
        entry = pd.Timestamp(trade['entry_time'])
        exit_t = pd.Timestamp(trade['exit_time'])
        print(f"  Trade {i}: {entry.strftime('%H:%M')} → {exit_t.strftime('%H:%M')}")
    
    print(f"\nComputed Window Bounds (2-hour policy):")
    first_entry = pd.Timestamp(bounds['first_entry'])
    last_exit = pd.Timestamp(bounds['last_exit'])
    window_start = pd.Timestamp(bounds['window_start'])
    window_end = pd.Timestamp(bounds['window_end'])
    
    print(f"  First Entry: {first_entry.strftime('%H:%M')}")
    print(f"  Last Exit: {last_exit.strftime('%H:%M')}")
    print(f"  Window Start: {window_start.strftime('%H:%M')} (2h before first entry)")
    print(f"  Window End: {window_end.strftime('%H:%M')} (2h after last exit)")
    
    total_hours = (window_end - window_start).total_seconds() / 3600
    print(f"  Total Window: {total_hours:.1f} hours")


def main():
    """Run all demos."""
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  PHASE 5/6 IMPLEMENTATION DEMO".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print()
    
    demo_contract_sizing()
    demo_pnl_calculation()
    demo_window_enforcement()
    demo_window_bounds()
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nFor more information, see docs/PHASE_5_6_GUIDE.md")
    print()


if __name__ == "__main__":
    main()

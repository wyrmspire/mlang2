"""
Window utilities for enforcing 2-hour policy.

According to ARCHITECTURE_AGREEMENT.md Section 3:
- Exporter MUST guarantee 2 hours of context before first trade entry
- Exporter MUST guarantee 2 hours after last exit
- This is enforced at exporter level, not UI hack
"""

import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import timedelta


def enforce_2hour_window(
    df_1m: pd.DataFrame,
    entry_time: pd.Timestamp,
    exit_time: Optional[pd.Timestamp] = None,
    bars_held: int = 0
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Enforce 2-hour window policy for OHLCV data.
    
    Args:
        df_1m: DataFrame with 1-minute OHLCV data
        entry_time: Trade entry time
        exit_time: Trade exit time (if known)
        bars_held: Number of bars held (used if exit_time not provided)
    
    Returns:
        Tuple of (raw_ohlcv_list, warning_message)
        - raw_ohlcv_list: List of dicts with time, open, high, low, close, volume
        - warning_message: String if data is missing, None otherwise
    
    Policy:
        window_start = entry_time - 2 hours
        window_end = exit_time + 2 hours (or entry_time + bars_held + 2 hours)
    """
    # Compute window bounds
    window_start = entry_time - timedelta(hours=2)
    
    if exit_time is not None:
        window_end = exit_time + timedelta(hours=2)
    else:
        # Estimate exit time from bars_held (assume 1-minute bars)
        exit_time_est = entry_time + timedelta(minutes=bars_held)
        window_end = exit_time_est + timedelta(hours=2)
    
    # Filter data
    if 'time' not in df_1m.columns:
        return [], "DataFrame missing 'time' column"
    
    mask = (df_1m['time'] >= window_start) & (df_1m['time'] <= window_end)
    window_df = df_1m.loc[mask]
    
    # Check if we have full coverage
    warning = None
    if window_df.empty:
        warning = f"No data available for window {window_start} to {window_end}"
    elif window_df['time'].min() > window_start:
        missing_start = (window_df['time'].min() - window_start).total_seconds() / 60
        warning = f"Missing {missing_start:.0f} minutes at start of 2h window"
    elif window_df['time'].max() < window_end:
        missing_end = (window_end - window_df['time'].max()).total_seconds() / 60
        warning = f"Missing {missing_end:.0f} minutes at end of 2h window"
    
    # Format as list of dicts
    raw_ohlcv = [
        {
            "time": row['time'].isoformat() if hasattr(row['time'], 'isoformat') else str(row['time']),
            "open": float(row['open']),
            "high": float(row['high']),
            "low": float(row['low']),
            "close": float(row['close']),
            "volume": int(row.get('volume', 0))
        }
        for _, row in window_df.iterrows()
    ]
    
    return raw_ohlcv, warning


def get_window_bounds_from_trades(
    trades: List[Dict[str, Any]]
) -> Optional[Dict[str, str]]:
    """
    Compute required window bounds from a list of trades.
    
    Args:
        trades: List of trade dicts with entry_time and exit_time
    
    Returns:
        Dict with window_start and window_end, or None if no trades
    """
    if not trades:
        return None
    
    # Find earliest entry and latest exit
    entry_times = []
    exit_times = []
    
    for trade in trades:
        if 'entry_time' in trade and trade['entry_time']:
            entry_times.append(pd.Timestamp(trade['entry_time']))
        if 'exit_time' in trade and trade['exit_time']:
            exit_times.append(pd.Timestamp(trade['exit_time']))
    
    if not entry_times or not exit_times:
        return None
    
    first_entry = min(entry_times)
    last_exit = max(exit_times)
    
    # Apply 2-hour policy
    window_start = first_entry - timedelta(hours=2)
    window_end = last_exit + timedelta(hours=2)
    
    return {
        'window_start': window_start.isoformat(),
        'window_end': window_end.isoformat(),
        'first_entry': first_entry.isoformat(),
        'last_exit': last_exit.isoformat(),
    }

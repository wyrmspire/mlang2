"""
Session Levels
Compute session-specific high/low levels (Asian, London).

These levels are used for ICT-style strategies that trade overnight level breaks.
"""

import pandas as pd
from dataclasses import dataclass
from typing import Optional
from datetime import time
from zoneinfo import ZoneInfo

from src.config import NY_TZ


@dataclass
class SessionLevels:
    """Bundle of session high/low levels."""
    # Asian session (NY evening prior day)
    asian_high: float = 0.0
    asian_low: float = 0.0
    asian_range: float = 0.0
    
    # London session
    london_high: float = 0.0
    london_low: float = 0.0
    london_range: float = 0.0
    
    # Combined overnight range
    overnight_high: float = 0.0
    overnight_low: float = 0.0


# Session time boundaries (NY timezone)
ASIAN_START = time(19, 0)   # 7:00 PM previous day
ASIAN_END = time(0, 0)      # Midnight

LONDON_START = time(2, 0)    # 2:00 AM
LONDON_END = time(8, 30)     # 8:30 AM (cutoff for level establishment)

TRADE_WINDOW_START = time(9, 30)   # NY Open
TRADE_WINDOW_END = time(11, 30)    # Mid-morning cutoff

# Cache for session levels (keyed by date)
_session_cache = {}


def clear_session_cache():
    """Clear the session level cache. Call at start of new backtests."""
    global _session_cache
    _session_cache = {}


def compute_session_levels(
    df_1m: pd.DataFrame,
    current_time: pd.Timestamp,
    tz: ZoneInfo = NY_TZ
) -> SessionLevels:
    """
    Compute Asian and London session high/low levels.
    
    Asian Session: 19:00 - 00:00 NY (previous evening)
    London Session: 02:00 - 08:30 NY (cutoff before NY open)
    
    Uses caching to avoid recomputing for same trading day.
    
    Args:
        df_1m: 1-minute OHLCV data with 'time' column
        current_time: Current timestamp
        tz: Timezone for session boundaries
        
    Returns:
        SessionLevels with Asian and London high/low values
    """
    global _session_cache
    
    levels = SessionLevels()
    
    if df_1m is None or df_1m.empty:
        return levels
    
    # Get current date for caching
    current_ny = current_time.astimezone(tz)
    current_date = current_ny.date()
    
    # Check cache first
    if current_date in _session_cache:
        return _session_cache[current_date]
    
    df = df_1m.copy()
    
    # Ensure time column is datetime
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        if df['time'].dt.tz is None:
            df['time'] = df['time'].dt.tz_localize('UTC')
        df['time_ny'] = df['time'].dt.tz_convert(tz)
    else:
        return levels
    
    current_ny = current_time.astimezone(tz)
    current_date = current_ny.date()
    
    # Asian session: 19:00 previous day to 00:00 current day
    # We need to look at yesterday evening
    from datetime import datetime, timedelta
    
    prev_day = current_date - timedelta(days=1)
    
    # Asian start: yesterday at 19:00
    asian_start = datetime.combine(prev_day, ASIAN_START)
    asian_start = tz.localize(asian_start) if hasattr(tz, 'localize') else asian_start.replace(tzinfo=tz)
    
    # Asian end: today at 00:00 (midnight)
    asian_end = datetime.combine(current_date, time(0, 0))
    asian_end = tz.localize(asian_end) if hasattr(tz, 'localize') else asian_end.replace(tzinfo=tz)
    
    # Filter Asian session bars
    asian_mask = (df['time_ny'] >= asian_start) & (df['time_ny'] < asian_end)
    asian_bars = df.loc[asian_mask]
    
    if not asian_bars.empty:
        levels.asian_high = asian_bars['high'].max()
        levels.asian_low = asian_bars['low'].min()
        levels.asian_range = levels.asian_high - levels.asian_low
    
    # London session: 02:00 to 08:30 today
    london_start = datetime.combine(current_date, LONDON_START)
    london_start = tz.localize(london_start) if hasattr(tz, 'localize') else london_start.replace(tzinfo=tz)
    
    london_end = datetime.combine(current_date, LONDON_END)
    london_end = tz.localize(london_end) if hasattr(tz, 'localize') else london_end.replace(tzinfo=tz)
    
    # Only include bars up to current time and within session
    london_end_actual = min(london_end, current_ny)
    
    london_mask = (df['time_ny'] >= london_start) & (df['time_ny'] < london_end_actual)
    london_bars = df.loc[london_mask]
    
    if not london_bars.empty:
        levels.london_high = london_bars['high'].max()
        levels.london_low = london_bars['low'].min()
        levels.london_range = levels.london_high - levels.london_low
    
    # Combined overnight (Asian + London)
    if levels.asian_high > 0 or levels.london_high > 0:
        levels.overnight_high = max(
            levels.asian_high if levels.asian_high > 0 else 0,
            levels.london_high if levels.london_high > 0 else 0
        )
        levels.overnight_low = min(
            levels.asian_low if levels.asian_low > 0 else float('inf'),
            levels.london_low if levels.london_low > 0 else float('inf')
        )
        if levels.overnight_low == float('inf'):
            levels.overnight_low = 0.0
    
    # Cache the result only if London is complete (levels are final for the day)
    current_t = current_ny.time()
    if current_t >= LONDON_END:
        _session_cache[current_date] = levels
    
    return levels


def is_in_trade_window(current_time: pd.Timestamp, tz: ZoneInfo = NY_TZ) -> bool:
    """
    Check if current time is within the trade window (9:30 - 11:30 NY).
    
    Args:
        current_time: Current timestamp
        tz: Timezone
        
    Returns:
        True if within trade window
    """
    current_ny = current_time.astimezone(tz)
    current_t = current_ny.time()
    
    return TRADE_WINDOW_START <= current_t <= TRADE_WINDOW_END


def is_london_complete(current_time: pd.Timestamp, tz: ZoneInfo = NY_TZ) -> bool:
    """
    Check if London session is complete (past 8:30 AM NY).
    
    The London low/high must be established before trading.
    
    Args:
        current_time: Current timestamp
        tz: Timezone
        
    Returns:
        True if London session is complete
    """
    current_ny = current_time.astimezone(tz)
    current_t = current_ny.time()
    
    return current_t >= LONDON_END


def get_overnight_levels_for_trade(
    df_1m: pd.DataFrame,
    current_time: pd.Timestamp,
    tz: ZoneInfo = NY_TZ
) -> Optional[SessionLevels]:
    """
    Get overnight levels only if:
    1. We are in the trade window (9:30-11:30)
    2. London session is complete (past 8:30)
    
    Returns None if conditions not met.
    """
    if not is_in_trade_window(current_time, tz):
        return None
    
    if not is_london_complete(current_time, tz):
        return None
    
    return compute_session_levels(df_1m, current_time, tz)

"""
Opening Range False Break Trigger

Detects when OR is broken early but price comes back inside (trap).
Fade it back to the other side of the range.

Usage:
    trigger = ORFalseBreakTrigger(
        or_minutes=15,           # NY 9:30-9:45
        max_return_minutes=30,   # Must return within 30 mins
        session="NY"
    )
"""

from typing import Dict, Any, Optional
import pandas as pd

from src.policy.triggers.base import Trigger, TriggerResult, TriggerDirection
from src.config import NY_TZ


class ORFalseBreakTrigger(Trigger):
    """
    Trigger for Opening Range false break pattern.
    
    Strategy: If we break the OR early and then come right back inside
    within the first hour, that feels like a trap → fade back to the 
    other side of the range.
    
    Logic:
    1. Establish OR during first N minutes (default 15 = 9:30-9:45 NY)
    2. Detect breakout (price closes outside OR)
    3. If price closes back inside OR within max_return_minutes → FADE
    4. Target: Other side of the range
    
    For break above OR_high then return inside:
        → SHORT (fade back to OR_low)
        
    For break below OR_low then return inside:
        → LONG (fade back to OR_high)
    """
    
    def __init__(
        self,
        or_minutes: int = 15,          # How long to establish OR
        max_return_minutes: int = 30,  # Max time for price to return
        session: str = "NY",           # NY, LONDON, or ASIA
        require_close_break: bool = True,  # Require close outside OR, not just wick
    ):
        self._or_minutes = or_minutes
        self._max_return = max_return_minutes
        self._session = session.upper()
        self._require_close = require_close_break
        
        # State tracking
        self._or_high = 0
        self._or_low = 0
        self._or_established = False
        self._break_direction: Optional[str] = None  # "ABOVE" or "BELOW"
        self._break_bar: int = 0
        self._break_time: Optional[pd.Timestamp] = None
        self._current_date: Optional[pd.Timestamp] = None
        self._or_bars = []
        self._triggered_today = False
        
    @property
    def trigger_id(self) -> str:
        return f"or_false_break_{self._or_minutes}m_{self._session.lower()}"
    
    @property
    def params(self) -> Dict[str, Any]:
        return {
            "or_minutes": self._or_minutes,
            "max_return_minutes": self._max_return,
            "session": self._session
        }
    
    def _get_session_start(self) -> tuple[int, int]:
        """Get session start hour, minute based on session type."""
        if self._session == "NY":
            return (9, 30)
        elif self._session == "LONDON":
            return (3, 0)  # 3:00 AM NY time (8:00 London)
        elif self._session == "ASIA":
            return (19, 0)  # 7:00 PM NY time (previous day)
        else:
            return (9, 30)  # Default NY
    
    def _is_or_period(self, t: pd.Timestamp) -> bool:
        """Check if we're in the OR establishment period."""
        try:
            ny_time = t.astimezone(NY_TZ) if t.tzinfo else t.tz_localize(NY_TZ)
        except:
            return False
            
        hour, minute = ny_time.hour, ny_time.minute
        start_h, start_m = self._get_session_start()
        
        mins_since_midnight = hour * 60 + minute
        or_start = start_h * 60 + start_m
        or_end = or_start + self._or_minutes
        
        return or_start <= mins_since_midnight < or_end
    
    def _is_after_or(self, t: pd.Timestamp) -> bool:
        """Check if we're past the OR period."""
        try:
            ny_time = t.astimezone(NY_TZ) if t.tzinfo else t.tz_localize(NY_TZ)
        except:
            return False
            
        hour, minute = ny_time.hour, ny_time.minute
        start_h, start_m = self._get_session_start()
        
        mins_since_midnight = hour * 60 + minute
        or_end = start_h * 60 + start_m + self._or_minutes
        
        return mins_since_midnight >= or_end
    
    def _is_within_return_window(self, break_time: pd.Timestamp, current_time: pd.Timestamp) -> bool:
        """Check if we're still within max_return_minutes of the break."""
        if break_time is None:
            return False
        diff = (current_time - break_time).total_seconds() / 60
        return diff <= self._max_return
    
    def _is_new_day(self, t: pd.Timestamp) -> bool:
        """Check if this is a new trading day."""
        if self._current_date is None:
            return True
        try:
            ny_time = t.astimezone(NY_TZ) if t.tzinfo else t.tz_localize(NY_TZ)
            return ny_time.date() != self._current_date.date()
        except:
            return True
    
    def check(self, features, **kwargs) -> TriggerResult:
        """Check for OR false break pattern."""
        timestamp = getattr(features, 'timestamp', None)
        if timestamp is None:
            return TriggerResult(trigger_id=self.trigger_id, triggered=False)
        
        # Reset on new day
        if self._is_new_day(timestamp):
            try:
                self._current_date = timestamp.astimezone(NY_TZ) if timestamp.tzinfo else timestamp.tz_localize(NY_TZ)
            except:
                self._current_date = timestamp
            self._or_high = 0
            self._or_low = 0
            self._or_established = False
            self._break_direction = None
            self._break_time = None
            self._or_bars = []
            self._triggered_today = False
        
        bar_high = getattr(features, 'bar_high', 0)
        bar_low = getattr(features, 'bar_low', 0)
        bar_close = getattr(features, 'bar_close', 0) or getattr(features, 'current_price', 0)
        bar_idx = getattr(features, 'bar_idx', 0)
        
        # Collect bars during OR period
        if self._is_or_period(timestamp) and not self._or_established:
            self._or_bars.append({'high': bar_high, 'low': bar_low})
            return TriggerResult(trigger_id=self.trigger_id, triggered=False)
        
        # Establish OR after period ends
        if self._is_after_or(timestamp) and not self._or_established and len(self._or_bars) > 0:
            self._or_high = max(b['high'] for b in self._or_bars)
            self._or_low = min(b['low'] for b in self._or_bars)
            self._or_established = True
        
        # Can't trigger if OR not established or already triggered today
        if not self._or_established or self._triggered_today:
            return TriggerResult(trigger_id=self.trigger_id, triggered=False)
        
        # STATE MACHINE:
        # 1. Look for breakout
        # 2. If breakout detected, look for return inside
        
        triggered = False
        direction = TriggerDirection.NEUTRAL
        
        if self._break_direction is None:
            # Look for breakout
            if self._require_close:
                broke_above = bar_close > self._or_high
                broke_below = bar_close < self._or_low
            else:
                broke_above = bar_high > self._or_high
                broke_below = bar_low < self._or_low
            
            if broke_above:
                self._break_direction = "ABOVE"
                self._break_time = timestamp
                self._break_bar = bar_idx
            elif broke_below:
                self._break_direction = "BELOW"
                self._break_time = timestamp
                self._break_bar = bar_idx
                
        else:
            # Already have a breakout - look for return inside
            if not self._is_within_return_window(self._break_time, timestamp):
                # Timeout - reset break tracking
                self._break_direction = None
                self._break_time = None
            else:
                # Check if we're back inside the range
                inside_range = self._or_low <= bar_close <= self._or_high
                
                if inside_range:
                    # FALSE BREAK DETECTED - FADE IT
                    self._triggered_today = True
                    triggered = True
                    
                    if self._break_direction == "ABOVE":
                        # Broke above then came back = SHORT (fade to OR_low)
                        direction = TriggerDirection.SHORT
                    else:
                        # Broke below then came back = LONG (fade to OR_high)
                        direction = TriggerDirection.LONG
        
        return TriggerResult(
            trigger_id=self.trigger_id,
            triggered=triggered,
            direction=direction,
            context={
                "or_high": self._or_high,
                "or_low": self._or_low,
                "or_range": self._or_high - self._or_low,
                "break_direction": self._break_direction,
                "target": self._or_low if direction == TriggerDirection.SHORT else self._or_high,
                "session": self._session
            }
        )
    
    def reset(self):
        """Reset state for new run."""
        self._or_high = 0
        self._or_low = 0
        self._or_established = False
        self._break_direction = None
        self._break_time = None
        self._current_date = None
        self._or_bars = []
        self._triggered_today = False

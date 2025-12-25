"""
VWAP Reclaim Trigger

Detects when price reclaims VWAP after being below it and holds.
Perfect for power hour reversals.

Usage:
    trigger = VWAPReclaimTrigger(
        min_time="14:30",        # After 2:30 PM ET
        hold_minutes=10,         # Hold above VWAP for 10 mins
        min_below_minutes=30     # Must have been below for at least 30 mins
    )
"""

from typing import Dict, Any, Optional
import pandas as pd

from src.policy.triggers.base import Trigger, TriggerResult, TriggerDirection
from src.config import NY_TZ


class VWAPReclaimTrigger(Trigger):
    """
    Trigger for VWAP reclaim pattern.
    
    Strategy: After 2:30 PM, if price was under VWAP all morning,
    then reclaims and holds for 10 minutes → long to PDH/day high.
    
    Logic:
    1. Only trigger after min_time (default 14:30 = 2:30 PM ET)
    2. Track if price has been below VWAP for min_below_minutes
    3. Detect when close > VWAP (reclaim)
    4. Wait for hold_minutes consecutive closes above VWAP
    5. Trigger LONG when hold confirmed
    
    Required features: vwap_session (must be in FeatureBundle)
    """
    
    def __init__(
        self,
        min_time: str = "14:30",      # Earliest trigger time (HH:MM in ET)
        hold_minutes: int = 10,        # Hold above VWAP requirement
        min_below_minutes: int = 30,   # Must have been below for this long
        direction: str = "LONG"        # Usually LONG on reclaim
    ):
        self._min_time = min_time
        self._hold_minutes = hold_minutes
        self._min_below = min_below_minutes
        self._direction = direction
        
        # State tracking
        self._current_date: Optional[pd.Timestamp] = None
        self._below_vwap_since: Optional[pd.Timestamp] = None
        self._reclaim_time: Optional[pd.Timestamp] = None
        self._consecutive_above: int = 0
        self._triggered_today = False
        
    @property
    def trigger_id(self) -> str:
        return f"vwap_reclaim_{self._min_time.replace(':', '')}_{self._hold_minutes}m"
    
    @property
    def params(self) -> Dict[str, Any]:
        return {
            "min_time": self._min_time,
            "hold_minutes": self._hold_minutes,
            "min_below_minutes": self._min_below,
            "direction": self._direction
        }
    
    def _is_new_day(self, t: pd.Timestamp) -> bool:
        if self._current_date is None:
            return True
        try:
            ny_time = t.astimezone(NY_TZ) if t.tzinfo else t.tz_localize(NY_TZ)
            return ny_time.date() != self._current_date.date()
        except:
            return True
    
    def _is_after_min_time(self, t: pd.Timestamp) -> bool:
        try:
            ny_time = t.astimezone(NY_TZ) if t.tzinfo else t.tz_localize(NY_TZ)
            hour, minute = map(int, self._min_time.split(':'))
            current_mins = ny_time.hour * 60 + ny_time.minute
            min_mins = hour * 60 + minute
            return current_mins >= min_mins
        except:
            return False
    
    def check(self, features, **kwargs) -> TriggerResult:
        """Check for VWAP reclaim pattern."""
        timestamp = getattr(features, 'timestamp', None)
        if timestamp is None:
            return TriggerResult(trigger_id=self.trigger_id, triggered=False)
        
        # Reset on new day
        if self._is_new_day(timestamp):
            try:
                self._current_date = timestamp.astimezone(NY_TZ) if timestamp.tzinfo else timestamp.tz_localize(NY_TZ)
            except:
                self._current_date = timestamp
            self._below_vwap_since = None
            self._reclaim_time = None
            self._consecutive_above = 0
            self._triggered_today = False
        
        # Get VWAP - check multiple possible attribute names
        vwap = getattr(features, 'vwap_session', None) or getattr(features, 'vwap', None)
        if vwap is None or vwap <= 0:
            return TriggerResult(trigger_id=self.trigger_id, triggered=False)
        
        current_price = getattr(features, 'current_price', 0) or getattr(features, 'bar_close', 0)
        if current_price <= 0:
            return TriggerResult(trigger_id=self.trigger_id, triggered=False)
        
        # Track below VWAP period
        if current_price < vwap:
            if self._below_vwap_since is None:
                self._below_vwap_since = timestamp
            self._consecutive_above = 0
            self._reclaim_time = None
        else:
            # Price is above VWAP
            if self._reclaim_time is None and self._below_vwap_since is not None:
                # Just reclaimed!
                self._reclaim_time = timestamp
            self._consecutive_above += 1
        
        # Check trigger conditions
        if self._triggered_today:
            return TriggerResult(trigger_id=self.trigger_id, triggered=False)
        
        if not self._is_after_min_time(timestamp):
            return TriggerResult(trigger_id=self.trigger_id, triggered=False)
        
        if self._below_vwap_since is None:
            return TriggerResult(trigger_id=self.trigger_id, triggered=False)
        
        # Check minimum below time
        below_duration = (self._reclaim_time or timestamp) - self._below_vwap_since
        below_minutes = below_duration.total_seconds() / 60
        if below_minutes < self._min_below:
            return TriggerResult(trigger_id=self.trigger_id, triggered=False)
        
        # Check hold requirement
        if self._consecutive_above < self._hold_minutes:
            return TriggerResult(trigger_id=self.trigger_id, triggered=False)
        
        # TRIGGER!
        self._triggered_today = True
        
        direction = TriggerDirection.LONG if self._direction == "LONG" else TriggerDirection.SHORT
        
        return TriggerResult(
            trigger_id=self.trigger_id,
            triggered=True,
            direction=direction,
            context={
                "vwap": vwap,
                "current_price": current_price,
                "below_minutes": below_minutes,
                "hold_minutes": self._consecutive_above,
                "reclaim_time": str(self._reclaim_time) if self._reclaim_time else None
            }
        )
    
    def reset(self):
        self._current_date = None
        self._below_vwap_since = None
        self._reclaim_time = None
        self._consecutive_above = 0
        self._triggered_today = False

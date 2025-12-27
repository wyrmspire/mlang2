"""
Time Trigger

Fires at specific times of day. Simple and predictable for testing.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import pandas as pd
from zoneinfo import ZoneInfo

from .base import Trigger, TriggerResult, TriggerDirection
from src.config import NY_TZ


class TimeTrigger(Trigger):
    """
    Trigger that fires at specific time(s) daily.
    
    Agent config examples:
        {"type": "time", "hour": 10, "minute": 0}
        {"type": "time", "hour": 10, "minute": 0, "direction": "LONG"}
        {"type": "time", "hours": [10, 14], "minute": 0}  # Multiple times
    """
    
    def __init__(
        self,
        hour: Optional[int] = None,
        minute: int = 0,
        hours: Optional[List[int]] = None,
        time: Optional[str] = None,  # Accept "HH:MM" format from agent
        direction: str = "NEUTRAL",
        timezone: str = "America/New_York",
    ):
        # Support 'time' parameter as string (e.g., "10:00", "11:30")
        if time is not None and hour is None:
            parts = str(time).split(":")
            hour = int(parts[0])
            if len(parts) > 1:
                minute = int(parts[1])
        
        # Support single hour or multiple hours
        if hours is not None:
            self._hours = hours
        elif hour is not None:
            self._hours = [hour]
        else:
            raise ValueError("Must specify 'hour', 'hours', or 'time'")
        
        self._minute = minute
        self._direction = TriggerDirection[direction.upper()]
        self._tz = ZoneInfo(timezone)
        
        # Track last trigger date to avoid double-firing
        self._last_trigger_dates: Dict[int, Any] = {}
    
    @property
    def trigger_id(self) -> str:
        hours_str = "_".join(str(h) for h in self._hours)
        return f"time_{hours_str}_{self._minute:02d}"
    
    @property
    def params(self) -> Dict[str, Any]:
        return {
            "hours": self._hours,
            "minute": self._minute,
            "direction": self._direction.value,
        }
    
    def check(self, features) -> TriggerResult:
        """Check if current time matches target time."""
        t = features.timestamp
        if t is None:
            return TriggerResult(trigger_id=self.trigger_id, triggered=False)
        
        # Convert to target timezone
        local_time = t.astimezone(self._tz)
        current_hour = local_time.hour
        current_minute = local_time.minute
        current_date = local_time.date()
        
        # Check if current time matches any target hour
        for target_hour in self._hours:
            if current_hour == target_hour and current_minute == self._minute:
                # Check if we already triggered for this hour today
                if self._last_trigger_dates.get(target_hour) == current_date:
                    continue  # Already triggered
                
                # Mark as triggered
                self._last_trigger_dates[target_hour] = current_date
                
                return TriggerResult(
                    trigger_id=self.trigger_id,
                    triggered=True,
                    direction=self._direction,
                    context={
                        "hour": target_hour,
                        "minute": self._minute,
                        "local_time": local_time.strftime("%H:%M"),
                    },
                    confidence=1.0
                )
        
        return TriggerResult(trigger_id=self.trigger_id, triggered=False)
    
    def reset(self):
        """Reset trigger state (for new simulation runs)."""
        self._last_trigger_dates.clear()

"""
Fakeout Trigger

Detects level breaks that fail and close back through the level.
Examples: PDH fakeout (break above, close back below) → SHORT
         PDL fakeout (break below, close back above) → LONG
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import pandas as pd

from src.policy.triggers.base import Trigger, TriggerResult, TriggerDirection
from src.features.pipeline import FeatureBundle


@dataclass
class FakeoutConfig:
    """Configuration for fakeout detection."""
    level: str = "pdh"  # pdh, pdl
    buffer_points: float = 0.0  # Buffer for level break detection


class FakeoutTrigger(Trigger):
    """
    Trigger for level fakeout detection.
    
    A fakeout occurs when:
    1. Price breaks through a key level (PDH/PDL)
    2. But fails to hold and closes back through it
    
    For PDH fakeout (SHORT):
        - Bar high > PDH (broke above)
        - Bar close < PDH (closed back below)
        
    For PDL fakeout (LONG):
        - Bar low < PDL (broke below)
        - Bar close > PDL (closed back above)
    
    Agent config:
        {"type": "fakeout", "level": "pdh"}  # SHORT on PDH fakeout
        {"type": "fakeout", "level": "pdl"}  # LONG on PDL fakeout
    """
    
    def __init__(
        self,
        level: str = "pdh",
        buffer_points: float = 0.0
    ):
        self._level = level.lower()
        self._buffer = buffer_points
        
    @property
    def trigger_id(self) -> str:
        return f"fakeout_{self._level}"
    
    @property
    def params(self) -> Dict[str, Any]:
        return {
            "level": self._level,
            "buffer_points": self._buffer
        }
    
    def check(self, features: FeatureBundle, **kwargs) -> TriggerResult:
        """
        Check for fakeout pattern.
        
        Requires features to have:
        - pdh, pdl: Previous day high/low levels
        - current_bar or last bar data with high, low, close
        """
        # Get the level value
        if self._level == "pdh":
            level_value = getattr(features, 'pdh', None)
            if level_value is None:
                level_value = features.levels.get('pdh') if hasattr(features, 'levels') else None
        elif self._level == "pdl":
            level_value = getattr(features, 'pdl', None)
            if level_value is None:
                level_value = features.levels.get('pdl') if hasattr(features, 'levels') else None
        else:
            return TriggerResult(
                trigger_id=self.trigger_id,
                triggered=False,
                context={"error": f"Unknown level: {self._level}"}
            )
        
        if level_value is None or level_value == 0:
            return TriggerResult(
                trigger_id=self.trigger_id,
                triggered=False,
                context={"error": "Level not available"}
            )
        
        # Get current bar OHLC
        bar_high = features.bar_high if hasattr(features, 'bar_high') else None
        bar_low = features.bar_low if hasattr(features, 'bar_low') else None
        bar_close = features.bar_close if hasattr(features, 'bar_close') else features.current_price
        
        # Fallback to market_state if available
        if bar_high is None and hasattr(features, 'market_state'):
            bar_high = features.market_state.bar_high
            bar_low = features.market_state.bar_low
            bar_close = features.market_state.bar_close
        
        if bar_high is None or bar_low is None:
            return TriggerResult(
                trigger_id=self.trigger_id,
                triggered=False,
                context={"error": "Bar OHLC not available"}
            )
        
        # Check for fakeout pattern
        level_with_buffer = level_value + self._buffer if self._level == "pdh" else level_value - self._buffer
        
        if self._level == "pdh":
            # PDH Fakeout: broke above but closed below
            broke_level = bar_high > level_with_buffer
            closed_back = bar_close < level_value
            direction = TriggerDirection.SHORT
        else:
            # PDL Fakeout: broke below but closed above
            broke_level = bar_low < level_with_buffer
            closed_back = bar_close > level_value
            direction = TriggerDirection.LONG
        
        triggered = broke_level and closed_back
        
        return TriggerResult(
            trigger_id=self.trigger_id,
            triggered=triggered,
            direction=direction if triggered else TriggerDirection.NEUTRAL,
            context={
                "level": self._level,
                "level_value": level_value,
                "bar_high": bar_high,
                "bar_low": bar_low,
                "bar_close": bar_close,
                "broke_level": broke_level,
                "closed_back": closed_back,
                "fakeout_type": f"{self._level.upper()}_FAKEOUT" if triggered else None
            }
        )
    
    def reset(self):
        """Reset any state (stateless trigger, nothing to reset)."""
        pass

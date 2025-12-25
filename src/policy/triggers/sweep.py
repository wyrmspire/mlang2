"""
Sweep/Stop Run Trigger

Detects stop runs where price barely takes out a level then immediately reverses.
Perfect for fading PDH/PDL sweeps (liquidity grabs).

Usage:
    # Fade PDH sweep (short after barely taking out PDH)
    trigger = SweepTrigger(level="pdh", sweep_type="fade")
    
    # Fade PDL sweep (long after barely taking out PDL)
    trigger = SweepTrigger(level="pdl", sweep_type="fade")
"""

from typing import Dict, Any
import pandas as pd

from src.policy.triggers.base import Trigger, TriggerResult, TriggerDirection


class SweepTrigger(Trigger):
    """
    Trigger for stop run/sweep patterns.
    
    A sweep occurs when:
    1. Price barely takes out a key level (PDH, PDL, prior swing)
    2. The move is small (within max_sweep_pts of level)
    3. Price closes back on the opposite side of the level
    
    For PDH sweep (SHORT fade):
        - Bar high > PDH (swept above)
        - Sweep distance = bar_high - PDH < max_sweep
        - Bar close < PDH (closed back below)
        - → SHORT signal
        
    For PDL sweep (LONG fade):
        - Bar low < PDL (swept below)
        - Sweep distance = PDL - bar_low < max_sweep
        - Bar close > PDL (closed back above)
        - → LONG signal
    """
    
    def __init__(
        self,
        level: str = "pdh",
        sweep_type: str = "fade",  # "fade" to trade against sweep
        max_sweep_pts: float = 5.0,  # Max points beyond level to count as "barely"
        min_sweep_pts: float = 0.25,  # Min points beyond level (must actually sweep)
    ):
        self._level = level.lower()
        self._sweep_type = sweep_type
        self._max_sweep = max_sweep_pts
        self._min_sweep = min_sweep_pts
        
    @property
    def trigger_id(self) -> str:
        return f"sweep_{self._level}_{self._sweep_type}"
    
    @property
    def params(self) -> Dict[str, Any]:
        return {
            "level": self._level,
            "sweep_type": self._sweep_type,
            "max_sweep_pts": self._max_sweep,
            "min_sweep_pts": self._min_sweep
        }
    
    def check(self, features, **kwargs) -> TriggerResult:
        """
        Check for sweep pattern at the specified level.
        
        Requires features to have:
        - The level value (e.g., features.pdh, features.pdl)
        - bar_high, bar_low, bar_close
        """
        # Get level value
        level_value = getattr(features, self._level, None)
        
        if level_value is None or level_value == 0:
            return TriggerResult(
                trigger_id=self.trigger_id,
                triggered=False,
                context={"error": f"Level '{self._level}' not available"}
            )
        
        # Get bar OHLC
        bar_high = getattr(features, 'bar_high', None)
        bar_low = getattr(features, 'bar_low', None)
        bar_close = getattr(features, 'bar_close', None) or getattr(features, 'current_price', None)
        
        if bar_high is None or bar_low is None or bar_close is None:
            return TriggerResult(
                trigger_id=self.trigger_id,
                triggered=False,
                context={"error": "Bar OHLC not available"}
            )
        
        triggered = False
        direction = TriggerDirection.NEUTRAL
        sweep_type_detected = None
        sweep_distance = 0
        
        # Check for PDH/high level sweep (SHORT fade)
        if self._level in ["pdh", "high", "swing_high"]:
            swept = bar_high > level_value  # Took out the high
            sweep_distance = bar_high - level_value
            barely_swept = self._min_sweep <= sweep_distance <= self._max_sweep
            closed_back = bar_close < level_value  # Closed back below
            
            if swept and barely_swept and closed_back:
                triggered = True
                direction = TriggerDirection.SHORT if self._sweep_type == "fade" else TriggerDirection.LONG
                sweep_type_detected = f"{self._level}_sweep_fade"
                
        # Check for PDL/low level sweep (LONG fade)
        elif self._level in ["pdl", "low", "swing_low"]:
            swept = bar_low < level_value  # Took out the low
            sweep_distance = level_value - bar_low
            barely_swept = self._min_sweep <= sweep_distance <= self._max_sweep
            closed_back = bar_close > level_value  # Closed back above
            
            if swept and barely_swept and closed_back:
                triggered = True
                direction = TriggerDirection.LONG if self._sweep_type == "fade" else TriggerDirection.SHORT
                sweep_type_detected = f"{self._level}_sweep_fade"
        
        return TriggerResult(
            trigger_id=self.trigger_id,
            triggered=triggered,
            direction=direction,
            context={
                "level": self._level,
                "level_value": float(level_value),
                "bar_high": float(bar_high),
                "bar_low": float(bar_low),
                "bar_close": float(bar_close),
                "sweep_distance": float(sweep_distance),
                "sweep_type": sweep_type_detected,
                "direction": direction.value if triggered else None
            }
        )
    
    def reset(self):
        """Reset state (stateless trigger)."""
        pass

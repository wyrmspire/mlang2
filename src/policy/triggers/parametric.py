"""
Generic Rejection Trigger

Parametric trigger that detects rejections at any feature level.
The agent says: "rejection on ema_200" → RejectionTrigger(feature="ema_200")

Supported features:
- ema_X: EMA with period X (ema_20, ema_50, ema_200)
- pdh: Previous Day High
- pdl: Previous Day Low
- pdc: Previous Day Close
- vwap: VWAP
- level_X: Custom named level
"""

from typing import Dict, Any, Optional
import pandas as pd

from src.policy.triggers.base import Trigger, TriggerResult, TriggerDirection


class RejectionTrigger(Trigger):
    """
    Generic trigger for price rejections at any feature level.
    
    A rejection occurs when price touches a feature but closes back
    on the opposite side, forming a rejection candle.
    
    For LONG rejection (bearish feature above price):
        - Bar high touches/crosses above feature
        - Bar closes below feature
        - → SHORT signal
        
    For SHORT covering rejection (bullish feature below price):
        - Bar low touches/crosses below feature
        - Bar closes above feature
        - → LONG signal
    
    Usage:
        # Rejection at 200 EMA
        trigger = RejectionTrigger(feature="ema_200")
        
        # Rejection at PDH (Previous Day High)
        trigger = RejectionTrigger(feature="pdh")
        
        # Long-only rejection (only take bullish rejections)
        trigger = RejectionTrigger(feature="ema_50", direction="long_only")
    """
    
    def __init__(
        self,
        feature: str,
        direction: str = "both",  # "both", "long_only", "short_only"
        require_trend: bool = False,  # If True, require trend alignment
        trend_feature: Optional[str] = None,  # e.g., "ema_20_slope"
        trend_threshold: float = 0.0,
    ):
        self._feature = feature
        self._direction = direction
        self._require_trend = require_trend
        self._trend_feature = trend_feature
        self._trend_threshold = trend_threshold
        
    @property
    def trigger_id(self) -> str:
        return f"rejection_{self._feature}"
    
    @property
    def params(self) -> Dict[str, Any]:
        return {
            "feature": self._feature,
            "direction": self._direction,
            "require_trend": self._require_trend
        }
    
    def check(self, features, **kwargs) -> TriggerResult:
        """
        Check for rejection at the specified feature.
        
        The features object should have:
        - The feature value as an attribute (e.g., features.ema_200)
        - bar_high, bar_low, bar_close for the current candle
        """
        # Get feature value
        feature_value = getattr(features, self._feature, None)
        
        if feature_value is None or feature_value == 0:
            return TriggerResult(
                trigger_id=self.trigger_id,
                triggered=False,
                context={"error": f"Feature '{self._feature}' not available"}
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
        
        # Check for rejection patterns
        triggered = False
        direction = TriggerDirection.NEUTRAL
        rejection_type = None
        
        # Bearish rejection at feature above price → SHORT
        # Bar spiked up to touch feature but closed below it
        touched_above = bar_high >= feature_value
        closed_below = bar_close < feature_value
        
        if touched_above and closed_below and self._direction in ["both", "short_only"]:
            triggered = True
            direction = TriggerDirection.SHORT
            rejection_type = f"bearish_rejection_at_{self._feature}"
        
        # Bullish rejection at feature below price → LONG
        # Bar dipped down to touch feature but closed above it
        touched_below = bar_low <= feature_value
        closed_above = bar_close > feature_value
        
        if touched_below and closed_above and self._direction in ["both", "long_only"]:
            # Only trigger if not already triggered for SHORT (priority: first match)
            if not triggered:
                triggered = True
                direction = TriggerDirection.LONG
                rejection_type = f"bullish_rejection_at_{self._feature}"
        
        # Optional trend filter
        if triggered and self._require_trend:
            trend_value = getattr(features, self._trend_feature, 0) if self._trend_feature else 0
            
            # For LONG, want positive trend; for SHORT, want negative trend
            trend_aligned = (
                (direction == TriggerDirection.LONG and trend_value > self._trend_threshold) or
                (direction == TriggerDirection.SHORT and trend_value < -self._trend_threshold)
            )
            
            if not trend_aligned:
                triggered = False
                rejection_type = None
        
        return TriggerResult(
            trigger_id=self.trigger_id,
            triggered=triggered,
            direction=direction,
            context={
                "feature": self._feature,
                "feature_value": float(feature_value),
                "bar_high": float(bar_high),
                "bar_low": float(bar_low),
                "bar_close": float(bar_close),
                "touched_above": touched_above,
                "touched_below": touched_below,
                "rejection_type": rejection_type,
                "direction": direction.value if triggered else None
            }
        )
    
    def reset(self):
        """Reset state (stateless trigger)."""
        pass


class ComparisonTrigger(Trigger):
    """
    Generic trigger for comparing two features.
    
    Usage:
        # 20 EMA crosses above 200 EMA
        trigger = ComparisonTrigger(
            feature_a="ema_20",
            feature_b="ema_200", 
            condition="crosses_above"
        )
        
        # RSI below 30
        trigger = ComparisonTrigger(
            feature_a="rsi_14",
            feature_b=30,  # Can be a constant
            condition="below"
        )
        
        # Price above VWAP
        trigger = ComparisonTrigger(
            feature_a="current_price",
            feature_b="vwap",
            condition="above"
        )
    
    Conditions:
        - "above": A > B
        - "below": A < B
        - "crosses_above": A crossed above B (was below, now above)
        - "crosses_below": A crossed below B (was above, now below)
    """
    
    def __init__(
        self,
        feature_a: str,
        feature_b: str | float,  # Can be feature name or constant
        condition: str = "above",  # "above", "below", "crosses_above", "crosses_below"
        direction_on_true: str = "LONG",  # What direction when condition is true
    ):
        self._feature_a = feature_a
        self._feature_b = feature_b
        self._condition = condition
        self._direction_on_true = direction_on_true
        
        # For cross detection, track previous state
        self._prev_a_above_b: Optional[bool] = None
        
    @property
    def trigger_id(self) -> str:
        b_str = self._feature_b if isinstance(self._feature_b, str) else f"const_{self._feature_b}"
        return f"compare_{self._feature_a}_{self._condition}_{b_str}"
    
    @property
    def params(self) -> Dict[str, Any]:
        return {
            "feature_a": self._feature_a,
            "feature_b": self._feature_b,
            "condition": self._condition,
            "direction_on_true": self._direction_on_true
        }
    
    def check(self, features, **kwargs) -> TriggerResult:
        """Check comparison between features."""
        # Get feature A value
        value_a = getattr(features, self._feature_a, None)
        if value_a is None:
            return TriggerResult(
                trigger_id=self.trigger_id,
                triggered=False,
                context={"error": f"Feature '{self._feature_a}' not available"}
            )
        
        # Get feature B value (constant or feature)
        if isinstance(self._feature_b, (int, float)):
            value_b = self._feature_b
        else:
            value_b = getattr(features, self._feature_b, None)
            if value_b is None:
                return TriggerResult(
                    trigger_id=self.trigger_id,
                    triggered=False,
                    context={"error": f"Feature '{self._feature_b}' not available"}
                )
        
        # Compute current state
        a_above_b = value_a > value_b
        
        # Check condition
        triggered = False
        
        if self._condition == "above":
            triggered = a_above_b
        elif self._condition == "below":
            triggered = not a_above_b
        elif self._condition == "crosses_above":
            # Was below, now above
            if self._prev_a_above_b is not None:
                triggered = not self._prev_a_above_b and a_above_b
        elif self._condition == "crosses_below":
            # Was above, now below
            if self._prev_a_above_b is not None:
                triggered = self._prev_a_above_b and not a_above_b
        
        # Update previous state for cross detection
        self._prev_a_above_b = a_above_b
        
        # Determine direction
        if triggered:
            direction = TriggerDirection.LONG if self._direction_on_true == "LONG" else TriggerDirection.SHORT
        else:
            direction = TriggerDirection.NEUTRAL
        
        return TriggerResult(
            trigger_id=self.trigger_id,
            triggered=triggered,
            direction=direction,
            context={
                "feature_a": self._feature_a,
                "value_a": float(value_a),
                "feature_b": self._feature_b,
                "value_b": float(value_b) if isinstance(value_b, (int, float)) else value_b,
                "condition": self._condition,
                "a_above_b": a_above_b
            }
        )
    
    def reset(self):
        """Reset cross detection state."""
        self._prev_a_above_b = None

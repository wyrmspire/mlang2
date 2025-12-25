"""
EMA 200 Rejection Trigger

Detects trending pullbacks that reject at the 200 EMA:
- Price in trend (20 EMA angled)
- Pullback touches/crosses 200 EMA
- Candle shows rejection (wick, close back on trend side)
- Near a key level for confluence

Agent config:
    {"type": "ema200_rejection"}
"""

from typing import Dict, Any
import pandas as pd
import numpy as np

from src.policy.triggers.base import Trigger, TriggerResult, TriggerDirection
from src.features.pipeline import FeatureBundle


class EMA200RejectionTrigger(Trigger):
    """
    Trigger for 200 EMA rejection pullbacks in trending markets.
    
    Strategy Logic:
    1. 20 EMA must be angled (trending) - slope > threshold
    2. Price pulls back to touch/cross 200 EMA
    3. Candle shows rejection (closes back on trend side)
    4. Optionally: near a key level (PDH/PDL/HTF)
    
    For UPTREND (20 > 200, 20 EMA sloping up):
        - Price touches/dips below 200 EMA
        - Candle closes above 200 EMA → LONG
        
    For DOWNTREND (20 < 200, 20 EMA sloping down):
        - Price touches/spikes above 200 EMA
        - Candle closes below 200 EMA → SHORT
    """
    
    def __init__(
        self,
        ema_fast_period: int = 20,
        ema_slow_period: int = 200,
        slope_threshold: float = 0.02,  # Minimum slope for "trending"
        slope_lookback: int = 5,  # Bars to measure slope
        require_touch: bool = True,  # Must touch the 200 EMA
        level_proximity_atr: float = 0.0  # If > 0, require near level
    ):
        self._fast_period = ema_fast_period
        self._slow_period = ema_slow_period
        self._slope_threshold = slope_threshold
        self._slope_lookback = slope_lookback
        self._require_touch = require_touch
        self._level_proximity = level_proximity_atr
        
    @property
    def trigger_id(self) -> str:
        return f"ema{self._slow_period}_rejection"
    
    @property
    def params(self) -> Dict[str, Any]:
        return {
            "ema_fast": self._fast_period,
            "ema_slow": self._slow_period,
            "slope_threshold": self._slope_threshold,
            "slope_lookback": self._slope_lookback
        }
    
    def check(self, features: FeatureBundle, **kwargs) -> TriggerResult:
        """
        Check for EMA 200 rejection pattern.
        
        Requires features to have:
        - ema_20: 20 EMA value
        - ema_200: 200 EMA value
        - ema_20_prev: Previous 20 EMA values for slope (or we compute from history)
        - bar_high, bar_low, bar_close
        
        Or pass df_5m/df_15m in kwargs for computing EMAs.
        """
        # Get EMA values
        ema_fast = getattr(features, f'ema_{self._fast_period}', None)
        ema_slow = getattr(features, f'ema_{self._slow_period}', None)
        
        # Also check alternative attribute names
        if ema_fast is None:
            ema_fast = getattr(features, 'ema_20', None)
        if ema_slow is None:
            ema_slow = getattr(features, 'ema_200', None)
            
        if ema_fast is None or ema_slow is None:
            return TriggerResult(
                trigger_id=self.trigger_id,
                triggered=False,
                context={"error": "EMA values not available"}
            )
        
        # Get bar OHLC
        bar_high = getattr(features, 'bar_high', None)
        bar_low = getattr(features, 'bar_low', None)
        bar_close = getattr(features, 'bar_close', features.current_price)
        
        if bar_high is None or bar_low is None:
            return TriggerResult(
                trigger_id=self.trigger_id,
                triggered=False,
                context={"error": "Bar OHLC not available"}
            )
        
        # Get EMA slope (requires previous values)
        ema_fast_slope = getattr(features, 'ema_20_slope', 0)
        
        # Determine trend
        uptrend = ema_fast > ema_slow and ema_fast_slope > self._slope_threshold
        downtrend = ema_fast < ema_slow and ema_fast_slope < -self._slope_threshold
        
        if not uptrend and not downtrend:
            return TriggerResult(
                trigger_id=self.trigger_id,
                triggered=False,
                context={
                    "reason": "No clear trend",
                    "ema_fast": ema_fast,
                    "ema_slow": ema_slow,
                    "slope": ema_fast_slope
                }
            )
        
        # Check for rejection pattern
        triggered = False
        direction = TriggerDirection.NEUTRAL
        rejection_type = None
        
        if uptrend:
            # LONG setup: Bar touched/crossed below 200 EMA, closed above
            touched_200 = bar_low <= ema_slow
            closed_above = bar_close > ema_slow
            
            if touched_200 and closed_above:
                triggered = True
                direction = TriggerDirection.LONG
                rejection_type = "bullish_200_rejection"
                
        elif downtrend:
            # SHORT setup: Bar touched/crossed above 200 EMA, closed below
            touched_200 = bar_high >= ema_slow
            closed_below = bar_close < ema_slow
            
            if touched_200 and closed_below:
                triggered = True
                direction = TriggerDirection.SHORT
                rejection_type = "bearish_200_rejection"
        
        # Optional level proximity check
        if triggered and self._level_proximity > 0:
            atr = getattr(features, 'atr', 5.0)
            pdh = getattr(features, 'pdh', 0)
            pdl = getattr(features, 'pdl', 0)
            
            near_level = False
            if pdh > 0:
                dist_pdh = abs(bar_close - pdh) / atr
                if dist_pdh < self._level_proximity:
                    near_level = True
            if pdl > 0:
                dist_pdl = abs(bar_close - pdl) / atr
                if dist_pdl < self._level_proximity:
                    near_level = True
            
            if not near_level:
                triggered = False
        
        return TriggerResult(
            trigger_id=self.trigger_id,
            triggered=triggered,
            direction=direction,
            context={
                "ema_fast": float(ema_fast),
                "ema_slow": float(ema_slow),
                "ema_slope": float(ema_fast_slope),
                "bar_high": float(bar_high),
                "bar_low": float(bar_low),
                "bar_close": float(bar_close),
                "uptrend": uptrend,
                "downtrend": downtrend,
                "rejection_type": rejection_type
            }
        )
    
    def reset(self):
        """Reset state (stateless trigger)."""
        pass

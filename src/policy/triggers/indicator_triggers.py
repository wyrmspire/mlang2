"""
Indicator-Based Triggers

EMA crossovers, RSI thresholds, etc.
"""

from typing import Dict, Any, Optional
import numpy as np

from .base import Trigger, TriggerResult, TriggerDirection


class EMACrossTrigger(Trigger):
    """
    Trigger on EMA crossover.
    
    Agent config examples:
        {"type": "ema_cross", "fast": 9, "slow": 21}
        {"type": "ema_cross", "fast": 9, "slow": 21, "timeframe": "5m"}
    """
    
    def __init__(
        self,
        fast: int = 9,
        slow: int = 21,
        timeframe: str = "5m",
    ):
        self._fast = fast
        self._slow = slow
        self._timeframe = timeframe
        self._prev_fast_above = None  # Track previous state for crossover
    
    @property
    def trigger_id(self) -> str:
        return f"ema_cross_{self._fast}_{self._slow}_{self._timeframe}"
    
    @property
    def params(self) -> Dict[str, Any]:
        return {
            "fast": self._fast,
            "slow": self._slow,
            "timeframe": self._timeframe,
        }
    
    def check(self, features) -> TriggerResult:
        """Check for EMA crossover."""
        # Get indicator values from features
        indicators = features.indicators
        if indicators is None:
            return TriggerResult(trigger_id=self.trigger_id, triggered=False)
        
        # Look up EMA values based on timeframe
        # Features store: ema_5m_20, ema_15m_20, ema_5m_200, etc.
        fast_key = f"ema_{self._timeframe}_{self._fast}"
        slow_key = f"ema_{self._timeframe}_{self._slow}"
        
        # Try to get values - fall back to common periods
        fast_ema = getattr(indicators, fast_key, None)
        slow_ema = getattr(indicators, slow_key, None)
        
        # Fallback: use available EMAs if exact match not found
        if fast_ema is None:
            fast_ema = getattr(indicators, 'ema_5m_20', 0)
        if slow_ema is None:
            slow_ema = getattr(indicators, 'ema_5m_200', 0)
        
        if fast_ema == 0 or slow_ema == 0:
            return TriggerResult(trigger_id=self.trigger_id, triggered=False)
        
        fast_above = fast_ema > slow_ema
        
        # Detect crossover
        if self._prev_fast_above is not None:
            if fast_above and not self._prev_fast_above:
                # Bullish cross: fast crossed above slow
                self._prev_fast_above = fast_above
                return TriggerResult(
                    trigger_id=self.trigger_id,
                    triggered=True,
                    direction=TriggerDirection.LONG,
                    context={
                        "cross_type": "bullish",
                        "fast_ema": fast_ema,
                        "slow_ema": slow_ema,
                    },
                    confidence=0.9
                )
            elif not fast_above and self._prev_fast_above:
                # Bearish cross: fast crossed below slow
                self._prev_fast_above = fast_above
                return TriggerResult(
                    trigger_id=self.trigger_id,
                    triggered=True,
                    direction=TriggerDirection.SHORT,
                    context={
                        "cross_type": "bearish",
                        "fast_ema": fast_ema,
                        "slow_ema": slow_ema,
                    },
                    confidence=0.9
                )
        
        self._prev_fast_above = fast_above
        return TriggerResult(trigger_id=self.trigger_id, triggered=False)
    
    def reset(self):
        """Reset state for new simulation."""
        self._prev_fast_above = None


class RSIThresholdTrigger(Trigger):
    """
    Trigger when RSI crosses threshold.
    
    Agent config examples:
        {"type": "rsi_threshold", "threshold": 30, "direction": "below"}
        {"type": "rsi_threshold", "threshold": 70, "direction": "above"}
        {"type": "rsi_threshold", "oversold": 30, "overbought": 70}  # Both
    """
    
    def __init__(
        self,
        threshold: Optional[float] = None,
        direction: Optional[str] = None,  # "above" or "below"
        oversold: float = 30.0,
        overbought: float = 70.0,
        timeframe: str = "5m",
        **kwargs  # Accept unknown args for robustness
    ):
        # Handle 'above'/'below' as direction aliases (agent may pass these)
        if 'above' in kwargs and kwargs['above']:
            direction = 'above'
            threshold = threshold or 70.0
        if 'below' in kwargs and kwargs['below']:
            direction = 'below'
            threshold = threshold or 30.0
        
        # Single threshold mode
        if threshold is not None and direction is not None:
            self._mode = "single"
            self._threshold = threshold
            self._direction = direction.lower()
            self._oversold = None
            self._overbought = None
        # Dual threshold mode
        else:
            self._mode = "dual"
            self._oversold = oversold
            self._overbought = overbought
            self._threshold = None
            self._direction = None
        
        self._timeframe = timeframe
        self._prev_rsi = None
    
    @property
    def trigger_id(self) -> str:
        if self._mode == "single":
            return f"rsi_{self._direction}_{int(self._threshold)}_{self._timeframe}"
        else:
            return f"rsi_{int(self._oversold)}_{int(self._overbought)}_{self._timeframe}"
    
    @property
    def params(self) -> Dict[str, Any]:
        if self._mode == "single":
            return {
                "threshold": self._threshold,
                "direction": self._direction,
                "timeframe": self._timeframe,
            }
        else:
            return {
                "oversold": self._oversold,
                "overbought": self._overbought,
                "timeframe": self._timeframe,
            }
    
    def check(self, features) -> TriggerResult:
        """Check RSI threshold conditions."""
        indicators = features.indicators
        if indicators is None:
            return TriggerResult(trigger_id=self.trigger_id, triggered=False)
        
        # Get RSI value
        rsi_key = f"rsi_{self._timeframe}_14"
        rsi = getattr(indicators, rsi_key, None)
        if rsi is None:
            rsi = getattr(indicators, 'rsi_5m_14', 50.0)
        
        triggered = False
        direction = TriggerDirection.NEUTRAL
        context = {"rsi": rsi}
        
        if self._mode == "single":
            if self._direction == "below" and rsi < self._threshold:
                if self._prev_rsi is None or self._prev_rsi >= self._threshold:
                    triggered = True
                    direction = TriggerDirection.LONG  # Oversold = bullish
                    context["condition"] = f"rsi_below_{self._threshold}"
            elif self._direction == "above" and rsi > self._threshold:
                if self._prev_rsi is None or self._prev_rsi <= self._threshold:
                    triggered = True
                    direction = TriggerDirection.SHORT  # Overbought = bearish
                    context["condition"] = f"rsi_above_{self._threshold}"
        else:
            # Dual threshold mode
            if rsi <= self._oversold:
                if self._prev_rsi is None or self._prev_rsi > self._oversold:
                    triggered = True
                    direction = TriggerDirection.LONG
                    context["condition"] = "oversold"
            elif rsi >= self._overbought:
                if self._prev_rsi is None or self._prev_rsi < self._overbought:
                    triggered = True
                    direction = TriggerDirection.SHORT
                    context["condition"] = "overbought"
        
        self._prev_rsi = rsi
        
        if triggered:
            return TriggerResult(
                trigger_id=self.trigger_id,
                triggered=True,
                direction=direction,
                context=context,
                confidence=0.85
            )
        
        return TriggerResult(trigger_id=self.trigger_id, triggered=False)
    
    def reset(self):
        """Reset state for new simulation."""
        self._prev_rsi = None

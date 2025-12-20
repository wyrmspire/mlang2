"""
Candle Pattern Trigger

Detects common candlestick patterns.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np

from .base import Trigger, TriggerResult, TriggerDirection


class CandlePattern(Enum):
    """Supported candle patterns."""
    HAMMER = "hammer"
    INVERTED_HAMMER = "inverted_hammer"
    BULLISH_ENGULFING = "bullish_engulfing"
    BEARISH_ENGULFING = "bearish_engulfing"
    DOJI = "doji"
    MORNING_STAR = "morning_star"  # Future
    EVENING_STAR = "evening_star"  # Future


def _body_size(o: float, c: float) -> float:
    """Absolute body size."""
    return abs(c - o)


def _upper_wick(o: float, h: float, c: float) -> float:
    """Upper wick length."""
    return h - max(o, c)


def _lower_wick(o: float, l: float, c: float) -> float:
    """Lower wick length."""
    return min(o, c) - l


def _total_range(h: float, l: float) -> float:
    """Total candle range."""
    return h - l


def _is_bullish(o: float, c: float) -> bool:
    """Is candle bullish (close > open)."""
    return c > o


class CandlePatternTrigger(Trigger):
    """
    Trigger that detects candlestick patterns.
    
    Agent config examples:
        {"type": "candle_pattern", "patterns": ["hammer"]}
        {"type": "candle_pattern", "patterns": ["bullish_engulfing", "bearish_engulfing"]}
        {"type": "candle_pattern", "patterns": ["doji"], "min_range_atr": 0.5}
    """
    
    def __init__(
        self,
        patterns: List[str],
        min_range_atr: float = 0.3,  # Min candle range as ATR multiple
        body_ratio_doji: float = 0.1,  # Max body/range for doji
        wick_ratio_hammer: float = 2.0,  # Min lower_wick/body for hammer
    ):
        self._patterns = [CandlePattern(p.lower()) for p in patterns]
        self._min_range_atr = min_range_atr
        self._body_ratio_doji = body_ratio_doji
        self._wick_ratio_hammer = wick_ratio_hammer
    
    @property
    def trigger_id(self) -> str:
        patterns_str = "_".join(p.value for p in self._patterns)
        return f"candle_{patterns_str}"
    
    @property
    def params(self) -> Dict[str, Any]:
        return {
            "patterns": [p.value for p in self._patterns],
            "min_range_atr": self._min_range_atr,
        }
    
    def check(self, features) -> TriggerResult:
        """Check for candle patterns in recent price data."""
        atr = features.atr if features.atr > 0 else 1.0
        
        # Try to get OHLCV candles - prefer window.raw_ohlcv_1m
        candles = None
        
        # First try: window.raw_ohlcv_1m (actual OHLCV data)
        if hasattr(features, 'window') and features.window is not None:
            raw = getattr(features.window, 'raw_ohlcv_1m', None)
            if raw is not None and len(raw) >= 2:
                candles = raw[-3:] if len(raw) >= 3 else raw[-2:]
        
        # Fallback: x_price_1m if it looks like OHLCV data
        if candles is None:
            prices = features.x_price_1m
            if prices is not None and len(prices) >= 2:
                # Check if it's OHLCV format (list of lists with 5 elements)
                if isinstance(prices[0], (list, tuple)) and len(prices[0]) >= 4:
                    candles = prices[-3:] if len(prices) >= 3 else prices[-2:]
        
        if candles is None or len(candles) < 2:
            return TriggerResult(trigger_id=self.trigger_id, triggered=False)
        
        # Current candle is last one
        curr = candles[-1]  # [o, h, l, c, v]
        o, h, l, c = curr[0], curr[1], curr[2], curr[3]
        
        # Previous candle
        prev = candles[-2] if len(candles) >= 2 else None
        
        total_range = _total_range(h, l)
        body = _body_size(o, c)
        upper = _upper_wick(o, h, c)
        lower = _lower_wick(o, l, c)
        
        # Check minimum range
        if total_range < self._min_range_atr * atr:
            return TriggerResult(trigger_id=self.trigger_id, triggered=False)
        
        # Check each pattern
        for pattern in self._patterns:
            detected = False
            direction = TriggerDirection.NEUTRAL
            
            if pattern == CandlePattern.HAMMER:
                # Hammer: long lower wick, small body at top
                if body > 0 and lower >= self._wick_ratio_hammer * body and upper < body:
                    detected = True
                    direction = TriggerDirection.LONG
                    
            elif pattern == CandlePattern.INVERTED_HAMMER:
                # Inverted hammer: long upper wick, small body at bottom
                if body > 0 and upper >= self._wick_ratio_hammer * body and lower < body:
                    detected = True
                    direction = TriggerDirection.SHORT
                    
            elif pattern == CandlePattern.DOJI:
                # Doji: tiny body relative to range
                if total_range > 0 and body / total_range < self._body_ratio_doji:
                    detected = True
                    direction = TriggerDirection.NEUTRAL
                    
            elif pattern == CandlePattern.BULLISH_ENGULFING and prev is not None:
                # Bullish engulfing: current body engulfs previous, current is bullish
                po, pc = prev[0], prev[3]
                if _is_bullish(o, c) and not _is_bullish(po, pc):
                    if c > po and o < pc:  # Current body engulfs previous
                        detected = True
                        direction = TriggerDirection.LONG
                        
            elif pattern == CandlePattern.BEARISH_ENGULFING and prev is not None:
                # Bearish engulfing: current body engulfs previous, current is bearish
                po, pc = prev[0], prev[3]
                if not _is_bullish(o, c) and _is_bullish(po, pc):
                    if o > pc and c < po:  # Current body engulfs previous
                        detected = True
                        direction = TriggerDirection.SHORT
            
            if detected:
                return TriggerResult(
                    trigger_id=self.trigger_id,
                    triggered=True,
                    direction=direction,
                    context={
                        "pattern": pattern.value,
                        "body": body,
                        "upper_wick": upper,
                        "lower_wick": lower,
                        "range": total_range,
                    },
                    confidence=0.8
                )
        
        return TriggerResult(trigger_id=self.trigger_id, triggered=False)

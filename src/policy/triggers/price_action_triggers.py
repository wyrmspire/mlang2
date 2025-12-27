"""
Price Action Triggers
Atomic triggers based on candlestick patterns and market structure.

Includes:
1. Rejection (Wick Rejection)
2. Pin Bar
3. Engulfing
4. Inside Bar
5. Double Top/Bottom
6. Flag Pattern
"""

from typing import Dict, Any, Optional
import numpy as np

from .base import Trigger, TriggerResult, TriggerDirection

class RejectionTrigger(Trigger):
    """
    Rejection Trigger (User Request).
    
    Logic:
    1. Identify range of previous M minutes (e.g., 30m).
    2. Check if current bar extended 1.5x of that range in one direction.
    3. Check if current bar CLOSED back inside or below/above the extension.
    
    Params:
        lookback (int): Number of bars for range calculation (e.g. 6 for 5m * 6 = 30m)
        extension_factor (float): 1.5
    """
    
    def __init__(self, lookback: int = 6, extension_factor: float = 1.5, timeframe: str = "5m"):
        self.lookback = lookback
        self.extension_factor = extension_factor
        self.timeframe = timeframe
        self.trigger_id_str = f"rejection_{timeframe}_{lookback}"

    @property
    def trigger_id(self) -> str:
        return self.trigger_id_str

    @property
    def params(self) -> Dict[str, Any]:
        return {
            "lookback": self.lookback,
            "extension_factor": self.extension_factor,
            "timeframe": self.timeframe
        }

    def check(self, features) -> TriggerResult:
        # Requires OHLC history. Assuming features.raw_candles exists or similar access.
        # Ideally, triggers should be stateless, but they need data access.
        # We assume 'features' object provides access to recent history.
        
        # NOTE: Using features.candles (deque or list of recent bars)
        candles = features.candles
        if len(candles) < self.lookback + 1:
             return TriggerResult(self.trigger_id, False)

        current_bar = candles[-1]
        history = list(candles)[-(self.lookback+1):-1] # Previous 'lookback' bars
        
        # Calculate recent range
        highs = [c.high for c in history]
        lows = [c.low for c in history]
        range_high = max(highs)
        range_low = min(lows)
        range_height = range_high - range_low
        
        if range_height == 0:
            return TriggerResult(self.trigger_id, False)

        # Bullish Rejection (Dip below range and close back up)
        # Extension target: Low - (Range * 0.5)? Or just price went below low?
        # User said: "1.5 of the previous 30m range" implies MOVE size.
        
        # Interpretation: 
        # Price moves DOWN by 1.5x the range height relative to range high? 
        # Or extended 50% beyond the range?
        # "1.5 in direction of move" -> Distance from High to Low of current move?
        
        # Common Rejection Logic:
        # Wick is long. 
        
        # Let's stick to a clear Rejection definition based on wicks relative to body.
        body = abs(current_bar.close - current_bar.open)
        wick_upper = current_bar.high - max(current_bar.open, current_bar.close)
        wick_lower = min(current_bar.open, current_bar.close) - current_bar.low
        total_len = current_bar.high - current_bar.low
        
        if total_len == 0:
             return TriggerResult(self.trigger_id, False)

        # Bullish Pin/Rejection: Long lower wick
        if wick_lower > (body * 2) and wick_lower > wick_upper:
             # Check if it swept a low?
             if current_bar.low < range_low:
                 return TriggerResult(self.trigger_id, True, TriggerDirection.LONG, {"type": "bullish_rejection", "support": range_low}, 0.8)

        # Bearish Pin/Rejection: Long upper wick
        if wick_upper > (body * 2) and wick_upper > wick_lower:
             if current_bar.high > range_high:
                 return TriggerResult(self.trigger_id, True, TriggerDirection.SHORT, {"type": "bearish_rejection", "resistance": range_high}, 0.8)

        return TriggerResult(self.trigger_id, False)


class PinBarTrigger(Trigger):
    """
    Classic Pin Bar Trigger.
    """
    def __init__(self, wick_ratio: float = 0.66):
        self.wick_ratio = wick_ratio
        
    @property
    def trigger_id(self) -> str: return "pin_bar"
    
    def check(self, features) -> TriggerResult:
        c = features.candles[-1]
        high, low, open_, close = c.high, c.low, c.open, c.close
        total_range = high - low
        if total_range == 0: return TriggerResult(self.trigger_id, False)
        
        upper_wick = high - max(open_, close)
        lower_wick = min(open_, close) - low
        
        # Bearish Pin
        if upper_wick / total_range >= self.wick_ratio:
            return TriggerResult(self.trigger_id, True, TriggerDirection.SHORT, {}, 0.7)
            
        # Bullish Pin
        if lower_wick / total_range >= self.wick_ratio:
            return TriggerResult(self.trigger_id, True, TriggerDirection.LONG, {}, 0.7)
            
        return TriggerResult(self.trigger_id, False)


class EngulfingTrigger(Trigger):
    """
    Engulfing Candle Trigger.
    """
    @property
    def trigger_id(self) -> str: return "engulfing"
    
    def check(self, features) -> TriggerResult:
        if len(features.candles) < 2: return TriggerResult(self.trigger_id, False)
        
        curr = features.candles[-1]
        prev = features.candles[-2]
        
        # Bullish Engulfing
        if (curr.close > curr.open) and (prev.close < prev.open): # Green after Red
            if (curr.close > prev.open) and (curr.open < prev.close): # Envelops body
                return TriggerResult(self.trigger_id, True, TriggerDirection.LONG, {}, 0.75)

        # Bearish Engulfing
        if (curr.close < curr.open) and (prev.close > prev.open): # Red after Green
            if (curr.close < prev.open) and (curr.open > prev.close):
                return TriggerResult(self.trigger_id, True, TriggerDirection.SHORT, {}, 0.75)
                
        return TriggerResult(self.trigger_id, False)


class InsideBarTrigger(Trigger):
    """
    Inside Bar Trigger.
    """
    @property
    def trigger_id(self) -> str: return "inside_bar"

    def check(self, features) -> TriggerResult:
        if len(features.candles) < 2: return TriggerResult(self.trigger_id, False)
        curr = features.candles[-1]
        prev = features.candles[-2]
        
        if (curr.high < prev.high) and (curr.low > prev.low):
            # Inside bar - usually implies potential breakout. 
            # Direction is neutral unless combined with trend.
            return TriggerResult(self.trigger_id, True, TriggerDirection.NEUTRAL, {}, 0.6)
            
        return TriggerResult(self.trigger_id, False)


class DoubleTopBottomTrigger(Trigger):
    """
    Simplified Double Top/Bottom detection.
    """
    @property
    def trigger_id(self) -> str: return "double_top_bottom"

    def check(self, features) -> TriggerResult:
        # Requires significant logic/zig-zag. Placeholder for now.
        return TriggerResult(self.trigger_id, False)


class FlagPatternTrigger(Trigger):
    """
    Bull/Bear Flag detection.
    """
    @property
    def trigger_id(self) -> str: return "flag_pattern"

    def check(self, features) -> TriggerResult:
         return TriggerResult(self.trigger_id, False)

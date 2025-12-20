"""
Structure Break Trigger
Triggers when price makes a new high/low then breaks the SECOND previous swing level.

Agent-configurable parameters:
- swing_lookback: Bars to confirm swing points (default 5)
- rr_ratio: Risk:Reward ratio for TP (default 2.0)
- atr_padding: ATR multiple for SL padding (default 0.5)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from src.policy.triggers.base import Trigger, TriggerResult, TriggerDirection
from src.features.pipeline import FeatureBundle


@dataclass
class SwingPoint:
    """A detected swing high or low."""
    price: float
    bar_idx: int
    is_high: bool


@dataclass 
class StructureBreakState:
    """Persistent state for tracking swings."""
    swing_highs: List[SwingPoint] = field(default_factory=list)
    swing_lows: List[SwingPoint] = field(default_factory=list)
    last_trigger_bar: int = -1000


class StructureBreakTrigger(Trigger):
    """
    Triggers on structure break pattern:
    
    SHORT Setup:
    1. Price makes new swing HIGH
    2. Then breaks the SECOND previous swing LOW (not immediate)
    3. Enter at candle close
    
    LONG Setup:
    1. Price makes new swing LOW  
    2. Then breaks the SECOND previous swing HIGH
    3. Enter at candle close
    
    Agent Config Example:
    {"type": "structure_break", "swing_lookback": 5, "rr_ratio": 2.0, "atr_padding": 0.5}
    """
    
    def __init__(
        self,
        swing_lookback: int = 5,
        rr_ratio: float = 2.0,
        atr_padding: float = 0.5,
        cooldown_bars: int = 3
    ):
        self.swing_lookback = swing_lookback
        self.rr_ratio = rr_ratio
        self.atr_padding = atr_padding
        self.cooldown_bars = cooldown_bars
        self._state = StructureBreakState()
    
    @property
    def trigger_id(self) -> str:
        return "structure_break"
    
    @property
    def params(self) -> Dict[str, Any]:
        return {
            "swing_lookback": self.swing_lookback,
            "rr_ratio": self.rr_ratio,
            "atr_padding": self.atr_padding,
            "cooldown_bars": self.cooldown_bars
        }
    
    def _update_swings(self, df_15m: pd.DataFrame, current_bar_idx: int):
        """Detect swing highs and lows in 15m data."""
        if df_15m is None or len(df_15m) < self.swing_lookback * 2 + 1:
            return
        
        recent = df_15m.tail(60).copy()
        if len(recent) < self.swing_lookback * 2 + 1:
            return
        
        highs = recent['high'].values
        lows = recent['low'].values
        
        new_highs = []
        new_lows = []
        
        for i in range(self.swing_lookback, len(recent) - self.swing_lookback):
            # Swing high: higher than neighbors
            is_sh = all(highs[i] > highs[i-j] and highs[i] > highs[i+j] 
                       for j in range(1, self.swing_lookback + 1))
            if is_sh:
                new_highs.append(SwingPoint(
                    price=highs[i],
                    bar_idx=current_bar_idx - (len(recent) - 1 - i),
                    is_high=True
                ))
            
            # Swing low: lower than neighbors
            is_sl = all(lows[i] < lows[i-j] and lows[i] < lows[i+j]
                       for j in range(1, self.swing_lookback + 1))
            if is_sl:
                new_lows.append(SwingPoint(
                    price=lows[i],
                    bar_idx=current_bar_idx - (len(recent) - 1 - i),
                    is_high=False
                ))
        
        self._state.swing_highs = new_highs[-10:]
        self._state.swing_lows = new_lows[-10:]
    
    def check(self, features: FeatureBundle, df_15m: pd.DataFrame = None) -> TriggerResult:
        """Check for structure break pattern."""
        
        # Cooldown
        if features.bar_idx - self._state.last_trigger_bar < self.cooldown_bars:
            return TriggerResult(trigger_id=self.trigger_id, triggered=False)
        
        if df_15m is None or df_15m.empty:
            return TriggerResult(trigger_id=self.trigger_id, triggered=False)
        
        self._update_swings(df_15m, features.bar_idx)
        
        if len(self._state.swing_highs) < 2 or len(self._state.swing_lows) < 2:
            return TriggerResult(trigger_id=self.trigger_id, triggered=False)
        
        current_bar = df_15m.iloc[-1]
        close = current_bar['close']
        high = current_bar['high']
        low = current_bar['low']
        atr = features.atr if features.atr > 0 else 5.0
        
        sorted_highs = sorted(self._state.swing_highs, key=lambda x: x.bar_idx, reverse=True)
        sorted_lows = sorted(self._state.swing_lows, key=lambda x: x.bar_idx, reverse=True)
        
        # SHORT: Recent swing high exists + break second prev low
        # (removed is_new_high requirement - we just need a swing high followed by structure break)
        if len(sorted_highs) >= 1 and len(sorted_lows) >= 2:
            newest_high = sorted_highs[0]
            # Second previous low (skip the most recent)
            second_low = sorted_lows[1] if len(sorted_lows) > 1 else sorted_lows[0]
            
            # Only trigger if the swing high is more recent than the second low
            if newest_high.bar_idx > second_low.bar_idx and close < second_low.price:
                stop = high + (self.atr_padding * atr)
                risk = stop - close
                if risk > 0:
                    tp = close - (self.rr_ratio * risk)
                    self._state.last_trigger_bar = features.bar_idx
                    
                    return TriggerResult(
                        trigger_id=self.trigger_id,
                        triggered=True,
                        direction=TriggerDirection.SHORT,
                        context={
                            'entry_price': close,
                            'stop_price': stop,
                            'tp_price': tp,
                            'broken_level': second_low.price,
                            'new_extreme': newest_high.price,
                            'trigger_high': high,
                            'trigger_low': low,
                            'risk_points': risk,
                            'rr_ratio': self.rr_ratio
                        }
                    )
        
        # LONG: Recent swing low exists + break second prev high
        # (removed is_new_low requirement - we just need a swing low followed by structure break)
        if len(sorted_lows) >= 1 and len(sorted_highs) >= 2:
            newest_low = sorted_lows[0]
            # Second previous high (skip the most recent)
            second_high = sorted_highs[1] if len(sorted_highs) > 1 else sorted_highs[0]
            
            # Only trigger if the swing low is more recent than the second high
            if newest_low.bar_idx > second_high.bar_idx and close > second_high.price:
                stop = low - (self.atr_padding * atr)
                risk = close - stop
                if risk > 0:
                    tp = close + (self.rr_ratio * risk)
                    self._state.last_trigger_bar = features.bar_idx
                    
                    return TriggerResult(
                        trigger_id=self.trigger_id,
                        triggered=True,
                        direction=TriggerDirection.LONG,
                        context={
                            'entry_price': close,
                            'stop_price': stop,
                            'tp_price': tp,
                            'broken_level': second_high.price,
                            'new_extreme': newest_low.price,
                            'trigger_high': high,
                            'trigger_low': low,
                            'risk_points': risk,
                            'rr_ratio': self.rr_ratio
                        }
                    )
        
        return TriggerResult(trigger_id=self.trigger_id, triggered=False)

"""
Unit tests for Trigger and Bracket components.
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from dataclasses import dataclass
from typing import Optional
import pandas as pd

from src.policy.triggers import (
    trigger_from_dict, list_triggers,
    TimeTrigger, CandlePatternTrigger, EMACrossTrigger, RSIThresholdTrigger,
    TriggerDirection
)
from src.policy.brackets import (
    bracket_from_dict, list_brackets,
    ATRBracket, PercentBracket, FixedBracket
)


# ===========================================================================
# Mock FeatureBundle for testing
# ===========================================================================

@dataclass
class MockIndicators:
    ema_5m_20: float = 4500.0
    ema_5m_200: float = 4480.0
    rsi_5m_14: float = 50.0


@dataclass
class MockWindow:
    raw_ohlcv_1m: list = None
    
    def __post_init__(self):
        if self.raw_ohlcv_1m is None:
            # Default: normal candles
            self.raw_ohlcv_1m = [
                [4500, 4510, 4495, 4505, 1000],  # Normal candle
                [4505, 4515, 4500, 4510, 1200],  # Normal candle
            ]


@dataclass
class MockFeatures:
    timestamp: Optional[pd.Timestamp] = None
    current_price: float = 4500.0
    atr: float = 5.0
    indicators: MockIndicators = None
    window: MockWindow = None
    x_price_1m: list = None
    bar_idx: int = 100
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = pd.Timestamp("2025-03-17 10:00:00", tz="America/New_York")
        if self.indicators is None:
            self.indicators = MockIndicators()
        if self.window is None:
            self.window = MockWindow()


# ===========================================================================
# Trigger Tests
# ===========================================================================

class TestTimeTrigger:
    def test_fires_at_correct_time(self):
        trigger = TimeTrigger(hour=10, minute=0)
        
        # Should fire at 10:00
        features = MockFeatures(
            timestamp=pd.Timestamp("2025-03-17 10:00:00", tz="America/New_York")
        )
        result = trigger.check(features)
        assert result.triggered is True
        assert result.context["hour"] == 10
        
    def test_does_not_fire_at_wrong_time(self):
        trigger = TimeTrigger(hour=10, minute=0)
        
        # Should NOT fire at 9:30
        features = MockFeatures(
            timestamp=pd.Timestamp("2025-03-17 09:30:00", tz="America/New_York")
        )
        result = trigger.check(features)
        assert result.triggered is False
        
    def test_fires_only_once_per_day(self):
        trigger = TimeTrigger(hour=10, minute=0)
        features = MockFeatures(
            timestamp=pd.Timestamp("2025-03-17 10:00:00", tz="America/New_York")
        )
        
        # First call: fires
        result1 = trigger.check(features)
        assert result1.triggered is True
        
        # Second call same time: should NOT fire again
        result2 = trigger.check(features)
        assert result2.triggered is False

    def test_factory_function(self):
        trigger = trigger_from_dict({"type": "time", "hour": 14, "minute": 30})
        assert isinstance(trigger, TimeTrigger)
        assert trigger._hours == [14]
        assert trigger._minute == 30


class TestCandlePatternTrigger:
    def test_detects_hammer(self):
        trigger = CandlePatternTrigger(patterns=["hammer"])
        
        # Create hammer candle: long lower wick, small body at top
        # Body must be small, lower wick >= 2x body, upper wick < body
        # o=4503, h=4506, l=4490, c=4506 -> body=3, lower=13, upper=0
        hammer_candle = [4503, 4506, 4490, 4506, 1000]
        candles = [[4490, 4495, 4485, 4492, 800], hammer_candle]
        
        features = MockFeatures(
            window=MockWindow(raw_ohlcv_1m=candles),
            atr=5.0
        )
        
        result = trigger.check(features)
        assert result.triggered is True
        assert result.context["pattern"] == "hammer"
        assert result.direction == TriggerDirection.LONG
        
    def test_detects_doji(self):
        trigger = CandlePatternTrigger(patterns=["doji"])
        
        # Create doji: tiny body relative to range (body/range < 0.1)
        # o=4500, h=4520, l=4480, c=4501 -> body=1, range=40, ratio=0.025
        doji_candle = [4500, 4520, 4480, 4501, 1000]
        candles = [[4490, 4495, 4485, 4492, 800], doji_candle]
        
        features = MockFeatures(
            window=MockWindow(raw_ohlcv_1m=candles),
            atr=5.0
        )
        
        result = trigger.check(features)
        assert result.triggered is True
        assert result.context["pattern"] == "doji"

    def test_factory_function(self):
        trigger = trigger_from_dict({"type": "candle_pattern", "patterns": ["hammer", "doji"]})
        assert isinstance(trigger, CandlePatternTrigger)


class TestRSIThresholdTrigger:
    def test_detects_oversold(self):
        trigger = RSIThresholdTrigger(oversold=30, overbought=70)
        
        features = MockFeatures()
        features.indicators = MockIndicators(rsi_5m_14=25.0)
        
        result = trigger.check(features)
        assert result.triggered is True
        assert result.direction == TriggerDirection.LONG
        assert result.context["condition"] == "oversold"
        
    def test_detects_overbought(self):
        trigger = RSIThresholdTrigger(oversold=30, overbought=70)
        trigger._prev_rsi = 65  # Set prev to avoid first-check issues
        
        features = MockFeatures()
        features.indicators = MockIndicators(rsi_5m_14=75.0)
        
        result = trigger.check(features)
        assert result.triggered is True
        assert result.direction == TriggerDirection.SHORT
        
    def test_factory_function(self):
        trigger = trigger_from_dict({"type": "rsi_threshold", "threshold": 30, "direction": "below"})
        assert isinstance(trigger, RSIThresholdTrigger)


class TestEMACrossTrigger:
    def test_detects_bullish_cross(self):
        trigger = EMACrossTrigger(fast=9, slow=21)
        
        # Set previous state: fast below slow
        trigger._prev_fast_above = False
        
        # Now fast > slow
        features = MockFeatures()
        features.indicators = MockIndicators(ema_5m_20=4510.0, ema_5m_200=4500.0)
        
        result = trigger.check(features)
        assert result.triggered is True
        assert result.direction == TriggerDirection.LONG
        assert result.context["cross_type"] == "bullish"

    def test_factory_function(self):
        trigger = trigger_from_dict({"type": "ema_cross", "fast": 9, "slow": 21})
        assert isinstance(trigger, EMACrossTrigger)


# ===========================================================================
# Bracket Tests
# ===========================================================================

class TestATRBracket:
    def test_long_bracket(self):
        bracket = ATRBracket(stop_atr=2.0, tp_atr=3.0)
        
        levels = bracket.compute(
            entry_price=4500.0,
            direction="LONG",
            atr=5.0
        )
        
        assert levels.stop_price == 4490.0  # 4500 - 2*5
        assert levels.tp_price == 4515.0    # 4500 + 3*5
        assert levels.risk_points == 10.0
        assert levels.reward_points == 15.0
        assert levels.r_multiple == 1.5
        
    def test_short_bracket(self):
        bracket = ATRBracket(stop_atr=2.0, tp_atr=3.0)
        
        levels = bracket.compute(
            entry_price=4500.0,
            direction="SHORT",
            atr=5.0
        )
        
        assert levels.stop_price == 4510.0  # 4500 + 2*5
        assert levels.tp_price == 4485.0    # 4500 - 3*5

    def test_factory_function(self):
        bracket = bracket_from_dict({"type": "atr", "stop_atr": 1.5, "tp_atr": 2.5})
        assert isinstance(bracket, ATRBracket)


class TestPercentBracket:
    def test_percent_calculation(self):
        bracket = PercentBracket(stop_pct=0.5, tp_pct=1.0)
        
        levels = bracket.compute(
            entry_price=4500.0,
            direction="LONG",
            atr=5.0
        )
        
        # 0.5% of 4500 = 22.5, 1.0% = 45
        assert levels.stop_price == 4500.0 - 22.5
        assert levels.tp_price == 4500.0 + 45.0


class TestFixedBracket:
    def test_fixed_points(self):
        bracket = FixedBracket(stop_points=10.0, tp_points=20.0)
        
        levels = bracket.compute(
            entry_price=4500.0,
            direction="LONG",
            atr=5.0  # Ignored for fixed
        )
        
        assert levels.stop_price == 4490.0
        assert levels.tp_price == 4520.0


# ===========================================================================
# Registry Tests
# ===========================================================================

def test_list_triggers():
    triggers = list_triggers()
    assert "time" in triggers
    assert "candle_pattern" in triggers
    assert "ema_cross" in triggers
    assert "rsi_threshold" in triggers


def test_list_brackets():
    brackets = list_brackets()
    assert "atr" in brackets
    assert "percent" in brackets
    assert "fixed" in brackets


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

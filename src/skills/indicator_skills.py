"""
Indicator Skills
Atomic tools for calculating technical indicators.
These skills wrap the core features library for the Agent's use during Research.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime

from src.core.tool_registry import ToolRegistry, ToolCategory
from src.features import indicators
from src.features import indicators_pro
from src.features import fvg


@ToolRegistry.register(
    tool_id="get_rsi",
    category=ToolCategory.INDICATOR,
    name="Get RSI",
    description="Calculate RSI (Relative Strength Index) for a list of prices",
    input_schema={
        "type": "object",
        "properties": {
            "prices": {
                "type": "array",
                "items": {"type": "number"},
                "description": "List of price values"
            },
            "period": {
                "type": "integer",
                "description": "RSI period (default: 14)",
                "default": 14
            }
        },
        "required": ["prices"]
    },
    output_schema={
        "type": "object",
        "properties": {
            "rsi_values": {
                "type": "array",
                "items": {"type": "number"}
            }
        }
    }
)
class GetRSITool:
    def execute(self, prices: List[float], period: int = 14, **kwargs) -> Dict[str, Any]:
        """Calculate RSI for a list of prices."""
        series = pd.Series(prices)
        rsi = indicators.calculate_rsi(series, period)
        return {"rsi_values": rsi.tolist()}


@ToolRegistry.register(
    tool_id="check_ema_cross",
    category=ToolCategory.INDICATOR,
    name="Check EMA Cross",
    description="Check if fast EMA crossed slow EMA on the most recent bar",
    input_schema={
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Symbol to check (default: MES)",
                "default": "continuous"
            },
            "fast": {
                "type": "integer",
                "description": "Fast EMA period",
                "default": 9
            },
            "slow": {
                "type": "integer",
                "description": "Slow EMA period",
                "default": 21
            },
            "lookback_bars": {
                "type": "integer",
                "description": "Number of bars to analyze",
                "default": 100
            }
        }
    },
    output_schema={
        "type": "object",
        "properties": {
            "cross_type": {
                "type": "string",
                "enum": ["BULLISH", "BEARISH", "NONE"]
            },
            "fast_value": {"type": "number"},
            "slow_value": {"type": "number"}
        }
    }
)
class CheckEMACrossTool:
    def execute(self, symbol: str = "continuous", fast: int = 9, slow: int = 21, lookback_bars: int = 100, **kwargs) -> Dict[str, Any]:
        """Check if EMA cross occurred."""
        from src.data.loader import load_continuous_contract
        
        df = load_continuous_contract()
        if len(df) > lookback_bars:
            df = df.tail(lookback_bars)
        
        ema_fast = indicators.calculate_ema(df['close'], fast)
        ema_slow = indicators.calculate_ema(df['close'], slow)
        
        if len(df) < 2:
            return {"cross_type": "NONE", "fast_value": 0.0, "slow_value": 0.0}
            
        curr_fast = float(ema_fast.iloc[-1])
        curr_slow = float(ema_slow.iloc[-1])
        prev_fast = float(ema_fast.iloc[-2])
        prev_slow = float(ema_slow.iloc[-2])
        
        cross_type = "NONE"
        if prev_fast <= prev_slow and curr_fast > curr_slow:
            cross_type = "BULLISH"
        elif prev_fast >= prev_slow and curr_fast < curr_slow:
            cross_type = "BEARISH"
            
        return {
            "cross_type": cross_type,
            "fast_value": curr_fast,
            "slow_value": curr_slow
        }


@ToolRegistry.register(
    tool_id="get_current_rsi",
    category=ToolCategory.INDICATOR,
    name="Get Current RSI",
    description="Get the current RSI value for a symbol",
    input_schema={
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Symbol (default: continuous)",
                "default": "continuous"
            },
            "period": {
                "type": "integer",
                "description": "RSI period",
                "default": 14
            },
            "lookback_bars": {
                "type": "integer",
                "description": "Number of bars to use for calculation",
                "default": 50
            }
        }
    },
    output_schema={
        "type": "object",
        "properties": {
            "rsi": {"type": "number"},
            "timestamp": {"type": "string"}
        }
    }
)
class GetCurrentRSITool:
    def execute(self, symbol: str = "continuous", period: int = 14, lookback_bars: int = 50, **kwargs) -> Dict[str, Any]:
        """Get current RSI value."""
        from src.data.loader import load_continuous_contract
        
        df = load_continuous_contract()
        if len(df) > lookback_bars:
            df = df.tail(lookback_bars)
        
        if len(df) < period + 1:
            return {"rsi": 50.0, "timestamp": ""}
            
        rsi_series = indicators.calculate_rsi(df['close'], period)
        current_rsi = float(rsi_series.iloc[-1])
        timestamp = str(df['time'].iloc[-1])
        
        return {
            "rsi": current_rsi,
            "timestamp": timestamp
        }

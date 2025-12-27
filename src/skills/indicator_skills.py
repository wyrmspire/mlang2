"""
Indicator Skills
Atomic tools for calculating technical indicators.
These skills wrap the core features library for the Agent's use during Research.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
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
@ToolRegistry.register(
    tool_id="get_atr",
    category=ToolCategory.INDICATOR,
    name="Get ATR",
    description="Calculate Average True Range (volatility) for a symbol",
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
                "description": "ATR period (default: 14)",
                "default": 14
            },
            "lookback_bars": {
                "type": "integer",
                "description": "Number of bars to use (default: 100)",
                "default": 100
            }
        }
    },
    output_schema={
        "type": "object",
        "properties": {
            "atr_values": {
                "type": "array",
                "items": {"type": "number"}
            },
            "current_atr": {"type": "number"}
        }
    }
)
class GetATRTool:
    def execute(self, symbol: str = "continuous", period: int = 14, lookback_bars: int = 100, **kwargs) -> Dict[str, Any]:
        """Calculate ATR."""
        from src.data.loader import load_continuous_contract
        df = load_continuous_contract()
        if len(df) > lookback_bars + period:
            df = df.tail(lookback_bars + period)
        
        atr_series = indicators.calculate_atr(df, period)
        # Shifted back by 1 because calculate_atr usually shifts to be causal, 
        # but for a point-in-time tool we might want the last calculated value
        atr_values = atr_series.tail(lookback_bars).dropna().tolist()
        current_atr = atr_values[-1] if atr_values else 0.0
        
        return {
            "atr_values": atr_values,
            "current_atr": current_atr
        }


@ToolRegistry.register(
    tool_id="get_vwap",
    category=ToolCategory.INDICATOR,
    name="Get VWAP",
    description="Calculate Volume Weighted Average Price for a symbol",
    input_schema={
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Symbol (default: continuous)",
                "default": "continuous"
            },
            "period": {
                "type": "string",
                "enum": ["session", "daily", "weekly"],
                "description": "VWAP anchor period",
                "default": "session"
            },
            "lookback_bars": {
                "type": "integer",
                "description": "Number of bars to return",
                "default": 50
            }
        }
    },
    output_schema={
        "type": "object",
        "properties": {
            "vwap_values": {
                "type": "array",
                "items": {"type": "number"}
            },
            "current_vwap": {"type": "number"}
        }
    }
)
class GetVWAPTool:
    def execute(self, symbol: str = "continuous", period: str = "session", lookback_bars: int = 50, **kwargs) -> Dict[str, Any]:
        """Calculate VWAP."""
        from src.data.loader import load_continuous_contract
        df = load_continuous_contract()
        # We need enough data for the session/period
        df = df.tail(max(500, lookback_bars))
        
        vwap_series = indicators.calculate_vwap(df, period=period)
        vwap_values = vwap_series.tail(lookback_bars).tolist()
        current_vwap = vwap_values[-1] if vwap_values else 0.0
        
        return {
            "vwap_values": vwap_values,
            "current_vwap": current_vwap
        }


@ToolRegistry.register(
    tool_id="detect_support_resistance",
    category=ToolCategory.INDICATOR,
    name="Detect Support & Resistance",
    description="Identify major support and resistance levels based on price clustering",
    input_schema={
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Symbol (default: continuous)",
                "default": "continuous"
            },
            "lookback_bars": {
                "type": "integer",
                "description": "Number of bars to analyze (default: 500)",
                "default": 500
            },
            "sensitivity": {
                "type": "number",
                "description": "Cluster sensitivity (default: 1.0)",
                "default": 1.0
            }
        }
    },
    output_schema={
        "type": "object",
        "properties": {
            "levels": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "price": {"type": "number"},
                        "strength": {"type": "number"},
                        "type": {"type": "string", "enum": ["SUPPORT", "RESISTANCE", "ZONE"]}
                    }
                }
            }
        }
    }
)
class DetectSupportResistanceTool:
    def execute(self, symbol: str = "continuous", lookback_bars: int = 500, sensitivity: float = 1.0, **kwargs) -> Dict[str, Any]:
        """Detect S&R levels."""
        from src.data.loader import load_continuous_contract
        df = load_continuous_contract().tail(lookback_bars)
        
        # Simple clustering: histogram of highs and lows
        prices = pd.concat([df['high'], df['low']])
        # Bin size ~ 0.5 points (typical for ES/MES)
        bins = int((prices.max() - prices.min()) / (0.5 * sensitivity))
        if bins < 5: bins = 5
        if bins > 100: bins = 100
        
        hist, edges = np.histogram(prices, bins=bins)
        
        # Find peaks in histogram
        levels = []
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > (lookback_bars * 0.05):
                price = float((edges[i] + edges[i+1]) / 2)
                # Determine if it's above or below current price
                current_price = df['close'].iloc[-1]
                ltype = "RESISTANCE" if price > current_price else "SUPPORT"
                
                levels.append({
                    "price": round(price, 2),
                    "strength": int(hist[i]),
                    "type": ltype
                })
        
        # Sort by strength
        levels.sort(key=lambda x: x['strength'], reverse=True)
        
        return {"levels": levels[:10]}


@ToolRegistry.register(
    tool_id="get_volume_profile",
    category=ToolCategory.INDICATOR,
    name="Get Volume Profile",
    description="Calculate volume-at-price profile over a period",
    input_schema={
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Symbol (default: continuous)",
                "default": "continuous"
            },
            "lookback_bars": {
                "type": "integer",
                "description": "Number of bars to analyze (default: 500)",
                "default": 500
            },
            "bins": {
                "type": "integer",
                "description": "Number of price bins (default: 50)",
                "default": 50
            }
        }
    },
    output_schema={
        "type": "object",
        "properties": {
            "bins": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "price": {"type": "number"},
                        "volume": {"type": "number"},
                        "is_poc": {"type": "boolean"}
                    }
                }
            },
            "poc_price": {"type": "number"}
        }
    }
)
class GetVolumeProfileTool:
    def execute(self, symbol: str = "continuous", lookback_bars: int = 500, bins: int = 50, **kwargs) -> Dict[str, Any]:
        """Calculate volume profile."""
        from src.data.loader import load_continuous_contract
        df = load_continuous_contract().tail(lookback_bars)
        
        # Determine price ranges and bin width
        p_min = df['low'].min()
        p_max = df['high'].max()
        if p_max == p_min:
            return {"bins": [], "poc_price": p_min}
            
        bin_width = (p_max - p_min) / bins
        
        # Accumulate volume for each bin
        # We simplify by using the close price or distributing between high/low
        # For a simple tool, we'll use close price
        # More advanced would use OHLC interpolation
        hist, edges = np.histogram(df['close'], bins=bins, weights=df['volume'])
        
        poc_idx = np.argmax(hist)
        poc_price = float((edges[poc_idx] + edges[poc_idx+1]) / 2)
        
        result_bins = []
        for i in range(len(hist)):
            price = float((edges[i] + edges[i+1]) / 2)
            result_bins.append({
                "price": round(price, 2),
                "volume": int(hist[i]),
                "is_poc": i == poc_idx
            })
            
        return {
            "bins": result_bins,
            "poc_price": round(poc_price, 2)
        }



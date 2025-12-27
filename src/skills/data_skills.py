"""
Data Skills
Atomic tools for fetching and inspecting market data.
Wraps src.data.loader for the Agent.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta

from src.core.tool_registry import ToolRegistry, ToolCategory
from src.data import loader
from src.config import NY_TZ


@ToolRegistry.register(
    tool_id="fetch_ohlcv",
    category=ToolCategory.UTILITY,
    name="Fetch OHLCV Data",
    description="Fetch OHLCV (candlestick) data for analysis",
    input_schema={
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Symbol to fetch (default: continuous)",
                "default": "continuous"
            },
            "start_date": {
                "type": "string",
                "description": "Start date (YYYY-MM-DD), optional"
            },
            "end_date": {
                "type": "string",
                "description": "End date (YYYY-MM-DD), optional"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of bars (default: 1000)",
                "default": 1000
            }
        }
    },
    output_schema={
        "type": "object",
        "properties": {
            "bars": {
                "type": "array",
                "items": {"type": "object"}
            },
            "count": {"type": "integer"}
        }
    }
)
class FetchOHLCVTool:
    def execute(self, symbol: str = "continuous", start_date: str = None, end_date: str = None, limit: int = 1000, **kwargs) -> Dict[str, Any]:
        """Fetch OHLCV data for analysis."""
        df = loader.load_continuous_contract()
        
        # Filter by date
        if start_date:
            start_dt = pd.to_datetime(start_date).tz_localize(NY_TZ)
            df = df[df['time'] >= start_dt]
            
        if end_date:
            end_dt = pd.to_datetime(end_date).tz_localize(NY_TZ) + timedelta(days=1)
            df = df[df['time'] < end_dt]
            
        # Limit rows
        if limit and len(df) > limit:
            df = df.iloc[:limit]
            
        # Convert to list of dicts
        records = df.to_dict('records')
        
        # Format timestamps
        for r in records:
            if isinstance(r['time'], pd.Timestamp):
                r['time'] = r['time'].isoformat()
                
        return {
            "bars": records,
            "count": len(records)
        }


@ToolRegistry.register(
    tool_id="get_current_price",
    category=ToolCategory.UTILITY,
    name="Get Current Price",
    description="Get the latest close price for a symbol",
    input_schema={
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Symbol (default: continuous)",
                "default": "continuous"
            }
        }
    },
    output_schema={
        "type": "object",
        "properties": {
            "price": {"type": "number"},
            "timestamp": {"type": "string"}
        }
    }
)
class GetCurrentPriceTool:
    def execute(self, symbol: str = "continuous", **kwargs) -> Dict[str, Any]:
        """Get the latest close price."""
        df = loader.load_continuous_contract()
        if df.empty:
            return {"price": 0.0, "timestamp": ""}
        
        price = float(df['close'].iloc[-1])
        timestamp = str(df['time'].iloc[-1])
        
        return {
            "price": price,
            "timestamp": timestamp
        }


@ToolRegistry.register(
    tool_id="get_market_regime",
    category=ToolCategory.UTILITY,
    name="Get Market Regime",
    description="Determine the market regime over the last N days (TRENDING_UP, TRENDING_DOWN, or RANGING)",
    input_schema={
        "type": "object",
        "properties": {
            "window_days": {
                "type": "integer",
                "description": "Number of days to analyze (default: 5)",
                "default": 5
            }
        }
    },
    output_schema={
        "type": "object",
        "properties": {
            "regime": {
                "type": "string",
                "enum": ["TRENDING_UP", "TRENDING_DOWN", "RANGING", "UNKNOWN"]
            },
            "return_pct": {"type": "number"}
        }
    }
)
class GetMarketRegimeTool:
    def execute(self, window_days: int = 5, **kwargs) -> Dict[str, Any]:
        """Determine market regime over last N days."""
        df = loader.load_continuous_contract()
        if df.empty:
            return {"regime": "UNKNOWN", "return_pct": 0.0}
            
        # Filter last N days
        cutoff = df['time'].iloc[-1] - timedelta(days=window_days)
        recent = df[df['time'] >= cutoff]
        
        if recent.empty:
            return {"regime": "UNKNOWN", "return_pct": 0.0}
            
        start_price = recent['close'].iloc[0]
        end_price = recent['close'].iloc[-1]
        ret = (end_price - start_price) / start_price
        
        regime = "RANGING"
        if ret > 0.02:  # > 2% up
            regime = "TRENDING_UP"
        elif ret < -0.02:  # > 2% down
            regime = "TRENDING_DOWN"
            
        return {
            "regime": regime,
            "return_pct": float(ret * 100)  # Convert to percentage
        }

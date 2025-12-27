"""
Data Skills
Atomic tools for fetching and inspecting market data.
Wraps src.data.loader for the Agent.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
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
    tool_id="get_dataset_last_price",
    category=ToolCategory.UTILITY,
    name="Get Dataset Last Price",
    description="Get the last close price in the historical dataset (end of Sept 2025). This is NOT live market data.",
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
            "timestamp": {"type": "string"},
            "note": {"type": "string"}
        }
    }
)
class GetDatasetLastPriceTool:
    def execute(self, symbol: str = "continuous", **kwargs) -> Dict[str, Any]:
        """Get the last price from the historical dataset."""
        df = loader.load_continuous_contract()
        if df.empty:
            return {"price": 0.0, "timestamp": "", "note": "Dataset is empty"}
        
        price = float(df['close'].iloc[-1])
        timestamp = str(df['time'].iloc[-1])
        
        return {
            "price": price,
            "timestamp": timestamp,
            "note": "This is the END of the historical dataset, not live market data"
        }


@ToolRegistry.register(
    tool_id="get_dataset_summary",
    category=ToolCategory.UTILITY,
    name="Get Dataset Summary",
    description="Get summary statistics about the historical dataset (March-Sept 2025): date range, total bars, volatility metrics",
    input_schema={
        "type": "object",
        "properties": {}
    },
    output_schema={
        "type": "object",
        "properties": {
            "start_date": {"type": "string"},
            "end_date": {"type": "string"},
            "total_bars": {"type": "integer"},
            "avg_daily_range": {"type": "number"},
            "avg_volume": {"type": "number"},
            "period_description": {"type": "string"}
        }
    }
)
class GetDatasetSummaryTool:
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Get summary statistics about the historical dataset."""
        df = loader.load_continuous_contract()
        if df.empty:
            return {
                "start_date": "",
                "end_date": "",
                "total_bars": 0,
                "avg_daily_range": 0.0,
                "avg_volume": 0.0,
                "period_description": "No data available"
            }
        
        start_date = str(df['time'].iloc[0].date())
        end_date = str(df['time'].iloc[-1].date())
        total_bars = len(df)
        
        # Calculate average daily range (high - low)
        daily_range = (df['high'] - df['low']).mean()
        avg_volume = df['volume'].mean()
        
        return {
            "start_date": start_date,
            "end_date": end_date,
            "total_bars": total_bars,
            "avg_daily_range": float(daily_range),
            "avg_volume": float(avg_volume),
            "period_description": f"Historical MES data from {start_date} to {end_date} ({total_bars:,} 1-minute bars)"
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
@ToolRegistry.register(
    tool_id="get_time_of_day_stats",
    category=ToolCategory.UTILITY,
    name="Get Time-of-Day Stats",
    description="Analyze volatility and price action average by hour of the day",
    input_schema={
        "type": "object",
        "properties": {
            "lookback_days": {
                "type": "integer",
                "description": "Number of days to analyze (default: 30)",
                "default": 30
            }
        }
    },
    output_schema={
        "type": "object",
        "properties": {
            "hourly_stats": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "hour": {"type": "integer"},
                        "avg_range": {"type": "number"},
                        "avg_volume": {"type": "number"},
                        "volatility": {"type": "number"}
                    }
                }
            }
        }
    }
)
class GetTimeOfDayStatsTool:
    def execute(self, lookback_days: int = 30, **kwargs) -> Dict[str, Any]:
        """Analyze average stats by hour."""
        from src.data.loader import load_continuous_contract
        df = load_continuous_contract()
        
        # Filter last N days
        cutoff = df['time'].iloc[-1] - timedelta(days=lookback_days)
        df = df[df['time'] >= cutoff].copy()
        
        df['hour'] = df['time'].dt.hour
        df['range'] = df['high'] - df['low']
        
        stats = df.groupby('hour').agg({
            'range': 'mean',
            'volume': 'mean',
            'close': 'std' # Simple volatility proxy
        }).reset_index()
        
        hourly_stats = []
        for _, row in stats.iterrows():
            hourly_stats.append({
                "hour": int(row['hour']),
                "avg_range": round(float(row['range']), 2),
                "avg_volume": round(float(row['volume']), 0),
                "volatility": round(float(row['close']), 2)
            })
            
        return {"hourly_stats": hourly_stats}

"""
Data Skills
Atomic tools for fetching and inspecting market data.
Wraps src.data.loader for the Agent.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta

from src.data import loader
from src.config import NY_TZ


def fetch_ohlcv(
    symbol: str = "continuous",
    start_date: str = None,
    end_date: str = None,
    limit: int = 1000
) -> List[Dict[str, Any]]:
    """
    Fetch OHLCV data for analysis.
    Returns list of dictionaries: {time, open, high, low, close, volume}
    """
    # Load full dataset (cached)
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
        
    # Convert to list of dicts for Agent
    records = df.to_dict('records')
    
    # Format timestamps to strings for JSON serializability
    for r in records:
        if isinstance(r['time'], pd.Timestamp):
            r['time'] = r['time'].isoformat()
            
    return records


def get_current_price(symbol: str = "continuous") -> float:
    """Get the latest close price."""
    df = loader.load_continuous_contract()
    if df.empty:
        return 0.0
    return float(df['close'].iloc[-1])


def get_market_regime(
    window_days: int = 5
) -> str:
    """
    Determine simplistic market regime over last N days.
    Returns: "TRENDING_UP", "TRENDING_DOWN", "RANGING"
    """
    df = loader.load_continuous_contract()
    if df.empty:
        return "UNKNOWN"
        
    # Filter last N days
    cutoff = df['time'].iloc[-1] - timedelta(days=window_days)
    recent = df[df['time'] >= cutoff]
    
    if recent.empty:
        return "UNKNOWN"
        
    start_price = recent['close'].iloc[0]
    end_price = recent['close'].iloc[-1]
    ret = (end_price - start_price) / start_price
    
    if ret > 0.02:  # > 2% up
        return "TRENDING_UP"
    elif ret < -0.02: # > 2% down
        return "TRENDING_DOWN"
    else:
        return "RANGING"

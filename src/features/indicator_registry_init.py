"""
Indicator Registration  
Wire existing indicators into the IndicatorRegistry.
"""

from src.core.registries import IndicatorRegistry, IndicatorSeries
import pandas as pd
from typing import Any


# =============================================================================
# Register built-in indicators
# =============================================================================

@IndicatorRegistry.register(
    indicator_id="ema",
    name="Exponential Moving Average",
    output_type="line",
    description="EMA of closing prices",
    params_schema={
        "period": {"type": "integer", "default": 20, "min": 2}
    }
)
class EMAIndicator:
    """EMA indicator."""
    def __init__(self, period: int = 20):
        self.period = period
    
    def compute(self, stepper: Any) -> IndicatorSeries:
        """Compute EMA from stepper data."""
        # Extract price data
        df = stepper.df if hasattr(stepper, 'df') else pd.DataFrame()
        
        if len(df) == 0:
            return IndicatorSeries(
                indicator_id=f"ema_{self.period}",
                name=f"EMA {self.period}",
                type="line",
                points=[]
            )
        
        # Calculate EMA
        ema = df['close'].ewm(span=self.period, adjust=False).mean()
        
        # Convert to points
        points = [
            {
                'time': row['time'].isoformat() if hasattr(row['time'], 'isoformat') else str(row['time']),
                'value': float(val) if pd.notna(val) else None
            }
            for (idx, row), val in zip(df.iterrows(), ema)
        ]
        
        return IndicatorSeries(
            indicator_id=f"ema_{self.period}",
            name=f"EMA {self.period}",
            type="line",
            points=points,
            style={'color': '#00ff00', 'lineWidth': 2}
        )


@IndicatorRegistry.register(
    indicator_id="atr",
    name="Average True Range",
    output_type="line",
    description="ATR volatility indicator",
    params_schema={
        "period": {"type": "integer", "default": 14, "min": 2}
    }
)
class ATRIndicator:
    """ATR indicator."""
    def __init__(self, period: int = 14):
        self.period = period
    
    def compute(self, stepper: Any) -> IndicatorSeries:
        """Compute ATR from stepper data."""
        df = stepper.df if hasattr(stepper, 'df') else pd.DataFrame()
        
        if len(df) == 0:
            return IndicatorSeries(
                indicator_id=f"atr_{self.period}",
                name=f"ATR {self.period}",
                type="line",
                points=[]
            )
        
        # Calculate True Range
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.ewm(span=self.period, adjust=False).mean()
        
        # Convert to points
        points = [
            {
                'time': row['time'].isoformat() if hasattr(row['time'], 'isoformat') else str(row['time']),
                'value': float(val) if pd.notna(val) else None
            }
            for (idx, row), val in zip(df.iterrows(), atr)
        ]
        
        return IndicatorSeries(
            indicator_id=f"atr_{self.period}",
            name=f"ATR {self.period}",
            type="line",
            points=points,
            style={'color': '#ff9900', 'lineWidth': 1}
        )


@IndicatorRegistry.register(
    indicator_id="vwap",
    name="Volume Weighted Average Price",
    output_type="line",
    description="VWAP - resets daily",
    params_schema={}
)
class VWAPIndicator:
    """VWAP indicator."""
    def __init__(self):
        pass
    
    def compute(self, stepper: Any) -> IndicatorSeries:
        """Compute VWAP from stepper data."""
        df = stepper.df if hasattr(stepper, 'df') else pd.DataFrame()
        
        if len(df) == 0:
            return IndicatorSeries(
                indicator_id="vwap",
                name="VWAP",
                type="line",
                points=[]
            )
        
        # Simple VWAP (not session-aware for now)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        
        # Convert to points
        points = [
            {
                'time': row['time'].isoformat() if hasattr(row['time'], 'isoformat') else str(row['time']),
                'value': float(val) if pd.notna(val) else None
            }
            for (idx, row), val in zip(df.iterrows(), vwap)
        ]
        
        return IndicatorSeries(
            indicator_id="vwap",
            name="VWAP",
            type="line",
            points=points,
            style={'color': '#ffff00', 'lineWidth': 2}
        )


# Auto-register on import
def register_all_indicators():
    """
    Register all available indicators.
    Call this at startup to populate the registry.
    """
    pass

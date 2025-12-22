"""
Dynamic Feature Registry

Solves the "Static Feature Definitions" problem by:
1. Moving from hardcoded dataclass to dictionary-based features
2. Creating an indicator library with composable definitions
3. Allowing FeatureConfig to accept JSON-driven feature lists

This enables agents to "invent" indicators without Python code changes.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Callable, Optional
import numpy as np
import pandas as pd

from src.features.indicators import (
    calculate_ema, calculate_rsi, calculate_atr, 
    calculate_vwap, calculate_adr
)


@dataclass
class FeatureDefinition:
    """
    Definition of a single feature that can be computed.
    
    Example:
        FeatureDefinition(
            name="sma_50",
            func="sma",
            params={"period": 50, "source": "close"}
        )
    """
    name: str  # Unique feature name
    func: str  # Function identifier (e.g., "sma", "rsi", "ema")
    params: Dict[str, Any] = field(default_factory=dict)
    timeframe: str = "1m"  # Which timeframe to compute on
    normalize_by_atr: bool = False  # Whether to normalize by ATR
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'func': self.func,
            'params': self.params,
            'timeframe': self.timeframe,
            'normalize_by_atr': self.normalize_by_atr,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureDefinition':
        return cls(
            name=data['name'],
            func=data['func'],
            params=data.get('params', {}),
            timeframe=data.get('timeframe', '1m'),
            normalize_by_atr=data.get('normalize_by_atr', False),
        )


class IndicatorLibrary:
    """
    Registry of available indicator functions.
    
    Agents can discover and use any registered indicator.
    
    Usage:
        # Register a custom indicator
        @IndicatorLibrary.register("custom_indicator")
        def compute_custom(series: pd.Series, **params) -> pd.Series:
            return series.rolling(params['window']).mean()
        
        # Use it
        value = IndicatorLibrary.compute("custom_indicator", series, window=20)
    """
    
    _registry: Dict[str, Callable] = {}
    _metadata: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register(
        cls,
        func_id: str,
        description: str = "",
        params_schema: Optional[Dict[str, Any]] = None
    ):
        """
        Decorator to register an indicator function.
        
        Args:
            func_id: Unique identifier for this indicator
            description: Human-readable description
            params_schema: Expected parameters with types/defaults
        """
        def decorator(func: Callable):
            cls._registry[func_id] = func
            cls._metadata[func_id] = {
                'description': description,
                'params_schema': params_schema or {},
            }
            return func
        return decorator
    
    @classmethod
    def compute(cls, func_id: str, *args, **kwargs) -> Any:
        """
        Compute an indicator by ID.
        
        Args:
            func_id: Indicator function ID
            *args, **kwargs: Arguments to pass to indicator function
            
        Returns:
            Computed indicator values
        """
        if func_id not in cls._registry:
            raise ValueError(
                f"Unknown indicator: {func_id}. "
                f"Available: {list(cls._registry.keys())}"
            )
        
        return cls._registry[func_id](*args, **kwargs)
    
    @classmethod
    def list_indicators(cls) -> List[Dict[str, Any]]:
        """List all available indicators with metadata."""
        return [
            {
                'func_id': func_id,
                'description': meta['description'],
                'params_schema': meta['params_schema'],
            }
            for func_id, meta in cls._metadata.items()
        ]
    
    @classmethod
    def get_metadata(cls, func_id: str) -> Dict[str, Any]:
        """Get metadata for a specific indicator."""
        if func_id not in cls._metadata:
            raise ValueError(f"Unknown indicator: {func_id}")
        return cls._metadata[func_id]


# =============================================================================
# Register Built-in Indicators
# =============================================================================

@IndicatorLibrary.register(
    "ema",
    description="Exponential Moving Average",
    params_schema={'period': 'int (default: 20)'}
)
def compute_ema(series: pd.Series, period: int = 20) -> pd.Series:
    """Compute EMA on a price series."""
    return calculate_ema(series, period)


@IndicatorLibrary.register(
    "sma",
    description="Simple Moving Average",
    params_schema={'period': 'int (default: 20)'}
)
def compute_sma(series: pd.Series, period: int = 20) -> pd.Series:
    """Compute SMA on a price series."""
    return series.rolling(window=period).mean()


@IndicatorLibrary.register(
    "rsi",
    description="Relative Strength Index",
    params_schema={'period': 'int (default: 14)'}
)
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI on a price series."""
    return calculate_rsi(series, period)


@IndicatorLibrary.register(
    "atr",
    description="Average True Range",
    params_schema={'period': 'int (default: 14)'}
)
def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute ATR (requires OHLC dataframe)."""
    return calculate_atr(df, period)


@IndicatorLibrary.register(
    "vwap",
    description="Volume Weighted Average Price",
    params_schema={'period': 'str (default: "session")'}
)
def compute_vwap(df: pd.DataFrame, period: str = "session") -> pd.Series:
    """Compute VWAP (requires OHLCV dataframe)."""
    return calculate_vwap(df, period)


@IndicatorLibrary.register(
    "adr",
    description="Average Daily Range",
    params_schema={'period': 'int (default: 14)'}
)
def compute_adr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute ADR (requires OHLC dataframe)."""
    return calculate_adr(df, period)


@IndicatorLibrary.register(
    "bollinger_bands",
    description="Bollinger Bands (mean +/- std_dev * period)",
    params_schema={'period': 'int (default: 20)', 'std_dev': 'float (default: 2.0)'}
)
def compute_bollinger_bands(
    series: pd.Series,
    period: int = 20,
    std_dev: float = 2.0
) -> Dict[str, pd.Series]:
    """
    Compute Bollinger Bands.
    
    Returns dict with 'upper', 'middle', 'lower' bands.
    """
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    
    return {
        'upper': middle + (std_dev * std),
        'middle': middle,
        'lower': middle - (std_dev * std),
    }


@IndicatorLibrary.register(
    "stochastic",
    description="Stochastic Oscillator",
    params_schema={'period': 'int (default: 14)', 'smooth_k': 'int (default: 3)'}
)
def compute_stochastic(
    df: pd.DataFrame,
    period: int = 14,
    smooth_k: int = 3
) -> Dict[str, pd.Series]:
    """
    Compute Stochastic Oscillator (%K and %D).
    
    Returns dict with 'k' and 'd' values.
    """
    lowest_low = df['low'].rolling(window=period).min()
    highest_high = df['high'].rolling(window=period).max()
    
    k_raw = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)
    k = k_raw.rolling(window=smooth_k).mean()
    d = k.rolling(window=3).mean()
    
    return {'k': k, 'd': d}


# =============================================================================
# Dynamic Feature Schema
# =============================================================================

@dataclass
class DynamicFeatureSchema:
    """
    Dynamic feature schema that can be configured via JSON.
    
    Replaces hardcoded ContextFeatures dataclass.
    
    Example:
        schema = DynamicFeatureSchema()
        schema.add_feature(FeatureDefinition(
            name="sma_50",
            func="sma",
            params={"period": 50},
            timeframe="5m"
        ))
        
        # Compute all features
        values = schema.compute_all(df_1m, df_5m, df_15m)
        # values = {"sma_50": 100.5, "rsi_14": 55.2, ...}
    """
    
    features: List[FeatureDefinition] = field(default_factory=list)
    
    def add_feature(self, feature: FeatureDefinition):
        """Add a feature to the schema."""
        self.features.append(feature)
    
    def compute_all(
        self,
        df_1m: pd.DataFrame,
        df_5m: Optional[pd.DataFrame] = None,
        df_15m: Optional[pd.DataFrame] = None,
        current_idx: Optional[int] = None,
        atr: float = 1.0
    ) -> Dict[str, float]:
        """
        Compute all features in the schema.
        
        Args:
            df_1m: 1-minute data
            df_5m: 5-minute data
            df_15m: 15-minute data
            current_idx: Current bar index (for causal computation)
            atr: ATR value for normalization
            
        Returns:
            Dictionary mapping feature names to values
        """
        results = {}
        
        # Select appropriate dataframe
        df_map = {
            '1m': df_1m,
            '5m': df_5m,
            '15m': df_15m,
        }
        
        for feature in self.features:
            df = df_map.get(feature.timeframe)
            if df is None:
                results[feature.name] = 0.0
                continue
            
            # Get data up to current point (causal)
            if current_idx is not None:
                df = df.iloc[:current_idx + 1]
            
            try:
                # Compute indicator
                value = IndicatorLibrary.compute(
                    feature.func,
                    df if feature.func in ['atr', 'vwap', 'adr', 'bollinger_bands', 'stochastic'] else df['close'],
                    **feature.params
                )
                
                # Get most recent value
                if isinstance(value, dict):
                    # Multi-value indicator (e.g., Bollinger Bands)
                    for sub_name, sub_series in value.items():
                        results[f"{feature.name}_{sub_name}"] = float(sub_series.iloc[-1]) if len(sub_series) > 0 else 0.0
                else:
                    final_value = float(value.iloc[-1]) if len(value) > 0 else 0.0
                    
                    # Normalize by ATR if requested
                    if feature.normalize_by_atr and atr > 0:
                        final_value = final_value / atr
                    
                    results[feature.name] = final_value
                    
            except Exception as e:
                # Fallback to 0 if computation fails
                results[feature.name] = 0.0
        
        return results
    
    def to_array(self, feature_values: Dict[str, float]) -> np.ndarray:
        """
        Convert feature dictionary to numpy array.
        
        Args:
            feature_values: Dictionary from compute_all()
            
        Returns:
            Numpy array with values in consistent order
        """
        # Expand multi-value features
        all_names = []
        for feature in self.features:
            if feature.func in ['bollinger_bands', 'stochastic']:
                # Multi-value features
                if feature.func == 'bollinger_bands':
                    all_names.extend([f"{feature.name}_upper", f"{feature.name}_middle", f"{feature.name}_lower"])
                elif feature.func == 'stochastic':
                    all_names.extend([f"{feature.name}_k", f"{feature.name}_d"])
            else:
                all_names.append(feature.name)
        
        return np.array([feature_values.get(name, 0.0) for name in all_names], dtype=np.float32)
    
    def get_dimension(self) -> int:
        """Get total feature dimension."""
        count = 0
        for feature in self.features:
            if feature.func == 'bollinger_bands':
                count += 3  # upper, middle, lower
            elif feature.func == 'stochastic':
                count += 2  # k, d
            else:
                count += 1
        return count
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize schema to dictionary."""
        return {
            'features': [f.to_dict() for f in self.features]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DynamicFeatureSchema':
        """Load schema from dictionary."""
        schema = cls()
        for feature_data in data.get('features', []):
            schema.add_feature(FeatureDefinition.from_dict(feature_data))
        return schema
    
    @classmethod
    def from_json(cls, json_str: str) -> 'DynamicFeatureSchema':
        """Load schema from JSON string."""
        import json
        return cls.from_dict(json.loads(json_str))

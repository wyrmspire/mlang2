"""
Feature Pipeline
Compose all features into a single bundle for model input.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Optional, Any

from src.sim.stepper import MarketStepper
from src.features.state import MarketState, get_market_state, normalize_ohlcv
from src.features.indicators import IndicatorValues
from src.features.levels import LevelValues
from src.features.time_features import TimeFeatures, compute_time_features
from src.features.context import ContextFeatures, compute_context_features
from src.config import DEFAULT_LOOKBACK_1M, DEFAULT_LOOKBACK_5M, DEFAULT_LOOKBACK_15M


@dataclass
class FeatureConfig:
    """Configuration for feature computation."""
    # Lookback windows (in bars)
    lookback_1m: int = DEFAULT_LOOKBACK_1M    # 120 bars = 2 hours
    lookback_5m: int = DEFAULT_LOOKBACK_5M    # 24 bars = 2 hours
    lookback_15m: int = DEFAULT_LOOKBACK_15M  # 8 bars = 2 hours
    
    # What to include
    include_ohlcv: bool = True
    include_indicators: bool = True
    include_levels: bool = True
    include_time: bool = True
    
    # Normalization
    price_norm: str = "zscore"  # 'zscore', 'minmax', 'none'
    
    def to_dict(self) -> dict:
        return {
            'lookback_1m': self.lookback_1m,
            'lookback_5m': self.lookback_5m,
            'lookback_15m': self.lookback_15m,
            'include_ohlcv': self.include_ohlcv,
            'include_indicators': self.include_indicators,
            'include_levels': self.include_levels,
            'include_time': self.include_time,
            'price_norm': self.price_norm,
        }


@dataclass
class Candle:
    """Simple candle for trigger compatibility."""
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass
class FeatureBundle:
    """
    Complete feature bundle for a single decision point.
    
    Separates price windows (for CNN) from context vector (for MLP).
    """
    # Price windows - (lookback, channels) for CNN
    x_price_1m: np.ndarray      # (120, 5) default
    x_price_5m: np.ndarray      # (24, 5) default  
    x_price_15m: np.ndarray     # (8, 5) default
    
    # Context vector - (dim,) for MLP
    x_context: np.ndarray       # (20,) default
    
    # Raw components (for debugging/logging)
    market_state: Optional[MarketState] = None
    indicators: Optional[IndicatorValues] = None
    levels: Optional[LevelValues] = None
    time_features: Optional[TimeFeatures] = None
    context_features: Optional[ContextFeatures] = None
    
    # Metadata
    bar_idx: int = 0
    timestamp: Optional[pd.Timestamp] = None
    current_price: float = 0.0
    atr: float = 0.0
    
    @property
    def candles(self):
        """Return ohlcv_1m as list of Candle objects for trigger compatibility."""
        if self.market_state is None:
            return []
        ohlcv = self.market_state.ohlcv_1m
        return [Candle(open=row[0], high=row[1], low=row[2], close=row[3], volume=row[4]) for row in ohlcv]


def compute_features(
    stepper: MarketStepper,
    config: FeatureConfig,
    df_5m: pd.DataFrame = None,
    df_15m: pd.DataFrame = None,
    df_1h: pd.DataFrame = None,
    df_4h: pd.DataFrame = None,
    precomputed_indicators: Dict[int, IndicatorValues] = None,
    precomputed_levels: Dict[int, LevelValues] = None,
) -> FeatureBundle:
    """
    Compute all features at current stepper position.
    
    All data access is CAUSAL (via stepper.get_history only).
    
    Args:
        stepper: MarketStepper at current position
        config: Feature configuration
        df_5m, df_15m, df_1h, df_4h: Pre-resampled higher timeframe data
        precomputed_indicators: Optional pre-computed indicator values by bar_idx
        precomputed_levels: Optional pre-computed level values by bar_idx
        
    Returns:
        FeatureBundle with all features
    """
    # Get market state
    state = get_market_state(
        stepper,
        df_5m=df_5m,
        df_15m=df_15m,
        lookback_1m=config.lookback_1m,
        lookback_5m=config.lookback_5m,
        lookback_15m=config.lookback_15m,
    )
    
    # Normalize price windows
    x_price_1m = normalize_ohlcv(state.ohlcv_1m, config.price_norm)
    x_price_5m = normalize_ohlcv(state.ohlcv_5m, config.price_norm)
    x_price_15m = normalize_ohlcv(state.ohlcv_15m, config.price_norm)
    
    # Get indicator values (from precomputed or compute)
    bar_idx = state.current_bar_idx
    if precomputed_indicators and bar_idx in precomputed_indicators:
        indicators = precomputed_indicators[bar_idx]
    else:
        indicators = IndicatorValues()  # Empty if not provided
    
    # Get level values
    if precomputed_levels and bar_idx in precomputed_levels:
        levels = precomputed_levels[bar_idx]
    else:
        levels = LevelValues()
    
    # Time features
    time_feats = TimeFeatures()
    if state.current_time:
        time_feats = compute_time_features(state.current_time)
    
    # Context features (combines everything)
    atr = indicators.atr_5m_14 if indicators.atr_5m_14 > 0 else 1.0
    context_feats = compute_context_features(
        state.current_price,
        indicators,
        levels,
        time_feats,
        atr=atr
    )
    
    x_context = context_feats.to_array()
    
    return FeatureBundle(
        x_price_1m=x_price_1m,
        x_price_5m=x_price_5m,
        x_price_15m=x_price_15m,
        x_context=x_context,
        market_state=state,
        indicators=indicators,
        levels=levels,
        time_features=time_feats,
        context_features=context_feats,
        bar_idx=bar_idx,
        timestamp=state.current_time,
        current_price=state.current_price,
        atr=atr,
    )


def precompute_indicators(
    df_1m: pd.DataFrame,
    df_5m: pd.DataFrame,
    df_15m: pd.DataFrame,
) -> Dict[int, IndicatorValues]:
    """
    Pre-compute all indicators for efficiency.
    
    Returns dict mapping bar_idx to IndicatorValues.
    This should be cached via FeatureStore.
    """
    from src.features.indicators import calculate_ema, calculate_rsi, calculate_atr, calculate_vwap, calculate_adr
    
    # Compute on each timeframe
    df_5m = df_5m.copy()
    df_5m['ema_20'] = calculate_ema(df_5m['close'], 20)
    df_5m['ema_200'] = calculate_ema(df_5m['close'], 200)
    df_5m['rsi_14'] = calculate_rsi(df_5m['close'], 14)
    df_5m['atr_14'] = calculate_atr(df_5m, 14)
    
    df_15m = df_15m.copy()
    df_15m['ema_20'] = calculate_ema(df_15m['close'], 20)
    df_15m['ema_200'] = calculate_ema(df_15m['close'], 200)
    df_15m['rsi_14'] = calculate_rsi(df_15m['close'], 14)
    df_15m['atr_14'] = calculate_atr(df_15m, 14)
    
    # VWAP on 1m
    df_1m = df_1m.copy()
    df_1m['vwap_session'] = calculate_vwap(df_1m, period='session')
    df_1m['vwap_weekly'] = calculate_vwap(df_1m, period='weekly')
    
    # ADR
    df_1m['adr'] = calculate_adr(df_1m, 14)
    
    # Map back to 1m bar indices
    # Forward fill 5m and 15m data to align with 1m timestamps
    # We assume df_1m, df_5m, df_15m are indexed by time or have 'time' column
    
    # Ensure all have time index for alignment
    def ensure_time_index(df):
        if 'time' in df.columns:
            return df.set_index('time')
        return df
        
    df_1m_idx = ensure_time_index(df_1m)
    df_5m_idx = ensure_time_index(df_5m)
    df_15m_idx = ensure_time_index(df_15m)
    
    # Reindex 5m/15m to 1m (forward fill)
    df_5m_aligned = df_5m_idx.reindex(df_1m_idx.index, method='ffill')
    df_15m_aligned = df_15m_idx.reindex(df_1m_idx.index, method='ffill')
    
    result = {}
    
    for idx, (time, row) in enumerate(df_1m_idx.iterrows()):
        row_5m = df_5m_aligned.iloc[idx]
        row_15m = df_15m_aligned.iloc[idx]
        
        result[idx] = IndicatorValues(
            ema_5m_20=float(row_5m.get('ema_20', 0.0)),
            ema_15m_20=float(row_15m.get('ema_20', 0.0)),
            ema_5m_200=float(row_5m.get('ema_200', 0.0)),
            ema_15m_200=float(row_15m.get('ema_200', 0.0)),
            rsi_5m_14=float(row_5m.get('rsi_14', 50.0)),
            rsi_15m_14=float(row_15m.get('rsi_14', 50.0)),
            atr_5m_14=float(row_5m.get('atr_14', 0.0)),
            atr_15m_14=float(row_15m.get('atr_14', 0.0)),
            adr_14=float(row.get('adr', 0.0)),
            vwap_session=float(row.get('vwap_session', 0.0)),
            vwap_weekly=float(row.get('vwap_weekly', 0.0)),
        )
    
    return result

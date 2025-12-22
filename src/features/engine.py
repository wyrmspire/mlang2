"""
Unified Feature Engine

Single source of truth for feature computation.
Ensures TRAIN, SCAN, REPLAY, and INFER all use identical normalization.

This prevents "inference skew" - where training and production drift apart.
"""

import numpy as np
from typing import List, Dict, Any, Union
from dataclasses import dataclass


@dataclass
class FeatureConfig:
    """Configuration for feature computation."""
    lookback: int = 30
    normalization: str = "percent_change"  # "percent_change", "zscore", "minmax"
    volume_norm: str = "max"  # "max", "mean", "none"


def normalize_ohlcv_window(
    ohlcv_data: Union[np.ndarray, List[Dict[str, float]]],
    config: FeatureConfig = None
) -> np.ndarray:
    """
    Normalize OHLCV window for model input.
    
    THIS IS THE SINGLE SOURCE OF TRUTH FOR NORMALIZATION.
    Used by:
    - Training pipeline (train_ifvg_4class.py)
    - Inference endpoint (/infer)
    - Backend simulation (test_backend_simulation.py)
    - Frontend (via /infer API)
    
    Args:
        ohlcv_data: Either:
            - np.ndarray of shape (N, 5) [open, high, low, close, volume]
            - List of dicts [{open, high, low, close, volume}, ...]
        config: FeatureConfig (uses defaults if None)
    
    Returns:
        np.ndarray of shape (5, N) normalized for model input
    """
    if config is None:
        config = FeatureConfig()
    
    # Convert list of dicts to numpy array
    if isinstance(ohlcv_data, list):
        ohlcv_array = np.array([
            [b['open'], b['high'], b['low'], b['close'], b.get('volume', 0)]
            for b in ohlcv_data
        ], dtype=np.float32)
    else:
        ohlcv_array = np.asarray(ohlcv_data, dtype=np.float32)
    
    # Ensure correct shape (N, 5)
    if ohlcv_array.ndim == 1:
        ohlcv_array = ohlcv_array.reshape(-1, 5)
    
    # Transpose to (5, N) for model input
    x = ohlcv_array.T.copy()
    
    # Apply normalization to OHLC (indices 0-3)
    if config.normalization == "percent_change":
        # Normalize by first bar's close (percent change)
        first_close = x[3, 0]
        if first_close > 0:
            x[0:4] = (x[0:4] - first_close) / first_close * 100
    
    elif config.normalization == "zscore":
        # Z-score normalization
        for i in range(4):
            mean = x[i].mean()
            std = x[i].std()
            if std > 0:
                x[i] = (x[i] - mean) / std
    
    elif config.normalization == "minmax":
        # Min-max normalization to [0, 1]
        for i in range(4):
            min_val = x[i].min()
            max_val = x[i].max()
            if max_val > min_val:
                x[i] = (x[i] - min_val) / (max_val - min_val)
    
    # Apply volume normalization
    if config.volume_norm == "max":
        max_vol = x[4].max()
        if max_vol > 0:
            x[4] = x[4] / max_vol
        else:
            x[4] = 0
    elif config.volume_norm == "mean":
        mean_vol = x[4].mean()
        if mean_vol > 0:
            x[4] = x[4] / mean_vol
    elif config.volume_norm == "none":
        pass  # Keep raw volume
    
    return x


def compute_atr(bars: Union[np.ndarray, List[Dict[str, float]]], period: int = 14) -> float:
    """
    Compute Average True Range.
    
    Args:
        bars: OHLCV data (N, 5) or list of dicts
        period: ATR period
    
    Returns:
        ATR value
    """
    if isinstance(bars, list):
        highs = np.array([b['high'] for b in bars])
        lows = np.array([b['low'] for b in bars])
        closes = np.array([b['close'] for b in bars])
    else:
        highs = bars[:, 1]
        lows = bars[:, 2]
        closes = bars[:, 3]
    
    if len(bars) < 2:
        return highs[0] - lows[0] if len(bars) > 0 else 1.0
    
    # True Range = max(high-low, |high-prev_close|, |low-prev_close|)
    tr = np.zeros(len(bars))
    tr[0] = highs[0] - lows[0]
    
    for i in range(1, len(bars)):
        hl = highs[i] - lows[i]
        hpc = abs(highs[i] - closes[i-1])
        lpc = abs(lows[i] - closes[i-1])
        tr[i] = max(hl, hpc, lpc)
    
    # Use last 'period' bars for ATR
    atr = tr[-period:].mean() if len(tr) >= period else tr.mean()
    return float(atr)


def bars_to_model_input(
    bars: Union[np.ndarray, List[Dict[str, float]]],
    lookback: int = 30,
    config: FeatureConfig = None
) -> np.ndarray:
    """
    Convert bars to model input tensor.
    
    Takes last 'lookback' bars and normalizes them.
    
    Args:
        bars: OHLCV data
        lookback: Number of bars to use
        config: Feature configuration
    
    Returns:
        np.ndarray of shape (5, lookback) ready for model
    """
    if config is None:
        config = FeatureConfig(lookback=lookback)
    
    # Convert to numpy if needed
    if isinstance(bars, list):
        bars = np.array([
            [b['open'], b['high'], b['low'], b['close'], b.get('volume', 0)]
            for b in bars
        ], dtype=np.float32)
    
    # Take last N bars
    if len(bars) > lookback:
        bars = bars[-lookback:]
    elif len(bars) < lookback:
        # Pad with first bar if too few
        padding = np.tile(bars[0:1], (lookback - len(bars), 1))
        bars = np.vstack([padding, bars])
    
    return normalize_ohlcv_window(bars, config)

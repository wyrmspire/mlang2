"""
Swing Point Detection

Identifies swing highs and lows in price data for structure analysis.
A swing high is a bar whose high is higher than N bars before and after.
A swing low is a bar whose low is lower than N bars before and after.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class SwingPoint:
    """Represents a swing high or low."""
    price: float
    bar_idx: int
    time: pd.Timestamp
    is_high: bool  # True for swing high, False for swing low


def find_swings(
    df: pd.DataFrame,
    lookback: int = 5,
    max_points: int = 20
) -> Tuple[List[SwingPoint], List[SwingPoint]]:
    """
    Find swing highs and lows in OHLCV data.
    
    A swing high is a bar whose high is higher than `lookback` bars before and after.
    A swing low is a bar whose low is lower than `lookback` bars before and after.
    
    Args:
        df: OHLCV DataFrame with 'high', 'low', 'time' columns
        lookback: Number of bars on each side to confirm swing
        max_points: Maximum number of swing points to return per type
        
    Returns:
        Tuple of (swing_highs, swing_lows) as lists of SwingPoint
    """
    if df is None or len(df) < lookback * 2 + 1:
        return [], []
    
    # Use recent data for efficiency
    recent = df.tail(100).copy()
    if len(recent) < lookback * 2 + 1:
        return [], []
    
    highs = recent['high'].values
    lows = recent['low'].values
    times = recent['time'].values if 'time' in recent.columns else [pd.Timestamp.now()] * len(recent)
    indices = recent.index.tolist()
    
    swing_highs: List[SwingPoint] = []
    swing_lows: List[SwingPoint] = []
    
    # Iterate through bars that can be confirmed as swings
    for i in range(lookback, len(recent) - lookback):
        # Check for swing high
        is_swing_high = True
        for j in range(1, lookback + 1):
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                is_swing_high = False
                break
        
        if is_swing_high:
            swing_highs.append(SwingPoint(
                price=highs[i],
                bar_idx=indices[i],
                time=pd.Timestamp(times[i]),
                is_high=True
            ))
        
        # Check for swing low
        is_swing_low = True
        for j in range(1, lookback + 1):
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                is_swing_low = False
                break
        
        if is_swing_low:
            swing_lows.append(SwingPoint(
                price=lows[i],
                bar_idx=indices[i],
                time=pd.Timestamp(times[i]),
                is_high=False
            ))
    
    # Return most recent first, limited
    return swing_highs[-max_points:], swing_lows[-max_points:]


def count_levels_swept(
    swings: List[SwingPoint],
    price_from: float,
    price_to: float
) -> int:
    """
    Count how many swing levels were swept by a price move.
    
    Args:
        swings: List of swing points to check
        price_from: Starting price of the move
        price_to: Ending price of the move
        
    Returns:
        Number of swing levels between price_from and price_to
    """
    count = 0
    low_price = min(price_from, price_to)
    high_price = max(price_from, price_to)
    
    for swing in swings:
        if low_price < swing.price < high_price:
            count += 1
    
    return count

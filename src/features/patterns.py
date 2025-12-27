"""
Chart Pattern Recognition Features

Identifies intraday chart patterns like flags, wedges, and pullbacks.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class PatternResult:
    pattern_type: str  # "FLAG", "WEDGE", "PULLBACK"
    direction: str     # "BULLISH", "BEARISH"
    start_idx: int     # Index in the passed dataframe
    end_idx: int
    entry_price: float
    stop_loss: float
    target_price: float
    confidence: float  # 0.0 to 1.0

def _get_slope(y: np.ndarray) -> float:
    """Calculate linear regression slope of an array."""
    x = np.arange(len(y))
    if len(x) < 2:
        return 0.0
    slope, _ = np.polyfit(x, y, 1)
    return slope

def detect_flags(
    df: pd.DataFrame,
    lookback: int = 30,
    pole_bars: int = 15,
    flag_bars: int = 15,
    atr_period: int = 14
) -> List[PatternResult]:
    """
    Detect flag patterns (Bull/Bear Flags).

    Args:
        df: DataFrame with open, high, low, close
        lookback: Total window size to analyze
        pole_bars: Number of bars for the flagpole
        flag_bars: Number of bars for the flag/consolidation
        atr_period: ATR period for volatility normalization

    Returns:
        List of detected patterns.
    """
    results = []
    if len(df) < lookback + atr_period:
        return results

    # Calculate ATR for normalization
    # Simple TR calculation
    tr = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    atr = tr.rolling(window=atr_period).mean()

    # Iterate through the dataframe
    # We only check the most recent window for efficiency if used in real-time
    # But if scanning history, we might want to slide.
    # For now, let's just check the *current* window (tail) to act as a signal generator.
    # If the user wants historical scan, they can loop outside or we can loop here.
    # Given the request is "analyze price historically", I'll loop.

    for i in range(lookback, len(df) + 1):
        window = df.iloc[i-lookback:i]
        curr_atr = atr.iloc[i-1]

        if pd.isna(curr_atr) or curr_atr == 0:
            continue

        pole_data = window.iloc[:pole_bars]
        flag_data = window.iloc[pole_bars:]

        # --- BULL FLAG CHECK ---
        # 1. Pole: Strong move up
        pole_move = pole_data['close'].iloc[-1] - pole_data['open'].iloc[0]
        if pole_move > 2.0 * curr_atr: # Significant move
            # 2. Flag: Consolidation sloping down
            flag_highs = flag_data['high'].values
            flag_lows = flag_data['low'].values

            slope_highs = _get_slope(flag_highs)
            slope_lows = _get_slope(flag_lows)

            # Check for downward slope
            if slope_highs < 0 and slope_lows < 0:
                # Check for parallel (roughly)
                if abs(slope_highs - slope_lows) < 0.5 * (curr_atr / flag_bars):
                    # Found Bull Flag
                    results.append(PatternResult(
                        pattern_type="FLAG",
                        direction="BULLISH",
                        start_idx=window.index[0],
                        end_idx=window.index[-1],
                        entry_price=flag_data['high'].max(), # Breakout entry
                        stop_loss=flag_data['low'].min(),
                        target_price=flag_data['high'].max() + pole_move, # Measured move
                        confidence=0.8
                    ))

        # --- BEAR FLAG CHECK ---
        # 1. Pole: Strong move down
        pole_move = pole_data['close'].iloc[-1] - pole_data['open'].iloc[0]
        if pole_move < -2.0 * curr_atr:
            # 2. Flag: Consolidation sloping up
            flag_highs = flag_data['high'].values
            flag_lows = flag_data['low'].values

            slope_highs = _get_slope(flag_highs)
            slope_lows = _get_slope(flag_lows)

            # Check for upward slope
            if slope_highs > 0 and slope_lows > 0:
                # Check for parallel
                if abs(slope_highs - slope_lows) < 0.5 * (curr_atr / flag_bars):
                    # Found Bear Flag
                    results.append(PatternResult(
                        pattern_type="FLAG",
                        direction="BEARISH",
                        start_idx=window.index[0],
                        end_idx=window.index[-1],
                        entry_price=flag_data['low'].min(),
                        stop_loss=flag_data['high'].max(),
                        target_price=flag_data['low'].min() + pole_move,
                        confidence=0.8
                    ))

    return results

def detect_wedges(
    df: pd.DataFrame,
    lookback: int = 30,
    atr_period: int = 14
) -> List[PatternResult]:
    """
    Detect wedge patterns (Falling/Rising Wedges).
    """
    results = []
    if len(df) < lookback + atr_period:
        return results

    tr = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    atr = tr.rolling(window=atr_period).mean()

    for i in range(lookback, len(df) + 1):
        window = df.iloc[i-lookback:i]
        curr_atr = atr.iloc[i-1]

        if pd.isna(curr_atr) or curr_atr == 0:
            continue

        highs = window['high'].values
        lows = window['low'].values

        slope_highs = _get_slope(highs)
        slope_lows = _get_slope(lows)

        # --- FALLING WEDGE (BULLISH) ---
        # Both slopes negative, but lows steeper than highs (converging)
        # OR Highs negative, Lows flat/slightly negative
        # slope_lows > slope_highs means lows are LESS NEGATIVE (flatter) than highs?
        # Wait, if both negative: -1 (highs) vs -0.5 (lows). -0.5 > -1. Correct.
        # But for falling wedge, usually lows are steeper? No, highs are steeper?
        # Falling wedge: Price narrows. Highs drop fast. Lows drop slow.
        # So slope_highs is more negative (smaller). slope_lows is less negative (larger).
        # So slope_lows > slope_highs.

        if slope_highs < 0 and slope_lows < 0:
            if slope_lows > slope_highs: # Converging
                 results.append(PatternResult(
                    pattern_type="WEDGE",
                    direction="BULLISH",
                    start_idx=window.index[0],
                    end_idx=window.index[-1],
                    entry_price=window['high'].iloc[-1],
                    stop_loss=window['low'].min(),
                    target_price=window['high'].iloc[-1] + (window['high'].max() - window['low'].min()),
                    confidence=0.7
                ))

        # --- RISING WEDGE (BEARISH) ---
        # Both slopes positive, but lows steeper (converging)
        if slope_highs > 0 and slope_lows > 0:
             if slope_lows > slope_highs: # Converging (Lows rising faster than Highs? No, Highs rising slower than Lows)
                # Wait, for rising wedge:
                # Highs slope positive. Lows slope positive.
                # Lines converge. So Highs slope < Lows slope.
                results.append(PatternResult(
                    pattern_type="WEDGE",
                    direction="BEARISH",
                    start_idx=window.index[0],
                    end_idx=window.index[-1],
                    entry_price=window['low'].iloc[-1],
                    stop_loss=window['high'].max(),
                    target_price=window['low'].iloc[-1] - (window['high'].max() - window['low'].min()),
                    confidence=0.7
                ))

    return results

def detect_pullback(
    df: pd.DataFrame,
    ema_period: int = 20,
    lookback: int = 5
) -> List[PatternResult]:
    """
    Detect pullbacks to EMA.
    """
    results = []
    if len(df) < ema_period + lookback:
        return results

    ema = df['close'].ewm(span=ema_period, adjust=False).mean()

    # Check just the last bar for efficiency (this is more of a signal check)
    # But to be consistent, we scan

    for i in range(ema_period + lookback, len(df) + 1):
        curr_idx = i - 1
        curr_close = df['close'].iloc[curr_idx]
        curr_ema = ema.iloc[curr_idx]
        prev_close = df['close'].iloc[curr_idx-1]

        # Determine trend from EMA slope
        # Need to ensure we don't go out of bounds
        if curr_idx < 5:
            continue

        ema_slope = ema.iloc[curr_idx] - ema.iloc[curr_idx-5]

        # Bullish Pullback
        # Trend is up, Price pulled back to near EMA
        if ema_slope > 0:
            # Check if we touched EMA recently or are close
            dist = (curr_close - curr_ema) / curr_ema
            # Relax tolerance for testing
            if abs(dist) < 0.005: # 0.5%
                results.append(PatternResult(
                    pattern_type="PULLBACK",
                    direction="BULLISH",
                    start_idx=df.index[curr_idx],
                    end_idx=df.index[curr_idx],
                    entry_price=curr_close,
                    stop_loss=curr_ema * 0.995,
                    target_price=curr_close * 1.02,
                    confidence=0.6
                ))

        # Bearish Pullback
        if ema_slope < 0:
            dist = (curr_close - curr_ema) / curr_ema
            if abs(dist) < 0.005:
                results.append(PatternResult(
                    pattern_type="PULLBACK",
                    direction="BEARISH",
                    start_idx=df.index[curr_idx],
                    end_idx=df.index[curr_idx],
                    entry_price=curr_close,
                    stop_loss=curr_ema * 1.005,
                    target_price=curr_close * 0.98,
                    confidence=0.6
                ))

    return results

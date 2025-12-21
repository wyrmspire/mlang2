"""
Fair Value Gap (FVG) Detection

Identifies imbalances in price action on specified timeframes.
FVGs represent areas where price moved quickly, leaving a gap between candle wicks.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class FairValueGap:
    """Represents a Fair Value Gap in price."""
    high: float           # Top of the gap
    low: float            # Bottom of the gap
    midpoint: float       # 50% level for entry
    direction: str        # "BULLISH" or "BEARISH"
    bar_idx: int          # Bar index where it formed (the impulse candle)
    bar_time: pd.Timestamp  # Time of impulse candle
    gap_size: float       # Size of the gap in points
    
    def contains_price(self, price: float, min_pct: float = 0.5) -> bool:
        """
        Check if price has entered the FVG by at least min_pct.
        
        Args:
            price: Current price
            min_pct: Minimum percentage into FVG required (0.5 = 50%)
            
        Returns:
            True if price is at least min_pct into the FVG
        """
        if self.direction == "BULLISH":
            # For bullish FVG, price retracing down into gap
            if price > self.high:
                return False  # Above the gap
            if price <= self.low:
                return True   # Fully through gap
            # Calculate penetration percentage
            pct_into = (self.high - price) / self.gap_size if self.gap_size > 0 else 0
            return pct_into >= min_pct
        else:
            # For bearish FVG, price retracing up into gap
            if price < self.low:
                return False  # Below the gap
            if price >= self.high:
                return True   # Fully through gap
            pct_into = (price - self.low) / self.gap_size if self.gap_size > 0 else 0
            return pct_into >= min_pct


def find_fvg(
    df: pd.DataFrame,
    lookback: int = 20,
    min_gap_atr: float = 0.2,
    atr: float = 5.0
) -> List[FairValueGap]:
    """
    Find Fair Value Gaps in price data.
    
    FVG Definition:
    - Bullish FVG: Gap between candle[i-1].high and candle[i+1].low
      (price gapped up through the middle candle)
    - Bearish FVG: Gap between candle[i-1].low and candle[i+1].high
      (price gapped down through the middle candle)
    
    Args:
        df: OHLCV DataFrame with 'high', 'low', 'time' columns
        lookback: Number of bars to look back for FVGs
        min_gap_atr: Minimum gap size as ATR multiple
        atr: Current ATR for filtering
        
    Returns:
        List of FairValueGap objects, most recent first
    """
    if df is None or len(df) < 3:
        return []
    
    recent = df.tail(lookback + 2).copy()
    if len(recent) < 3:
        return []
    
    fvgs = []
    min_gap = min_gap_atr * atr
    
    # We need at least 3 candles to detect an FVG
    # The FVG forms between candle i-1 and i+1, with i being the impulse
    for i in range(1, len(recent) - 1):
        prev_bar = recent.iloc[i - 1]
        impulse_bar = recent.iloc[i]
        next_bar = recent.iloc[i + 1]
        
        # Get the original index for bar_idx
        bar_idx = recent.index[i]
        bar_time = impulse_bar.get('time', pd.Timestamp.now())
        
        # Check for BULLISH FVG (gap up)
        # Gap exists if next_bar.low > prev_bar.high
        bullish_gap = next_bar['low'] - prev_bar['high']
        if bullish_gap > min_gap:
            fvg = FairValueGap(
                high=next_bar['low'],      # Top of gap
                low=prev_bar['high'],       # Bottom of gap
                midpoint=(next_bar['low'] + prev_bar['high']) / 2,
                direction="BULLISH",
                bar_idx=bar_idx,
                bar_time=pd.Timestamp(bar_time) if not isinstance(bar_time, pd.Timestamp) else bar_time,
                gap_size=bullish_gap
            )
            fvgs.append(fvg)
        
        # Check for BEARISH FVG (gap down)
        # Gap exists if prev_bar.low > next_bar.high
        bearish_gap = prev_bar['low'] - next_bar['high']
        if bearish_gap > min_gap:
            fvg = FairValueGap(
                high=prev_bar['low'],       # Top of gap
                low=next_bar['high'],       # Bottom of gap
                midpoint=(prev_bar['low'] + next_bar['high']) / 2,
                direction="BEARISH",
                bar_idx=bar_idx,
                bar_time=pd.Timestamp(bar_time) if not isinstance(bar_time, pd.Timestamp) else bar_time,
                gap_size=bearish_gap
            )
            fvgs.append(fvg)
    
    # Return most recent first
    return list(reversed(fvgs))


def find_most_recent_fvg(
    df: pd.DataFrame,
    direction: str,
    lookback: int = 20,
    min_gap_atr: float = 0.2,
    atr: float = 5.0
) -> Optional[FairValueGap]:
    """
    Find the most recent FVG in the specified direction.
    
    Args:
        df: OHLCV DataFrame
        direction: "BULLISH" or "BEARISH"
        lookback: Bars to look back
        min_gap_atr: Minimum gap size
        atr: Current ATR
        
    Returns:
        Most recent FVG matching direction, or None
    """
    fvgs = find_fvg(df, lookback, min_gap_atr, atr)
    
    for fvg in fvgs:
        if fvg.direction == direction.upper():
            return fvg
    
    return None


def is_impulse_candle(
    candle: pd.Series,
    direction: str,
    min_body_pct: float = 0.6
) -> bool:
    """
    Check if a candle is an impulse candle (strong directional move).
    
    An impulse candle has:
    - Body at least min_body_pct of the total range
    - Close in the direction of the move
    
    Args:
        candle: Single candle row with OHLC
        direction: Expected direction ("BULLISH" or "BEARISH")
        min_body_pct: Minimum body percentage of range
        
    Returns:
        True if this is an impulse candle
    """
    body = abs(candle['close'] - candle['open'])
    full_range = candle['high'] - candle['low']
    
    if full_range <= 0:
        return False
    
    body_pct = body / full_range
    
    if body_pct < min_body_pct:
        return False
    
    if direction == "BULLISH":
        return candle['close'] > candle['open']
    else:
        return candle['close'] < candle['open']


def find_impulse_with_fvg(
    df_5m: pd.DataFrame,
    direction: str,
    lookback: int = 10,
    atr: float = 5.0
) -> Optional[tuple]:
    """
    Find an impulse candle that created an FVG.
    
    Used after a level break to identify structure change.
    
    Args:
        df_5m: 5-minute OHLCV data
        direction: Expected impulse direction
        lookback: Bars to look back
        atr: Current ATR
        
    Returns:
        Tuple of (impulse_bar_idx, FairValueGap) or None
    """
    if df_5m is None or len(df_5m) < 3:
        return None
    
    recent = df_5m.tail(lookback + 2).copy()
    
    for i in range(1, len(recent) - 1):
        impulse_bar = recent.iloc[i]
        
        if not is_impulse_candle(impulse_bar, direction):
            continue
        
        prev_bar = recent.iloc[i - 1]
        next_bar = recent.iloc[i + 1]
        
        # Check for FVG in the expected direction
        if direction == "BULLISH":
            gap = next_bar['low'] - prev_bar['high']
            if gap > 0.2 * atr:
                fvg = FairValueGap(
                    high=next_bar['low'],
                    low=prev_bar['high'],
                    midpoint=(next_bar['low'] + prev_bar['high']) / 2,
                    direction="BULLISH",
                    bar_idx=recent.index[i],
                    bar_time=pd.Timestamp(impulse_bar.get('time', pd.Timestamp.now())),
                    gap_size=gap
                )
                return (recent.index[i], fvg)
        else:
            gap = prev_bar['low'] - next_bar['high']
            if gap > 0.2 * atr:
                fvg = FairValueGap(
                    high=prev_bar['low'],
                    low=next_bar['high'],
                    midpoint=(prev_bar['low'] + next_bar['high']) / 2,
                    direction="BEARISH",
                    bar_idx=recent.index[i],
                    bar_time=pd.Timestamp(impulse_bar.get('time', pd.Timestamp.now())),
                    gap_size=gap
                )
                return (recent.index[i], fvg)
    
    return None

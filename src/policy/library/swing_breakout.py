"""
Swing Breakout Scanner
Triggers trades when price breaks above/below recent swing structure on 15m timeframe.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass

from src.policy.scanners import Scanner, ScannerResult
from src.features.state import MarketState
from src.features.pipeline import FeatureBundle


@dataclass
class SwingBreakoutState:
    """Tracks swing levels and cooldown."""
    last_trigger_bar: int = -1000
    last_swing_high: float = 0.0
    last_swing_low: float = 0.0


class SwingBreakoutScanner(Scanner):
    """
    Scanner that triggers on breakout of 15m swing structure.
    
    Logic:
    1. Compute swing high/low from last N 15m bars.
    2. Trigger LONG when current close breaks above swing high.
    3. Trigger SHORT when current close breaks below swing low.
    4. Stop placed at opposite swing level (structure-based).
    5. TP placed at next structure level (scanned).
    
    Config:
        lookback_bars: Number of 15m bars to look back for swing detection.
        min_atr_distance: Minimum breakout distance in ATR to avoid noise.
        cooldown_bars: Minimum 1m bars between triggers.
    """
    
    def __init__(
        self,
        lookback_bars: int = 10,
        min_atr_distance: float = 0.3,
        cooldown_bars: int = 15,
    ):
        self.lookback_bars = lookback_bars
        self.min_atr_distance = min_atr_distance
        self.cooldown_bars = cooldown_bars
        self._state = SwingBreakoutState()
        self._last_15m_bar_time: Optional[pd.Timestamp] = None
    
    @property
    def scanner_id(self) -> str:
        return f"swing_breakout_15m_{self.lookback_bars}"
    
    def _compute_swing_levels(self, df_15m: pd.DataFrame, current_time: pd.Timestamp) -> tuple:
        """
        Compute swing high and swing low from recent 15m bars.
        
        Returns:
            (swing_high, swing_low, swing_high_idx, swing_low_idx)
        """
        if df_15m is None or df_15m.empty:
            return (0.0, 0.0, -1, -1)
        
        # Get bars up to current time
        mask = df_15m['time'] <= current_time
        recent = df_15m.loc[mask].tail(self.lookback_bars + 1)  # +1 to exclude current bar
        
        if len(recent) < 3:
            return (0.0, 0.0, -1, -1)
        
        # Exclude current bar (last one) for swing computation
        lookback = recent.iloc[:-1] if len(recent) > 1 else recent
        
        swing_high = lookback['high'].max()
        swing_low = lookback['low'].min()
        
        swing_high_idx = lookback['high'].idxmax()
        swing_low_idx = lookback['low'].idxmin()
        
        return (swing_high, swing_low, swing_high_idx, swing_low_idx)
    
    def _find_next_structure_level(
        self, 
        df_15m: pd.DataFrame, 
        current_time: pd.Timestamp,
        direction: str,
        current_price: float
    ) -> float:
        """
        Find the next structure level for take profit.
        
        For LONG: Find next resistance (high) above current price.
        For SHORT: Find next support (low) below current price.
        """
        if df_15m is None or df_15m.empty:
            return 0.0
        
        # Look at more bars for TP target
        mask = df_15m['time'] <= current_time
        recent = df_15m.loc[mask].tail(self.lookback_bars * 2)
        
        if direction == "LONG":
            # Find highs above current price
            highs = recent['high'].values
            above_highs = [h for h in highs if h > current_price]
            if above_highs:
                # Return the nearest high above
                return min(above_highs)
            else:
                # No structure above, use highest high
                return max(highs) if len(highs) > 0 else current_price
        else:  # SHORT
            # Find lows below current price
            lows = recent['low'].values
            below_lows = [l for l in lows if l < current_price]
            if below_lows:
                # Return the nearest low below
                return max(below_lows)
            else:
                # No structure below, use lowest low
                return min(lows) if len(lows) > 0 else current_price
    
    def scan(
        self,
        state: MarketState,
        features: FeatureBundle,
        df_15m: pd.DataFrame = None
    ) -> ScannerResult:
        """
        Check for swing breakout.
        
        Note: df_15m must be passed in for swing computation.
        """
        t = features.timestamp
        if t is None:
            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
        
        # Cooldown check
        if features.bar_idx - self._state.last_trigger_bar < self.cooldown_bars:
            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
        
        # Need 15m data for swing computation
        if df_15m is None or df_15m.empty:
            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
        
        # Compute swing levels
        swing_high, swing_low, _, _ = self._compute_swing_levels(df_15m, t)
        
        if swing_high == 0 or swing_low == 0:
            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
        
        # Update state
        self._state.last_swing_high = swing_high
        self._state.last_swing_low = swing_low
        
        current_price = features.current_price
        atr = features.atr if features.atr > 0 else 1.0
        min_breakout = self.min_atr_distance * atr
        
        # Check for breakout
        long_breakout = current_price > swing_high and (current_price - swing_high) >= min_breakout
        short_breakout = current_price < swing_low and (swing_low - current_price) >= min_breakout
        
        if long_breakout or short_breakout:
            direction = "LONG" if long_breakout else "SHORT"
            
            # Structure-based stop
            if direction == "LONG":
                stop_price = swing_low
                # Find TP at next structure above
                tp_price = self._find_next_structure_level(df_15m, t, direction, current_price)
                if tp_price <= current_price:
                    # Fallback: use 2x risk
                    risk = current_price - stop_price
                    tp_price = current_price + (2 * risk)
            else:  # SHORT
                stop_price = swing_high
                tp_price = self._find_next_structure_level(df_15m, t, direction, current_price)
                if tp_price >= current_price:
                    # Fallback: use 2x risk
                    risk = stop_price - current_price
                    tp_price = current_price - (2 * risk)
            
            self._state.last_trigger_bar = features.bar_idx
            
            return ScannerResult(
                scanner_id=self.scanner_id,
                triggered=True,
                context={
                    'direction': direction,
                    'swing_high': swing_high,
                    'swing_low': swing_low,
                    'breakout_price': current_price,
                    'stop_price': stop_price,
                    'tp_price': tp_price,
                    'entry_price': current_price,
                    'risk_points': abs(current_price - stop_price),
                    'reward_points': abs(tp_price - current_price),
                },
                score=1.0
            )
        
        return ScannerResult(scanner_id=self.scanner_id, triggered=False)

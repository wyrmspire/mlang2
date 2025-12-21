"""
ICT Inverted Fair Value Gap (IFVG) Scanner

Strategy Logic:
1. Detect FVGs on 5m timeframe
2. When a new FVG forms opposite to a recent FVG (within 30m), it's an "Inverted FVG"
3. Score the IFVG by counting swept swing levels
4. If score >= 3, place limit order at 50% of new FVG
5. Stop at FVG invalidation, TP at 2R
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import timedelta

from src.features.fvg import find_fvg, FairValueGap
from src.features.swings import find_swings, count_levels_swept, SwingPoint
from src.config import POINT_VALUE, TICK_SIZE


@dataclass
class IFVGSetup:
    """Represents a valid IFVG trade setup."""
    direction: str              # "LONG" or "SHORT"
    entry_price: float          # Limit order at FVG midpoint
    stop_price: float           # FVG invalidation level
    tp_price: float             # 2R target
    new_fvg: FairValueGap       # The triggering FVG
    old_fvg: FairValueGap       # The inverted (opposite) FVG
    liquidity_score: int        # Number of swept swing levels
    bar_idx: int
    bar_time: pd.Timestamp


@dataclass 
class IFVGState:
    """Tracks state for IFVG detection."""
    recent_fvgs: List[FairValueGap] = field(default_factory=list)
    swing_highs: List[SwingPoint] = field(default_factory=list)
    swing_lows: List[SwingPoint] = field(default_factory=list)
    last_trigger_bar: int = -1000
    max_fvg_history: int = 20


class ICTIFVGScanner:
    """
    Scanner for Inverted Fair Value Gap setups.
    
    An IFVG occurs when:
    1. A FVG forms in one direction
    2. Within 30 minutes (6 bars on 5m), a new FVG forms in the opposite direction
    3. The move creating the new FVG swept at least 3 swing levels
    
    Trade execution:
    - Entry: Limit order at 50% of new FVG
    - Stop: At FVG high (for SHORT) or low (for LONG)
    - TP: 2.0 R:R
    """
    
    def __init__(
        self,
        min_liquidity_score: int = 3,
        inversion_window_bars: int = 6,  # 30 min on 5m
        swing_lookback: int = 5,
        min_gap_atr: float = 0.2,
        risk_reward: float = 2.0,
        cooldown_bars: int = 6,
        max_risk_dollars: float = 300.0
    ):
        self.min_liquidity_score = min_liquidity_score
        self.inversion_window_bars = inversion_window_bars
        self.swing_lookback = swing_lookback
        self.min_gap_atr = min_gap_atr
        self.risk_reward = risk_reward
        self.cooldown_bars = cooldown_bars
        self.max_risk_dollars = max_risk_dollars
        self._state = IFVGState()
    
    @property
    def scanner_id(self) -> str:
        return "ict_ifvg"
    
    def reset(self):
        """Reset scanner state for new day."""
        self._state = IFVGState()
    
    def _update_swings(self, df_5m: pd.DataFrame):
        """Update swing points from 5m data."""
        highs, lows = find_swings(df_5m, lookback=self.swing_lookback)
        self._state.swing_highs = highs
        self._state.swing_lows = lows
    
    def _find_opposite_fvg(
        self, 
        new_fvg: FairValueGap,
        current_bar_idx: int
    ) -> Optional[FairValueGap]:
        """
        Find an opposite FVG within the inversion window.
        
        For a BEARISH new_fvg, we look for a recent BULLISH FVG.
        For a BULLISH new_fvg, we look for a recent BEARISH FVG.
        """
        opposite_dir = "BULLISH" if new_fvg.direction == "BEARISH" else "BEARISH"
        
        for old_fvg in reversed(self._state.recent_fvgs):
            if old_fvg.direction != opposite_dir:
                continue
            
            # Check if within window (bar index difference)
            bar_diff = current_bar_idx - old_fvg.bar_idx
            if 0 < bar_diff <= self.inversion_window_bars:
                return old_fvg
        
        return None
    
    def _calculate_liquidity_score(
        self,
        new_fvg: FairValueGap,
        df_5m: pd.DataFrame
    ) -> int:
        """
        Calculate liquidity score based on swept swing levels.
        
        For BEARISH FVG (SHORT setup): Count swing lows swept by the down move
        For BULLISH FVG (LONG setup): Count swing highs swept by the up move
        """
        if len(df_5m) < 3:
            return 0
        
        # Get the impulse bar and its preceding bar to define the move
        fvg_bar = df_5m[df_5m.index == new_fvg.bar_idx]
        if fvg_bar.empty:
            # Try to find it by time
            fvg_bar = df_5m[df_5m['time'] == new_fvg.bar_time]
        
        if fvg_bar.empty:
            return 0
        
        fvg_bar = fvg_bar.iloc[0]
        
        if new_fvg.direction == "BEARISH":
            # Down move - check swing lows that were swept
            move_low = fvg_bar['low']
            # Look back to find the recent swing high as move start
            recent_high = df_5m.tail(10)['high'].max()
            return count_levels_swept(self._state.swing_lows, recent_high, move_low)
        else:
            # Up move - check swing highs that were swept
            move_high = fvg_bar['high']
            recent_low = df_5m.tail(10)['low'].min()
            return count_levels_swept(self._state.swing_highs, recent_low, move_high)
    
    def check(
        self,
        df_5m: pd.DataFrame,
        current_bar_idx: int,
        atr: float = 5.0
    ) -> Optional[IFVGSetup]:
        """
        Check for IFVG setup at current bar.
        
        Args:
            df_5m: 5-minute OHLCV data up to current bar
            current_bar_idx: Current bar index
            atr: Current ATR for gap filtering
            
        Returns:
            IFVGSetup if valid setup found, None otherwise
        """
        # Cooldown check
        if current_bar_idx - self._state.last_trigger_bar < self.cooldown_bars:
            return None
        
        # Update swings
        self._update_swings(df_5m)
        
        # Find new FVGs at current bar
        new_fvgs = find_fvg(df_5m, lookback=3, min_gap_atr=self.min_gap_atr, atr=atr)
        
        if not new_fvgs:
            return None
        
        # Check the most recent FVG
        new_fvg = new_fvgs[0]
        
        # Already in history? Skip
        for existing in self._state.recent_fvgs:
            if existing.bar_idx == new_fvg.bar_idx and existing.direction == new_fvg.direction:
                return None
        
        # Add to history
        self._state.recent_fvgs.append(new_fvg)
        if len(self._state.recent_fvgs) > self._state.max_fvg_history:
            self._state.recent_fvgs.pop(0)
        
        # Check for inversion (opposite FVG nearby)
        old_fvg = self._find_opposite_fvg(new_fvg, current_bar_idx)
        if not old_fvg:
            return None
        
        # Calculate liquidity score
        score = self._calculate_liquidity_score(new_fvg, df_5m)
        if score < self.min_liquidity_score:
            return None
        
        # Valid setup! Calculate levels
        entry = new_fvg.midpoint
        
        if new_fvg.direction == "BEARISH":
            # SHORT setup
            direction = "SHORT"
            stop = new_fvg.high  # Invalidation at top of bearish FVG
            risk = stop - entry
            tp = entry - (risk * self.risk_reward)
        else:
            # LONG setup
            direction = "LONG"
            stop = new_fvg.low  # Invalidation at bottom of bullish FVG
            risk = entry - stop
            tp = entry + (risk * self.risk_reward)
        
        self._state.last_trigger_bar = current_bar_idx
        
        return IFVGSetup(
            direction=direction,
            entry_price=entry,
            stop_price=stop,
            tp_price=tp,
            new_fvg=new_fvg,
            old_fvg=old_fvg,
            liquidity_score=score,
            bar_idx=current_bar_idx,
            bar_time=new_fvg.bar_time
        )
    
    def get_context(self, setup: IFVGSetup) -> Dict[str, Any]:
        """Get scanner context for record output."""
        return {
            "scanner_id": self.scanner_id,
            "direction": setup.direction,
            "entry_price": setup.entry_price,
            "stop_price": setup.stop_price,
            "tp_price": setup.tp_price,
            "liquidity_score": setup.liquidity_score,
            "new_fvg": {
                "direction": setup.new_fvg.direction,
                "high": setup.new_fvg.high,
                "low": setup.new_fvg.low,
                "midpoint": setup.new_fvg.midpoint,
                "gap_size": setup.new_fvg.gap_size,
                "bar_time": setup.new_fvg.bar_time.isoformat() if setup.new_fvg.bar_time else None
            },
            "old_fvg": {
                "direction": setup.old_fvg.direction,
                "high": setup.old_fvg.high,
                "low": setup.old_fvg.low,
                "bar_time": setup.old_fvg.bar_time.isoformat() if setup.old_fvg.bar_time else None
            },
            "risk_reward": self.risk_reward,
            "min_liquidity": self.min_liquidity_score
        }

"""
ICT Fair Value Gap Scanner

Strategy Logic:
1. Wait for price to break overnight level (Asian/London high or low)
2. Look for structure change in opposite direction with impulse candle + FVG
3. Wait for price to retrace into FVG at least 50%
4. Enter in direction of structure change
5. Stop at the wick that penetrated the overnight level
6. TP at PDH/PDL or 1:3 R:R minimum

Trade Window: 9:30 AM - 11:30 AM NY
London Cutoff: 8:30 AM NY (levels must be set before this)
Risk: $300 per trade, max 1 trade at a time
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import time

from src.policy.scanners import Scanner, ScannerResult
from src.features.state import MarketState
from src.features.pipeline import FeatureBundle
from src.features.session_levels import (
    SessionLevels, 
    compute_session_levels,
    is_in_trade_window,
    is_london_complete,
    TRADE_WINDOW_START,
    TRADE_WINDOW_END
)
from src.features.fvg import (
    FairValueGap,
    find_fvg,
    find_most_recent_fvg,
    find_impulse_with_fvg
)
from src.config import POINT_VALUE, TICK_SIZE, NY_TZ


@dataclass
class ICTFVGState:
    """Tracks state for ICT FVG strategy."""
    # Level break tracking
    asian_high_broken: bool = False
    asian_low_broken: bool = False
    london_high_broken: bool = False
    london_low_broken: bool = False
    
    # Active setup tracking
    active_fvg: Optional[FairValueGap] = None
    pending_direction: Optional[str] = None
    penetrating_wick: float = 0.0
    broken_level: float = 0.0
    break_type: str = ""  # "asian_high", "asian_low", "london_high", "london_low"
    
    # Trade management
    last_trigger_bar: int = -1000
    last_trigger_date: Optional[Any] = None
    in_position: bool = False
    
    # Session levels for current day
    session_levels: Optional[SessionLevels] = None


class ICTFVGScanner(Scanner):
    """
    ICT Fair Value Gap Strategy Scanner.
    
    Entry Logic:
    1. During trade window (9:30-11:30 NY), after London cutoff (8:30)
    2. Price breaks overnight level (Asian or London H/L)
    3. Market structure changes with impulse candle creating FVG
    4. Wait for 50% retracement into FVG
    5. Enter in direction of structure change
    
    Exit Logic:
    - Stop: Wick that penetrated the overnight level + small buffer
    - TP: PDH/PDL if favorable R:R, else 1:3 R:R minimum
    
    Config:
        cooldown_bars: Min bars between triggers (default 30 = 2.5 hours on 5m)
        max_risk_dollars: Max risk per trade (default 300)
        min_rr: Minimum Risk:Reward ratio (default 1.5)
        fvg_min_pct: Min percentage into FVG for entry (default 0.5)
    """
    
    def __init__(
        self,
        cooldown_bars: int = 30,
        max_risk_dollars: float = 300.0,
        min_rr: float = 1.5,
        fvg_min_pct: float = 0.5,
        atr_buffer: float = 0.25  # Buffer for stop beyond wick
    ):
        self.cooldown_bars = cooldown_bars
        self.max_risk_dollars = max_risk_dollars
        self.min_rr = min_rr
        self.fvg_min_pct = fvg_min_pct
        self.atr_buffer = atr_buffer
        self._state = ICTFVGState()
    
    @property
    def scanner_id(self) -> str:
        return "ict_fvg_5m"
    
    def reset(self):
        """Reset state for new day or simulation run."""
        self._state = ICTFVGState()
    
    def _calculate_position_size(self, entry: float, stop: float) -> tuple:
        """Calculate max contracts for given risk."""
        dist_points = abs(entry - stop)
        if dist_points < TICK_SIZE:
            return 0, 0.0
        
        risk_per_contract = dist_points * POINT_VALUE
        if risk_per_contract <= 0:
            return 0, 0.0
        
        contracts = int(self.max_risk_dollars // risk_per_contract)
        contracts = max(1, contracts)
        
        contract_risk = contracts * risk_per_contract
        if contract_risk > self.max_risk_dollars * 1.1:
            contracts = max(1, contracts - 1)
            contract_risk = contracts * risk_per_contract
        
        return contracts, contract_risk
    
    def _check_level_break(
        self,
        current_price: float,
        current_high: float,
        current_low: float,
        session_levels: SessionLevels
    ) -> Optional[Dict[str, Any]]:
        """
        Check if price has broken an overnight level.
        
        Returns dict with break info or None.
        """
        # Check Asian high break (SHORT setup potential)
        if session_levels.asian_high > 0 and not self._state.asian_high_broken:
            if current_high > session_levels.asian_high:
                self._state.asian_high_broken = True
                return {
                    'level_type': 'asian_high',
                    'level_price': session_levels.asian_high,
                    'break_direction': 'UP',
                    'setup_direction': 'SHORT',  # Trade opposite after structure change
                    'penetrating_wick': current_high
                }
        
        # Check Asian low break (LONG setup potential)
        if session_levels.asian_low > 0 and not self._state.asian_low_broken:
            if current_low < session_levels.asian_low:
                self._state.asian_low_broken = True
                return {
                    'level_type': 'asian_low',
                    'level_price': session_levels.asian_low,
                    'break_direction': 'DOWN',
                    'setup_direction': 'LONG',
                    'penetrating_wick': current_low
                }
        
        # Check London high break (SHORT setup potential)
        if session_levels.london_high > 0 and not self._state.london_high_broken:
            if current_high > session_levels.london_high:
                self._state.london_high_broken = True
                return {
                    'level_type': 'london_high',
                    'level_price': session_levels.london_high,
                    'break_direction': 'UP',
                    'setup_direction': 'SHORT',
                    'penetrating_wick': current_high
                }
        
        # Check London low break (LONG setup potential)
        if session_levels.london_low > 0 and not self._state.london_low_broken:
            if current_low < session_levels.london_low:
                self._state.london_low_broken = True
                return {
                    'level_type': 'london_low',
                    'level_price': session_levels.london_low,
                    'break_direction': 'DOWN',
                    'setup_direction': 'LONG',
                    'penetrating_wick': current_low
                }
        
        return None
    
    def _find_structure_change_fvg(
        self,
        df_5m: pd.DataFrame,
        expected_direction: str,
        atr: float
    ) -> Optional[FairValueGap]:
        """
        Look for a structure change with impulse candle creating FVG.
        
        After a level break UP, we expect a structure change DOWN (bearish FVG).
        After a level break DOWN, we expect a structure change UP (bullish FVG).
        """
        result = find_impulse_with_fvg(df_5m, expected_direction, lookback=10, atr=atr)
        if result:
            _, fvg = result
            return fvg
        return None
    
    def scan(
        self,
        state: MarketState,
        features: FeatureBundle,
        df_5m: pd.DataFrame = None,
        df_1m: pd.DataFrame = None
    ) -> ScannerResult:
        """Check for ICT FVG setup."""
        t = features.timestamp
        if t is None:
            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
        
        # 1. Check if we're in the trade window (9:30 - 11:30 NY)
        if not is_in_trade_window(t, NY_TZ):
            # Reset state at end of day
            current_date = t.astimezone(NY_TZ).date()
            if self._state.last_trigger_date != current_date:
                self.reset()
                self._state.last_trigger_date = current_date
            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
        
        # 2. Check if London session is complete (past 8:30)
        if not is_london_complete(t, NY_TZ):
            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
        
        # 3. Cooldown check
        if features.bar_idx - self._state.last_trigger_bar < self.cooldown_bars:
            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
        
        # 4. Check if in position (only one trade at a time)
        if self._state.in_position:
            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
        
        # 5. Need data
        if df_5m is None or df_5m.empty or df_1m is None or df_1m.empty:
            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
        
        current_price = features.current_price
        atr = features.atr if features.atr > 0 else 5.0
        
        # 6. Compute session levels if not already done today
        self._state.session_levels = compute_session_levels(df_1m, t, NY_TZ)
        session_levels = self._state.session_levels
        
        if session_levels is None:
            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
        
        # Get current bar info
        current_bar = df_5m.iloc[-1] if len(df_5m) > 0 else None
        if current_bar is None:
            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
        
        current_high = current_bar['high']
        current_low = current_bar['low']
        
        # 7. Check for level break if we don't have an active setup
        if self._state.active_fvg is None:
            level_break = self._check_level_break(
                current_price, current_high, current_low, session_levels
            )
            
            if level_break:
                # Level just broke - store info and look for structure change
                self._state.pending_direction = level_break['setup_direction']
                self._state.penetrating_wick = level_break['penetrating_wick']
                self._state.broken_level = level_break['level_price']
                self._state.break_type = level_break['level_type']
                
                # Look for FVG indicating structure change
                fvg = self._find_structure_change_fvg(
                    df_5m, level_break['setup_direction'], atr
                )
                
                if fvg:
                    self._state.active_fvg = fvg
        
        # 8. If no FVG yet but we have a pending direction, keep looking
        if self._state.active_fvg is None and self._state.pending_direction:
            fvg = self._find_structure_change_fvg(
                df_5m, self._state.pending_direction, atr
            )
            if fvg:
                self._state.active_fvg = fvg
        
        # 9. Check for FVG retracement entry
        if self._state.active_fvg is not None:
            fvg = self._state.active_fvg
            
            # Check if price has retraced at least 50% into FVG
            if fvg.contains_price(current_price, self.fvg_min_pct):
                direction = self._state.pending_direction
                
                if direction is None:
                    # Fallback based on FVG direction
                    direction = "LONG" if fvg.direction == "BULLISH" else "SHORT"
                
                # Calculate stop and TP
                if direction == "LONG":
                    # Stop below penetrating wick
                    stop_price = self._state.penetrating_wick - (self.atr_buffer * atr)
                    risk = current_price - stop_price
                    
                    if risk <= 0:
                        return ScannerResult(scanner_id=self.scanner_id, triggered=False)
                    
                    # TP: Use PDH if favorable, otherwise 1:3 RR minimum
                    pdh = features.levels.pdh if features.levels else 0
                    tp_at_rr = current_price + (self.min_rr * risk)
                    
                    if pdh > 0 and (pdh - current_price) >= (self.min_rr * risk):
                        tp_price = pdh
                    else:
                        tp_price = tp_at_rr
                        
                else:  # SHORT
                    # Stop above penetrating wick
                    stop_price = self._state.penetrating_wick + (self.atr_buffer * atr)
                    risk = stop_price - current_price
                    
                    if risk <= 0:
                        return ScannerResult(scanner_id=self.scanner_id, triggered=False)
                    
                    # TP: Use PDL if favorable, otherwise 1:3 RR minimum
                    pdl = features.levels.pdl if features.levels else 0
                    tp_at_rr = current_price - (self.min_rr * risk)
                    
                    if pdl > 0 and (current_price - pdl) >= (self.min_rr * risk):
                        tp_price = pdl
                    else:
                        tp_price = tp_at_rr
                
                # Calculate position size
                contracts, actual_risk = self._calculate_position_size(current_price, stop_price)
                if contracts == 0:
                    return ScannerResult(scanner_id=self.scanner_id, triggered=False)
                
                reward = abs(tp_price - current_price)
                rr_ratio = reward / risk if risk > 0 else 0
                
                # Update state
                self._state.last_trigger_bar = features.bar_idx
                self._state.in_position = True
                
                # Clear active setup
                active_fvg = self._state.active_fvg
                self._state.active_fvg = None
                self._state.pending_direction = None
                
                return ScannerResult(
                    scanner_id=self.scanner_id,
                    triggered=True,
                    context={
                        'direction': direction,
                        'entry_price': current_price,
                        'stop_price': stop_price,
                        'tp_price': tp_price,
                        'risk_points': risk,
                        'reward_points': reward,
                        'rr_ratio': rr_ratio,
                        'contracts': contracts,
                        'risk_dollars': actual_risk,
                        'level_broken': self._state.break_type,
                        'broken_level_price': self._state.broken_level,
                        'penetrating_wick': self._state.penetrating_wick,
                        'fvg_high': active_fvg.high if active_fvg else 0,
                        'fvg_low': active_fvg.low if active_fvg else 0,
                        'fvg_midpoint': active_fvg.midpoint if active_fvg else 0,
                        'fvg_direction': active_fvg.direction if active_fvg else "",
                        'asian_high': session_levels.asian_high,
                        'asian_low': session_levels.asian_low,
                        'london_high': session_levels.london_high,
                        'london_low': session_levels.london_low,
                    },
                    score=1.0
                )
        
        return ScannerResult(scanner_id=self.scanner_id, triggered=False)

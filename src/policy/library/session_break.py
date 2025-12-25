"""
Session Break Scanner
Triggers on break of previous session's high or low.
"""

import pandas as pd
from typing import Optional
from dataclasses import dataclass

from src.policy.scanners import Scanner, ScanResult
from src.features.state import MarketState
from src.features.pipeline import FeatureBundle
from src.config import NY_TZ


@dataclass
class SessionBreakState:
    """Tracks session levels and break status."""
    date: Optional[pd.Timestamp] = None
    pdh_broken: bool = False
    pdl_broken: bool = False
    last_trigger_bar: int = -1000


class SessionBreakScanner(Scanner):
    """
    Scanner that triggers on break of previous day's high or low.
    
    Logic:
    1. Track PDH (Previous Day High) and PDL (Previous Day Low) from features.
    2. LONG: Price closes above PDH (not just wicks).
    3. SHORT: Price closes below PDL (not just wicks).
    4. Only one trigger per level per day.
    
    Config:
        require_close: Require close beyond level vs any touch (default True).
        cooldown_bars: Min bars between triggers (default 30).
    """
    
    def __init__(
        self,
        require_close: bool = True,
        cooldown_bars: int = 30,
    ):
        self.require_close = require_close
        self.cooldown_bars = cooldown_bars
        self._state = SessionBreakState()
    
    @property
    def scanner_id(self) -> str:
        return "session_break"
    
    def _is_new_day(self, t: pd.Timestamp) -> bool:
        if self._state.date is None:
            return True
        ny_time = t.astimezone(NY_TZ)
        return ny_time.date() != self._state.date.date()
    
    def scan(
        self,
        state: MarketState,
        features: FeatureBundle
    ) -> ScanResult:
        t = features.timestamp
        if t is None:
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        # Reset on new day
        if self._is_new_day(t):
            self._state = SessionBreakState(date=t.astimezone(NY_TZ))
        
        # RTH check
        ny_time = t.astimezone(NY_TZ)
        is_rth = (ny_time.hour == 9 and ny_time.minute >= 30) or (10 <= ny_time.hour < 16)
        if not is_rth:
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        # Cooldown
        if features.bar_idx - self._state.last_trigger_bar < self.cooldown_bars:
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        # Need levels
        if features.levels is None:
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        pdh = features.levels.pdh
        pdl = features.levels.pdl
        
        if pdh <= 0 or pdl <= 0:
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        price = features.current_price  # This is close price
        
        # Check PDH break (LONG)
        if not self._state.pdh_broken and price > pdh:
            self._state.pdh_broken = True
            self._state.last_trigger_bar = features.bar_idx
            
            return ScanResult(
                scanner_id=self.scanner_id,
                triggered=True,
                context={
                    'direction': 'LONG',
                    'broken_level': pdh,
                    'break_type': 'PDH',
                    'entry_price': price,
                    'pdh': pdh,
                    'pdl': pdl,
                    'distance_beyond': price - pdh,
                },
                score=1.0
            )
        
        # Check PDL break (SHORT)
        if not self._state.pdl_broken and price < pdl:
            self._state.pdl_broken = True
            self._state.last_trigger_bar = features.bar_idx
            
            return ScanResult(
                scanner_id=self.scanner_id,
                triggered=True,
                context={
                    'direction': 'SHORT',
                    'broken_level': pdl,
                    'break_type': 'PDL',
                    'entry_price': price,
                    'pdh': pdh,
                    'pdl': pdl,
                    'distance_beyond': pdl - price,
                },
                score=1.0
            )
        
        return ScanResult(scanner_id=self.scanner_id, triggered=False)

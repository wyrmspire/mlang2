"""
VWAP Bounce Scanner
Triggers when price crosses and bounces off VWAP.
"""

import pandas as pd
from typing import Optional
from dataclasses import dataclass

from src.policy.scanners import Scanner, ScanResult
from src.features.state import MarketState
from src.features.pipeline import FeatureBundle
from src.config import NY_TZ


@dataclass
class VWAPBounceState:
    """Tracks VWAP interaction state."""
    was_above_vwap: Optional[bool] = None
    last_trigger_bar: int = -1000


class VWAPBounceScanner(Scanner):
    """
    Scanner that triggers on VWAP bounce setups.
    
    Logic:
    1. Track if price is above or below VWAP.
    2. LONG: Price dips below VWAP, then closes back above.
    3. SHORT: Price spikes above VWAP, then closes back below.
    4. RTH only (9:30 AM - 4:00 PM NY).
    
    Config:
        cooldown_bars: Minimum bars between triggers (default 15).
        min_penetration_atr: Min ATR penetration into VWAP zone (default 0.1).
    """
    
    def __init__(
        self,
        cooldown_bars: int = 15,
        min_penetration_atr: float = 0.1,
    ):
        self.cooldown_bars = cooldown_bars
        self.min_penetration_atr = min_penetration_atr
        self._state = VWAPBounceState()
    
    @property
    def scanner_id(self) -> str:
        return f"vwap_bounce_{self.cooldown_bars}"
    
    def scan(
        self,
        state: MarketState,
        features: FeatureBundle
    ) -> ScanResult:
        t = features.timestamp
        if t is None:
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        # RTH check (9:30 AM - 4:00 PM NY)
        ny_time = t.astimezone(NY_TZ)
        if not (9 <= ny_time.hour < 16 or (ny_time.hour == 9 and ny_time.minute >= 30)):
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        # Need indicators for VWAP
        if features.indicators is None:
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        vwap = features.indicators.vwap_session
        if vwap <= 0:
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        # Cooldown check
        if features.bar_idx - self._state.last_trigger_bar < self.cooldown_bars:
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        price = features.current_price
        atr = features.atr if features.atr > 0 else 1.0
        
        currently_above = price > vwap
        
        # Check for bounce condition
        triggered = False
        direction = "neutral"
        
        if self._state.was_above_vwap is not None:
            # Was above, now below, but closing above = LONG bounce
            if self._state.was_above_vwap and currently_above:
                # Check if we dipped below during this bar
                # Using price proximity as proxy
                dist_to_vwap = abs(price - vwap) / atr
                if dist_to_vwap < self.min_penetration_atr * 2:
                    # Close to VWAP, potential bounce
                    pass
            
            # Transition: was below, now above = LONG
            if not self._state.was_above_vwap and currently_above:
                triggered = True
                direction = "LONG"
            
            # Transition: was above, now below = SHORT
            elif self._state.was_above_vwap and not currently_above:
                triggered = True
                direction = "SHORT"
        
        # Update state
        self._state.was_above_vwap = currently_above
        
        if triggered:
            self._state.last_trigger_bar = features.bar_idx
            return ScanResult(
                scanner_id=self.scanner_id,
                triggered=True,
                context={
                    'direction': direction,
                    'vwap_level': vwap,
                    'entry_price': price,
                    'distance_atr': abs(price - vwap) / atr,
                },
                score=1.0
            )
        
        return ScanResult(scanner_id=self.scanner_id, triggered=False)

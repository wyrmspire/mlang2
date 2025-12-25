"""
First Pullback Scanner
After opening drive, triggers on first pullback to EMA.
"""

import pandas as pd
from typing import Optional
from dataclasses import dataclass

from src.policy.scanners import Scanner, ScanResult
from src.features.state import MarketState
from src.features.pipeline import FeatureBundle
from src.config import NY_TZ


@dataclass
class FirstPullbackState:
    """Tracks opening bias and pullback state."""
    date: Optional[pd.Timestamp] = None
    opening_price: float = 0.0
    opening_bias: Optional[str] = None  # 'BULLISH' or 'BEARISH'
    bias_established: bool = False
    pullback_triggered: bool = False


class FirstPullbackScanner(Scanner):
    """
    Scanner that triggers on first pullback to EMA after opening drive.
    
    Logic:
    1. At 10:00 AM NY, establish opening bias based on price vs 9:30 open.
    2. LONG: Bullish bias (price > open) and price pulls back to EMA20.
    3. SHORT: Bearish bias (price < open) and price pulls back to EMA20.
    4. Only one trigger per day.
    
    Config:
        bias_threshold_atr: Min move to establish bias (default 0.5 ATR).
        ema_threshold_atr: How close to EMA for pullback (default 0.3 ATR).
    """
    
    def __init__(
        self,
        bias_threshold_atr: float = 0.5,
        ema_threshold_atr: float = 0.3,
    ):
        self.bias_threshold_atr = bias_threshold_atr
        self.ema_threshold_atr = ema_threshold_atr
        self._state = FirstPullbackState()
    
    @property
    def scanner_id(self) -> str:
        return "first_pullback"
    
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
        
        ny_time = t.astimezone(NY_TZ)
        
        # Reset on new day
        if self._is_new_day(t):
            self._state = FirstPullbackState(date=t.astimezone(NY_TZ))
        
        # Capture opening price at 9:30-9:31
        if ny_time.hour == 9 and 30 <= ny_time.minute <= 31 and self._state.opening_price == 0:
            self._state.opening_price = features.current_price
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        # Establish bias at 10:00
        if ny_time.hour == 10 and ny_time.minute == 0 and not self._state.bias_established:
            if self._state.opening_price > 0:
                price = features.current_price
                atr = features.atr if features.atr > 0 else 1.0
                move = (price - self._state.opening_price) / atr
                
                if move > self.bias_threshold_atr:
                    self._state.opening_bias = "BULLISH"
                    self._state.bias_established = True
                elif move < -self.bias_threshold_atr:
                    self._state.opening_bias = "BEARISH"
                    self._state.bias_established = True
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        # Only look for pullback between 10:00 and 11:30
        if not (10 <= ny_time.hour < 12 or (ny_time.hour == 11 and ny_time.minute <= 30)):
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        # Already triggered today
        if self._state.pullback_triggered:
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        # Need bias
        if not self._state.bias_established or self._state.opening_bias is None:
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        # Need indicators
        if features.indicators is None:
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        price = features.current_price
        ema = features.indicators.ema_5m_20
        atr = features.atr if features.atr > 0 else 1.0
        
        if ema <= 0:
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        dist_to_ema = abs(price - ema) / atr
        
        # Check pullback condition
        if dist_to_ema <= self.ema_threshold_atr:
            direction = "LONG" if self._state.opening_bias == "BULLISH" else "SHORT"
            
            # Confirm pullback direction matches bias
            # Bullish: price should be pulling back down toward EMA (price near or below EMA OK)
            # Bearish: price should be pulling back up toward EMA
            valid_pullback = (
                (self._state.opening_bias == "BULLISH" and price <= ema + (self.ema_threshold_atr * atr)) or
                (self._state.opening_bias == "BEARISH" and price >= ema - (self.ema_threshold_atr * atr))
            )
            
            if valid_pullback:
                self._state.pullback_triggered = True
                return ScanResult(
                    scanner_id=self.scanner_id,
                    triggered=True,
                    context={
                        'direction': direction,
                        'opening_bias': self._state.opening_bias,
                        'opening_price': self._state.opening_price,
                        'entry_price': price,
                        'ema_level': ema,
                        'pullback_depth_atr': dist_to_ema,
                    },
                    score=1.0
                )
        
        return ScanResult(scanner_id=self.scanner_id, triggered=False)

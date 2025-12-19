"""
Simple Time Scanner
Triggers trades at a specific time of day based on momentum.
"""

import pandas as pd
from typing import Optional, Dict, Any
from dataclasses import dataclass
from zoneinfo import ZoneInfo

from src.policy.scanners import Scanner, ScannerResult
from src.features.state import MarketState
from src.features.pipeline import FeatureBundle
from src.config import NY_TZ


@dataclass
class SimpleTimeState:
    """Tracks the last trigger date to ensure one trade per day."""
    last_trigger_date: Optional[pd.Timestamp] = None


class SimpleTimeScanner(Scanner):
    """
    Scanner that triggers at a specific time daily.
    
    Logic:
    1. Wait for specific time (e.g. 10:00 NY).
    2. Check price change over last 'momentum_minutes'.
    3. If positive -> LONG, negative -> SHORT.
    """
    
    def __init__(
        self,
        hour: int = 10,
        minute: int = 0,
        momentum_minutes: int = 15,
    ):
        self.hour = hour
        self.minute = minute
        self.momentum_minutes = momentum_minutes
        self._state = SimpleTimeState()
    
    @property
    def scanner_id(self) -> str:
        return f"simple_time_{self.hour:02d}{self.minute:02d}"
    
    def scan(
        self,
        state: MarketState,
        features: FeatureBundle
    ) -> ScannerResult:
        t = features.timestamp
        if t is None:
            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
        
        ny_time = t.astimezone(NY_TZ)
        
        # Check if it's the target time
        if ny_time.hour == self.hour and ny_time.minute == self.minute:
            # Check if we already traded today
            if self._state.last_trigger_date == ny_time.date():
                return ScannerResult(scanner_id=self.scanner_id, triggered=False)
            
            # Helper to get price N minutes ago
            # features.x_price_1m is a numpy array of recent close prices
            # The last element is current price (-1).
            # We want price 'momentum_minutes' ago.
            prices = features.x_price_1m
            if hasattr(prices, 'flatten'):
                prices = prices.flatten()
            
            if prices is None or len(prices) < self.momentum_minutes:
                # Not enough data
                return ScannerResult(scanner_id=self.scanner_id, triggered=False)
            
            current_price = float(features.current_price)
            past_price = float(prices[-(self.momentum_minutes + 1)]) # Approximate
            
            direction = "LONG" if current_price > past_price else "SHORT"
            
            self._state.last_trigger_date = ny_time.date()
            
            return ScannerResult(
                scanner_id=self.scanner_id,
                triggered=True,
                context={
                    'direction': direction,
                    'entry_price': current_price,
                    'past_price': past_price,
                    'momentum_minutes': self.momentum_minutes
                },
                score=1.0
            )
            
        return ScannerResult(scanner_id=self.scanner_id, triggered=False)

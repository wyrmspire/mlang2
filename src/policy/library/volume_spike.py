"""
Volume Spike Scanner
Triggers on unusual volume with price confirmation.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass, field

from src.policy.scanners import Scanner, ScanResult
from src.features.state import MarketState
from src.features.pipeline import FeatureBundle
from src.config import NY_TZ


@dataclass
class VolumeSpikeState:
    """Tracks volume history for spike detection."""
    volume_history: list = field(default_factory=list)
    last_trigger_bar: int = -1000


class VolumeSpikeScanner(Scanner):
    """
    Scanner that triggers on high-volume breakout bars.
    
    Logic:
    1. Calculate 20-bar volume average.
    2. Trigger when current volume > 2x average.
    3. Direction based on close vs open of spike bar.
    4. RTH only.
    
    Config:
        volume_multiple: How many times average for spike (default 2.0).
        lookback: Bars for volume average (default 20).
        cooldown_bars: Min bars between triggers (default 10).
    """
    
    def __init__(
        self,
        volume_multiple: float = 2.0,
        lookback: int = 20,
        cooldown_bars: int = 10,
    ):
        self.volume_multiple = volume_multiple
        self.lookback = lookback
        self.cooldown_bars = cooldown_bars
        self._state = VolumeSpikeState()
    
    @property
    def scanner_id(self) -> str:
        return f"volume_spike_{self.volume_multiple}x"
    
    def scan(
        self,
        state: MarketState,
        features: FeatureBundle
    ) -> ScanResult:
        t = features.timestamp
        if t is None:
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        # RTH check
        ny_time = t.astimezone(NY_TZ)
        is_rth = (ny_time.hour == 9 and ny_time.minute >= 30) or (10 <= ny_time.hour < 16)
        if not is_rth:
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        # Cooldown
        if features.bar_idx - self._state.last_trigger_bar < self.cooldown_bars:
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        # Get current bar volume from market state
        current_volume = 0.0
        bar_open = features.current_price
        bar_close = features.current_price
        
        market_state = features.market_state
        if market_state is not None and market_state.ohlcv_1m is not None and len(market_state.ohlcv_1m) > 0:
            current_bar = market_state.ohlcv_1m[-1]
            if len(current_bar) >= 5:
                bar_open = float(current_bar[0])
                bar_close = float(current_bar[3])
                current_volume = float(current_bar[4])
        
        if current_volume <= 0:
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        # Update volume history
        self._state.volume_history.append(current_volume)
        if len(self._state.volume_history) > self.lookback:
            self._state.volume_history = self._state.volume_history[-self.lookback:]
        
        # Need enough history
        if len(self._state.volume_history) < self.lookback:
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        # Calculate average (excluding current bar)
        avg_volume = np.mean(self._state.volume_history[:-1])
        
        if avg_volume <= 0:
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        volume_ratio = current_volume / avg_volume
        
        # Check for spike
        if volume_ratio >= self.volume_multiple:
            direction = "LONG" if bar_close > bar_open else "SHORT"
            bar_range = abs(bar_close - bar_open)
            
            self._state.last_trigger_bar = features.bar_idx
            
            return ScanResult(
                scanner_id=self.scanner_id,
                triggered=True,
                context={
                    'direction': direction,
                    'volume_ratio': volume_ratio,
                    'current_volume': current_volume,
                    'avg_volume': avg_volume,
                    'spike_bar_range': bar_range,
                    'entry_price': features.current_price,
                },
                score=min(volume_ratio / self.volume_multiple, 2.0)  # Cap score at 2x threshold
            )
        
        return ScanResult(scanner_id=self.scanner_id, triggered=False)

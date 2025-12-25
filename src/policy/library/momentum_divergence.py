"""
Momentum Divergence Scanner
Triggers on RSI-price divergence patterns.
"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass, field

from src.policy.scanners import Scanner, ScanResult
from src.features.state import MarketState
from src.features.pipeline import FeatureBundle
from src.config import NY_TZ


@dataclass
class DivergenceState:
    """Tracks price and RSI history for divergence detection."""
    price_history: list = field(default_factory=list)
    rsi_history: list = field(default_factory=list)
    last_trigger_bar: int = -1000


class MomentumDivergenceScanner(Scanner):
    """
    Scanner that triggers on RSI-price divergence.
    
    Logic:
    1. Track last N bars of price lows/highs and RSI.
    2. Bullish Divergence: Price makes lower low, RSI makes higher low -> LONG.
    3. Bearish Divergence: Price makes higher high, RSI makes lower high -> SHORT.
    
    Config:
        lookback: Bars for swing detection (default 20).
        rsi_threshold: Min RSI difference for divergence (default 5.0).
        cooldown_bars: Min bars between triggers (default 20).
    """
    
    def __init__(
        self,
        lookback: int = 20,
        rsi_threshold: float = 5.0,
        cooldown_bars: int = 20,
    ):
        self.lookback = lookback
        self.rsi_threshold = rsi_threshold
        self.cooldown_bars = cooldown_bars
        self._state = DivergenceState()
    
    @property
    def scanner_id(self) -> str:
        return f"momentum_divergence_{self.lookback}"
    
    def _find_swing_lows(self, prices: List[float], window: int = 5) -> List[Tuple[int, float]]:
        """Find swing lows in price series."""
        swings = []
        for i in range(window, len(prices) - window):
            is_low = all(prices[i] <= prices[i-j] for j in range(1, window+1))
            is_low = is_low and all(prices[i] <= prices[i+j] for j in range(1, window+1))
            if is_low:
                swings.append((i, prices[i]))
        return swings
    
    def _find_swing_highs(self, prices: List[float], window: int = 5) -> List[Tuple[int, float]]:
        """Find swing highs in price series."""
        swings = []
        for i in range(window, len(prices) - window):
            is_high = all(prices[i] >= prices[i-j] for j in range(1, window+1))
            is_high = is_high and all(prices[i] >= prices[i+j] for j in range(1, window+1))
            if is_high:
                swings.append((i, prices[i]))
        return swings
    
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
        
        # Need indicators
        if features.indicators is None:
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        price = features.current_price
        rsi = features.indicators.rsi_5m_14
        
        # Update history
        self._state.price_history.append(price)
        self._state.rsi_history.append(rsi)
        
        if len(self._state.price_history) > self.lookback:
            self._state.price_history = self._state.price_history[-self.lookback:]
            self._state.rsi_history = self._state.rsi_history[-self.lookback:]
        
        # Need enough history
        if len(self._state.price_history) < self.lookback:
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        prices = self._state.price_history
        rsis = self._state.rsi_history
        
        # Find swing lows for bullish divergence
        price_lows = self._find_swing_lows(prices, window=3)
        rsi_lows = self._find_swing_lows(rsis, window=3)
        
        # Check bullish divergence (price lower low, RSI higher low)
        if len(price_lows) >= 2 and len(rsi_lows) >= 2:
            recent_price_low = price_lows[-1]
            prev_price_low = price_lows[-2]
            recent_rsi_low = rsis[recent_price_low[0]] if recent_price_low[0] < len(rsis) else 0
            prev_rsi_low = rsis[prev_price_low[0]] if prev_price_low[0] < len(rsis) else 0
            
            # Price: lower low, RSI: higher low
            if (recent_price_low[1] < prev_price_low[1] and 
                recent_rsi_low > prev_rsi_low + self.rsi_threshold):
                
                self._state.last_trigger_bar = features.bar_idx
                return ScanResult(
                    scanner_id=self.scanner_id,
                    triggered=True,
                    context={
                        'direction': 'LONG',
                        'divergence_type': 'bullish',
                        'rsi_current': rsi,
                        'price_swing_low': recent_price_low[1],
                        'price_prev_low': prev_price_low[1],
                        'rsi_at_low': recent_rsi_low,
                        'rsi_prev_low': prev_rsi_low,
                        'entry_price': price,
                    },
                    score=1.0
                )
        
        # Find swing highs for bearish divergence
        price_highs = self._find_swing_highs(prices, window=3)
        rsi_highs = self._find_swing_highs(rsis, window=3)
        
        # Check bearish divergence (price higher high, RSI lower high)
        if len(price_highs) >= 2 and len(rsi_highs) >= 2:
            recent_price_high = price_highs[-1]
            prev_price_high = price_highs[-2]
            recent_rsi_high = rsis[recent_price_high[0]] if recent_price_high[0] < len(rsis) else 0
            prev_rsi_high = rsis[prev_price_high[0]] if prev_price_high[0] < len(rsis) else 0
            
            # Price: higher high, RSI: lower high
            if (recent_price_high[1] > prev_price_high[1] and 
                recent_rsi_high < prev_rsi_high - self.rsi_threshold):
                
                self._state.last_trigger_bar = features.bar_idx
                return ScanResult(
                    scanner_id=self.scanner_id,
                    triggered=True,
                    context={
                        'direction': 'SHORT',
                        'divergence_type': 'bearish',
                        'rsi_current': rsi,
                        'price_swing_high': recent_price_high[1],
                        'price_prev_high': prev_price_high[1],
                        'rsi_at_high': recent_rsi_high,
                        'rsi_prev_high': prev_rsi_high,
                        'entry_price': price,
                    },
                    score=1.0
                )
        
        return ScanResult(scanner_id=self.scanner_id, triggered=False)

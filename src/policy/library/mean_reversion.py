"""
Mean Reversion Scanner
Triggers when price extends beyond Keltner Channels (EMA +/- ATR bands).
"""

from typing import Dict, Any
from src.policy.scanners import Scanner, ScanResult
from src.features.state import MarketState
from src.features.pipeline import FeatureBundle

class MeanReversionScanner(Scanner):
    """
    Triggers when price is outside EMA +/- N * ATR bands.
    Suggests reversion to the mean (EMA).
    """
    
    def __init__(
        self,
        ema_period: int = 20,
        atr_multiple: float = 3.0,
        rsi_min: float = 30.0,
        rsi_max: float = 70.0,
        timeframe: str = '5m'
    ):
        self.ema_period = ema_period
        self.atr_multiple = atr_multiple
        self.rsi_min = rsi_min
        self.rsi_max = rsi_max
        self.timeframe = timeframe
        
        # State
        self._was_triggered = False
    
    @property
    def scanner_id(self) -> str:
        return f"mean_reversion_{self.ema_period}_{self.atr_multiple}_{self.timeframe}_{int(self.rsi_min)}_{int(self.rsi_max)}"
    
    def scan(
        self,
        state: MarketState,
        features: FeatureBundle
    ) -> ScanResult:
        if features.indicators is None:
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        ind = features.indicators
        
        if self.timeframe == '5m':
            ema = ind.ema_5m_20
            atr = ind.atr_5m_14
            rsi = ind.rsi_5m_14
        elif self.timeframe == '15m':
            ema = ind.ema_15m_20
            atr = ind.atr_15m_14
            rsi = ind.rsi_15m_14
        else:
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
            
        if ema == 0 or atr == 0:
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
            
        current_price = features.current_price
        
        upper_band = ema + (atr * self.atr_multiple)
        lower_band = ema - (atr * self.atr_multiple)
        
        # Conditions
        short_signal = (current_price > upper_band) and (rsi > self.rsi_max)
        long_signal = (current_price < lower_band) and (rsi < self.rsi_min)
        
        is_signal_active = short_signal or long_signal
        
        triggered = False
        signal = "neutral"
        distance = 0.0
        
        if is_signal_active:
            # Check debounce state
            if not self._was_triggered:
                triggered = True
                self._was_triggered = True
                
                if short_signal:
                    signal = "short"
                    distance = current_price - upper_band
                else:
                    signal = "long"
                    distance = lower_band - current_price
            else:
                # Already triggered, waiting for reset
                triggered = False
        else:
            # Reset state when condition is lost
            self._was_triggered = False
            
        return ScanResult(
            scanner_id=self.scanner_id,
            triggered=triggered,
            context={
                'signal': signal,
                'ema': ema,
                'upper_band': upper_band,
                'lower_band': lower_band,
                'distance': distance,
                'atr': atr,
                'rsi': rsi
            },
            score=1.0 if triggered else 0.0
        )

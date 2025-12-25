"""
Puller Strategy
Scanner that looks for a specific "Measured Move" failure pattern.
"""

import numpy as np
from src.policy.scanners import Scanner, ScanResult
from src.features.state import MarketState
from src.features.pipeline import FeatureBundle
from typing import Dict, Any, Optional

class PullerScanner(Scanner):
    """
    "Puller" Strategy Scanner.
    
    Pattern:
    1. Identify a "Start" price (Start).
    2. Price rises to a "Peak" within `max_duration` minutes.
    3. The Move (Start -> Peak) is >= 1.5 * Unit (Proportional size).
    4. The Move is < 2.5 * Unit (Invalidation).
    5. Price retraces and CLOSES below Start.
    6. Action: Limit Sell at Start + 0.75 * Unit.
    
    Unit: Default is ATR(14) from 5m timeframe (or passed in).
    "Proportions not price" => We use volatility units (ATR).
    """

    def __init__(
        self,
        min_move_unit: float = 1.5,
        max_move_unit: float = 2.5,
        entry_unit: float = 0.75,
        stop_unit: float = 2.0,
        tp_unit: float = -4.0,
        max_duration_bars: int = 45, # 45 mins on 1m chart
        variation_id: str = "v1"
    ):
        self.min_move = min_move_unit
        self.max_move = max_move_unit
        self.entry_level = entry_unit
        self.stop_level = stop_unit
        self.tp_level = tp_unit
        self.max_duration = max_duration_bars
        self.var_id = variation_id

    @property
    def scanner_id(self) -> str:
        return f"puller_{self.var_id}"

    def scan(self, state: MarketState, features: FeatureBundle) -> ScanResult:
        # We need history and indicators
        if features.indicators is None:
            return ScanResult(scanner_id=self.scanner_id, triggered=False)

        # Use ATR as the "Unit" (Scale)
        # Assuming features.indicators has atr_5m_14 or similar.
        # Use simple 5m ATR for stable scale.
        atr = features.indicators.atr_5m_14
        if atr is None or atr <= 0:
             return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        unit = atr
        
        # We scan on 1m bars for precision "within 45 mins"
        # state.ohlcv_1m is (lookback, 5) -> [open, high, low, close, volume]
        history = state.ohlcv_1m
        if len(history) < self.max_duration + 1:
             return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        current_close = state.current_price
        
        # Lookback Logic:
        # Iterate backwards to find a valid "Start" point.
        # A valid Start point must be:
        # 1. Start Price > Current Close (since we just broke below it)
        # 2. Between Start and Now, Max High reached >= Start + 1.5*Unit
        # 3. Between Start and Now, Max High < Start + 2.5*Unit
        
        # We search from i=1 to max_duration
        # i represents "bars ago" for the Start Candidate.
        # history[-1] is current bar (but usually current_price is the *forming* bar or just closed? 
        # State usually gives `current_price` separate from `ohlcv_1m` history which might be closed bars.
        # Let's assume history[-1] is the previous closed bar if we are live, 
        # or the current bar if we are backtesting on close.
        # Usually ohlcv_1m includes the current bar as the last element if we are at 'close'.
        # Let's align: history[-1] is the triggering bar.
        
        triggered = False
        context: Dict[str, Any] = {}
        
        # Search for the *most recent* valid signal? Or just the first one we find?
        # Usually we want the smallest valid pattern or the structural one.
        # Let's iterate from 2 bars ago back to max_duration.
        
        # history: [... , Start, ... , Peak, ... , Trigger]
        # index:          -i           -k           -1
        
        # history: [... , Start, ... , Peak, ... , Trigger]
        # index:          -i           -k           -1
        
        for i in range(2, self.max_duration + 1):
            if i >= len(history):
                break
                
            start_idx = -i
            start_vals = history[start_idx]
            start_close = start_vals[3]
            
            # --- SHORT SETUP CHECK ---
            # Current close < Start close (Breakdown)
            if current_close < start_close:
                window = history[start_idx+1 : -1] # Exclude start and current
                if len(window) == 0: continue
                
                window_highs = window[:, 1]
                window_closes = window[:, 3]
                peak_high = np.max(window_highs)
                min_window_close = np.min(window_closes)
                
                # Invalid if we stayed below start previously
                if min_window_close < start_close:
                    continue

                move_up = peak_high - start_close
                if (move_up >= self.min_move * unit) and (move_up < self.max_move * unit):
                     # Valid SHORT
                    triggered = True
                    limit_price = start_close + (self.entry_level * unit) # Short entry > start
                    stop_price = start_close + (self.stop_level * unit)   # Stop above
                    tp_price = start_close + (self.tp_level * unit)       # TP below (tp_unit is negative usually)

                    context = {
                        "pattern": "Puller_Short",
                        "variation": self.var_id,
                        "start_price": float(start_close),
                        "peak_price": float(peak_high),
                        "atr": float(unit),
                        "move_in_units": float(move_up / unit),
                        "duration_bars": i,
                        "entry_price": float(limit_price),
                        "stop_loss": float(stop_price),
                        "take_profit": float(tp_price),
                        "direction": "SHORT",
                        "order_type": "LIMIT"
                    }
                    break
            
            # --- LONG SETUP CHECK ---
            # Current close > Start close (Breakout up)
            elif current_close > start_close:
                window = history[start_idx+1 : -1]
                if len(window) == 0: continue
                
                window_lows = window[:, 2] # Low is index 2
                window_closes = window[:, 3]
                peak_low = np.min(window_lows)
                max_window_close = np.max(window_closes)
                
                # Invalid if we stayed above start previously
                if max_window_close > start_close:
                    continue
                    
                move_down = start_close - peak_low
                
                # Check Size (using same params, symmetric)
                if (move_down >= self.min_move * unit) and (move_down < self.max_move * unit):
                    # Valid LONG
                    triggered = True
                    # Invert logic for Long
                    # Entry: Start - 0.75 unit (buy dip)
                    # Stop: Start - 2.0 unit (below)
                    # TP: Start + 4.0 unit (above)
                    
                    # Note: entry_unit is usually positive e.g. 0.75.
                    # For Short: Start + 0.75 (higher). 
                    # For Long: Start - 0.75 (lower). 
                    limit_price = start_close - (self.entry_level * unit)
                    
                    # Stop:
                    # Short: Start + 2.0
                    # Long: Start - 2.0
                    stop_price = start_close - (self.stop_level * unit)
                    
                    # TP:
                    # Short: Start + (-4.0) = Start - 4.0
                    # Long: Start - (-4.0) = Start + 4.0
                    tp_price = start_close - (self.tp_level * unit)
                    
                    context = {
                        "pattern": "Puller_Long",
                        "variation": self.var_id,
                        "start_price": float(start_close),
                        "peak_price": float(peak_low),
                        "atr": float(unit),
                        "move_in_units": float(move_down / unit),
                        "duration_bars": i,
                        "entry_price": float(limit_price),
                        "stop_loss": float(stop_price),
                        "take_profit": float(tp_price),
                        "direction": "LONG",
                        "order_type": "LIMIT"
                    }
                    break

        return ScanResult(
            scanner_id=self.scanner_id,
            triggered=triggered,
            context=context,
            score=1.0 if triggered else 0.0
        )

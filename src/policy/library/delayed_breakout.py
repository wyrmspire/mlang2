"""
Delayed Breakout Scanner (1.4 RR)
Triggers trades on 15m swing breakouts only after 11:30 AM (2 hours after open).
Uses a fixed 1.4 Risk:Reward ratio.
"""

import pandas as pd
import numpy as np
from typing import Optional
from dataclasses import dataclass
from datetime import time

from src.policy.scanners import Scanner, ScannerResult
from src.features.state import MarketState
from src.features.pipeline import FeatureBundle


from src.config import POINT_VALUE, TICK_SIZE

@dataclass
class DelayedBreakoutState:
    """Tracks swing levels and cooldown."""
    last_trigger_bar: int = -1000
    last_swing_high: float = 0.0
    last_swing_low: float = 0.0


class DelayedBreakoutScanner(Scanner):
    """
    Scanner that triggers on breakout of 15m swing structure.
    
    Specific Rules:
    1. TIME FILTER: No trades before 11:30 AM.
    2. ENTRY: Breakout of N-bar 15m high/low.
    3. STOP: Opposite swing level.
    4. TARGET: Fixed 1.4 Risk:Reward Ratio.
    5. SIZING: Dynamic contracts based on max_risk_dollars.
    
    Config:
        lookback_bars: Number of 15m bars to look back for swing detection.
        min_atr_distance: Minimum breakout distance in ATR to avoid noise.
        cooldown_bars: Minimum 1m bars between triggers.
        max_risk_dollars: Maximum dollar risk per trade (default 300).
    """
    
    def __init__(
        self,
        lookback_bars: int = 20, 
        min_atr_distance: float = 0.3,
        cooldown_bars: int = 30,
        max_risk_dollars: float = 300.0
    ):
        self.lookback_bars = lookback_bars
        self.min_atr_distance = min_atr_distance
        self.cooldown_bars = cooldown_bars
        self.max_risk_dollars = max_risk_dollars
        self._state = DelayedBreakoutState()
        
        # 11:30 AM cutoff
        self.start_time = time(11, 30)
        self.end_time = time(15, 30)
    
    @property
    def scanner_id(self) -> str:
        return f"delayed_breakout_1.4rr"
    
    def _compute_swing_levels(self, df_15m: pd.DataFrame, current_time: pd.Timestamp) -> tuple:
        """Compute swing high and swing low from recent 15m bars."""
        if df_15m is None or df_15m.empty:
            return (0.0, 0.0)
        
        # Get bars up to current time
        mask = df_15m['time'] <= current_time
        recent = df_15m.loc[mask].tail(self.lookback_bars + 1)
        
        if len(recent) < 5:
            return (0.0, 0.0)
        
        # Exclude current bar
        lookback = recent.iloc[:-1] if len(recent) > 1 else recent
        
        swing_high = lookback['high'].max()
        swing_low = lookback['low'].min()
        
        return (swing_high, swing_low)
    
    def _find_recent_engulfing(self, df_5m: pd.DataFrame, current_time: pd.Timestamp, direction: str) -> tuple:
        """
        Find the most recent 5m engulfing candle in the specified direction.
        Returns (low, high, found_bool).
        
        Engulfing Definition:
        - Bullish (LONG): Close > Open AND Body engulfs previous Red candle body. 
          (Simpler: Close > Prev Open and Open < Prev Close).
        - Bearish (SHORT): Close < Open AND Body engulfs previous Green candle body.
        
        Lookback: 4 hours (~48 bars).
        """
        if df_5m is None or df_5m.empty:
            return (0.0, 0.0, False)
            
        mask = df_5m['time'] < current_time # Strictly before entry trigger? Or including current? Trigger is current.
        # usually 5m bar is forming. If strategy runs on 1m bars, 5m bar might not be closed.
        # We should look at COMPLETED 5m bars.
        # Assuming df_5m contains closed bars or we check indices.
        # Safest to check bars < current_time
        
        recent = df_5m.loc[mask].tail(48) # 4 hours
        if len(recent) < 2:
            return (0.0, 0.0, False)
        
        # Iterate backwards
        bars = recent.to_dict('records')
        for i in range(len(bars) - 1, 0, -1):
            curr = bars[i]
            prev = bars[i-1]
            
            curr_body_top = max(curr['open'], curr['close'])
            curr_body_bottom = min(curr['open'], curr['close'])
            prev_body_top = max(prev['open'], prev['close'])
            prev_body_bottom = min(prev['open'], prev['close'])
            
            is_bullish = curr['close'] > curr['open']
            is_bearish = curr['close'] < curr['open']
            
            if direction == "LONG":
                # Bullish Engulfing
                if is_bullish and curr_body_top > prev_body_top and curr_body_bottom < prev_body_bottom:
                    return (curr['low'], curr['high'], True)
                    
            elif direction == "SHORT":
                # Bearish Engulfing
                if is_bearish and curr_body_top > prev_body_top and curr_body_bottom < prev_body_bottom:
                    return (curr['low'], curr['high'], True)
                    
        return (0.0, 0.0, False)

    def _calculate_position_size(self, entry: float, stop: float) -> tuple:
        """Calculate max contracts for given risk."""
        dist_points = abs(entry - stop)
        if dist_points < TICK_SIZE: return 0, 0.0
        
        risk_per_contract = dist_points * POINT_VALUE
        if risk_per_contract <= 0: return 0, 0.0
        
        # Max contracts
        contracts = int(self.max_risk_dollars // risk_per_contract)
        contracts = max(1, contracts) 
        
        if contracts * risk_per_contract > self.max_risk_dollars:
             if contracts == 0: return 0, 0.0
        
        contract_risk = contracts * risk_per_contract
        return contracts, contract_risk

    def scan(
        self,
        state: MarketState,
        features: FeatureBundle,
        df_15m: pd.DataFrame = None,
        df_5m: pd.DataFrame = None
    ) -> ScannerResult:
        """Check for delayed breakout."""
        t = features.timestamp
        if t is None:
            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
        
        # 1. TIME FILTER
        current_t = t.time()
        if current_t < self.start_time or current_t > self.end_time:
            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
            
        # Cooldown check
        if features.bar_idx - self._state.last_trigger_bar < self.cooldown_bars:
            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
        
        # Need data
        if df_15m is None or df_15m.empty:
            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
        
        # Compute swings (for Entry)
        swing_high, swing_low = self._compute_swing_levels(df_15m, t)
        
        if swing_high == 0 or swing_low == 0:
            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
        
        current_price = features.current_price
        atr = features.atr if features.atr > 0 else 1.0
        min_breakout = self.min_atr_distance * atr
        
        # Check Breakout
        long_breakout = current_price > swing_high and (current_price - swing_high) >= min_breakout
        short_breakout = current_price < swing_low and (swing_low - current_price) >= min_breakout
        
        if long_breakout or short_breakout:
            direction = "LONG" if long_breakout else "SHORT"
            
            # 2. Risk Management (Stop on Engulfing + ATR Padding)
            # Find Engulfing
            eng_low, eng_high, found_engulfing = self._find_recent_engulfing(df_5m, t, direction)
            
            # Stop Calculation
            padding = 0.5 * atr # "ATR padding" - let's assume 0.5 or 1.0? User said "atr padding". 1.0 is safe.
            # User said "sl the low with atr padding".
            
            if direction == "LONG":
                if found_engulfing:
                    stop_price = eng_low - padding
                else:
                    # Fallback to swing low if no engulfing found
                    stop_price = swing_low - padding
                
                risk = current_price - stop_price
                if risk <= 0: return ScannerResult(scanner_id=self.scanner_id, triggered=False)
                tp_price = current_price + (1.4 * risk)
                
            else: # SHORT
                if found_engulfing:
                    stop_price = eng_high + padding
                else:
                    stop_price = swing_high + padding
                
                risk = stop_price - current_price
                if risk <= 0: return ScannerResult(scanner_id=self.scanner_id, triggered=False)
                tp_price = current_price - (1.4 * risk)
            
            # 3. Position Sizing
            contracts, actual_risk = self._calculate_position_size(current_price, stop_price)
            if contracts == 0:
                 return ScannerResult(scanner_id=self.scanner_id, triggered=False)

            self._state.last_trigger_bar = features.bar_idx
            
            return ScannerResult(
                scanner_id=self.scanner_id,
                triggered=True,
                context={
                    'direction': direction,
                    'swing_high': swing_high,
                    'swing_low': swing_low,
                    'engulfing_found': found_engulfing,
                    'entry_price': current_price,
                    'stop_price': stop_price,
                    'tp_price': tp_price,
                    'risk_points': abs(risk),
                    'reward_points': abs(tp_price - current_price),
                    'contracts': contracts,
                    'risk_dollars': actual_risk,
                    'reward_dollars': actual_risk * 1.4,
                    'rr_ratio': 1.4
                },
                score=1.0
            )
            
        return ScannerResult(scanner_id=self.scanner_id, triggered=False)

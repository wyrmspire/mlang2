"""
Structure Break Scanner (15m, 2RR)
Triggers on structure break pattern:
1. Price makes new high/low
2. Then breaks the SECOND previous low/high (not the immediate one)
3. Wait for 15m candle close
4. Enter in direction of break
5. SL at candle extreme + 0.5 ATR
6. TP at 2RR
"""

import pandas as pd
import numpy as np
from typing import Optional, List
from dataclasses import dataclass, field
from datetime import time

from src.policy.scanners import Scanner, ScannerResult
from src.features.state import MarketState
from src.features.pipeline import FeatureBundle
from src.config import POINT_VALUE, TICK_SIZE


@dataclass
class SwingPoint:
    """Represents a swing high or low."""
    price: float
    bar_idx: int
    time: pd.Timestamp
    is_high: bool  # True for swing high, False for swing low


@dataclass
class StructureBreakState:
    """Tracks swing structure and pattern detection."""
    swing_highs: List[SwingPoint] = field(default_factory=list)
    swing_lows: List[SwingPoint] = field(default_factory=list)
    last_trigger_bar: int = -1000
    recent_new_high: bool = False
    recent_new_low: bool = False
    new_high_bar: int = -1
    new_low_bar: int = -1


class StructureBreakScanner(Scanner):
    """
    Scanner that triggers on structure break pattern.
    
    Pattern (SHORT example):
    1. Price makes a new swing HIGH
    2. Then breaks the SECOND previous swing LOW (not the immediate one)
    3. Wait for 15m candle to close
    4. Enter SHORT at close
    5. SL at HIGH of trigger candle + 0.5 ATR
    6. TP = Entry - 2 * Risk
    
    Config:
        swing_lookback: Bars to look back for swing detection
        min_swing_bars: Minimum bars between swings
        cooldown_bars: Minimum bars between triggers
        max_risk_dollars: Maximum dollar risk per trade
        atr_padding: ATR multiple for SL padding (default 0.5)
    """
    
    def __init__(
        self,
        swing_lookback: int = 5,  # Bars left/right for swing detection
        min_swing_bars: int = 3,  # Min bars between swings
        cooldown_bars: int = 3,   # 3 bars = 45 min on 15m
        max_risk_dollars: float = 300.0,
        atr_padding: float = 0.5
    ):
        self.swing_lookback = swing_lookback
        self.min_swing_bars = min_swing_bars
        self.cooldown_bars = cooldown_bars
        self.max_risk_dollars = max_risk_dollars
        self.atr_padding = atr_padding
        self._state = StructureBreakState()
    
    @property
    def scanner_id(self) -> str:
        return "structure_break_2rr"
    
    def _find_swings(self, df_15m: pd.DataFrame, current_bar_idx: int) -> tuple:
        """
        Find swing highs and swing lows in the 15m data.
        A swing high is a bar whose high is higher than N bars before and after.
        Returns updated lists of swing highs and lows.
        """
        if df_15m is None or len(df_15m) < self.swing_lookback * 2 + 1:
            return self._state.swing_highs, self._state.swing_lows
        
        # Use recent data
        recent = df_15m.tail(100).copy()
        if len(recent) < self.swing_lookback * 2 + 1:
            return self._state.swing_highs, self._state.swing_lows
        
        highs = recent['high'].values
        lows = recent['low'].values
        times = recent['time'].values
        
        swing_highs = []
        swing_lows = []
        
        # Find swings (excluding the last swing_lookback bars - they're not confirmed)
        for i in range(self.swing_lookback, len(recent) - self.swing_lookback):
            # Check swing high
            is_swing_high = True
            for j in range(1, self.swing_lookback + 1):
                if highs[i] <= highs[i-j] or highs[i] <= highs[i+j]:
                    is_swing_high = False
                    break
            
            if is_swing_high:
                swing_highs.append(SwingPoint(
                    price=highs[i],
                    bar_idx=current_bar_idx - (len(recent) - 1 - i),
                    time=pd.Timestamp(times[i]),
                    is_high=True
                ))
            
            # Check swing low
            is_swing_low = True
            for j in range(1, self.swing_lookback + 1):
                if lows[i] >= lows[i-j] or lows[i] >= lows[i+j]:
                    is_swing_low = False
                    break
            
            if is_swing_low:
                swing_lows.append(SwingPoint(
                    price=lows[i],
                    bar_idx=current_bar_idx - (len(recent) - 1 - i),
                    time=pd.Timestamp(times[i]),
                    is_high=False
                ))
        
        # Keep only recent swings (last 10 of each)
        return swing_highs[-10:], swing_lows[-10:]
    
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
        if contract_risk > self.max_risk_dollars * 1.1:  # Allow 10% tolerance
            contracts = max(1, contracts - 1)
            contract_risk = contracts * risk_per_contract
        
        return contracts, contract_risk

    def scan(
        self,
        state: MarketState,
        features: FeatureBundle,
        df_15m: pd.DataFrame = None
    ) -> ScannerResult:
        """Check for structure break pattern."""
        t = features.timestamp
        if t is None:
            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
        
        # Cooldown check
        if features.bar_idx - self._state.last_trigger_bar < self.cooldown_bars:
            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
        
        if df_15m is None or df_15m.empty:
            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
        
        # Update swing structure
        self._state.swing_highs, self._state.swing_lows = self._find_swings(df_15m, features.bar_idx)
        
        # Need at least 3 swing highs and 3 swing lows
        if len(self._state.swing_highs) < 3 or len(self._state.swing_lows) < 3:
            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
        
        # Get current bar data
        current_bar = df_15m.iloc[-1]
        current_close = current_bar['close']
        current_high = current_bar['high']
        current_low = current_bar['low']
        
        atr = features.atr if features.atr > 0 else 5.0  # Default ATR
        
        # Check for SHORT setup:
        # 1. Recent new high (highest of recent swing highs)
        # 2. Current bar breaks the SECOND previous swing low
        
        sorted_highs = sorted(self._state.swing_highs, key=lambda x: x.bar_idx, reverse=True)
        sorted_lows = sorted(self._state.swing_lows, key=lambda x: x.bar_idx, reverse=True)
        
        # Check if we made a new high recently
        if len(sorted_highs) >= 2:
            most_recent_high = sorted_highs[0]
            prev_highs = sorted_highs[1:]
            is_new_high = all(most_recent_high.price > h.price for h in prev_highs[:3])
            
            if is_new_high and len(sorted_lows) >= 2:
                # Second previous low (skip the most recent one)
                second_prev_low = sorted_lows[1] if len(sorted_lows) > 1 else sorted_lows[0]
                
                # Check if current bar breaks the second previous low
                if current_close < second_prev_low.price:
                    # SHORT SIGNAL
                    entry_price = current_close
                    stop_price = current_high + (self.atr_padding * atr)
                    risk = stop_price - entry_price
                    
                    if risk > 0:
                        tp_price = entry_price - (2.0 * risk)
                        
                        contracts, actual_risk = self._calculate_position_size(entry_price, stop_price)
                        if contracts > 0:
                            self._state.last_trigger_bar = features.bar_idx
                            
                            return ScannerResult(
                                scanner_id=self.scanner_id,
                                triggered=True,
                                context={
                                    'direction': 'SHORT',
                                    'entry_price': entry_price,
                                    'stop_price': stop_price,
                                    'tp_price': tp_price,
                                    'trigger_high': current_high,
                                    'trigger_low': current_low,
                                    'broken_level': second_prev_low.price,
                                    'new_high_price': most_recent_high.price,
                                    'risk_points': risk,
                                    'reward_points': 2.0 * risk,
                                    'contracts': contracts,
                                    'risk_dollars': actual_risk,
                                    'reward_dollars': actual_risk * 2.0,
                                    'rr_ratio': 2.0
                                },
                                score=1.0
                            )
        
        # Check for LONG setup:
        # 1. Recent new low (lowest of recent swing lows)
        # 2. Current bar breaks the SECOND previous swing high
        
        if len(sorted_lows) >= 2:
            most_recent_low = sorted_lows[0]
            prev_lows = sorted_lows[1:]
            is_new_low = all(most_recent_low.price < l.price for l in prev_lows[:3])
            
            if is_new_low and len(sorted_highs) >= 2:
                # Second previous high (skip the most recent one)
                second_prev_high = sorted_highs[1] if len(sorted_highs) > 1 else sorted_highs[0]
                
                # Check if current bar breaks the second previous high
                if current_close > second_prev_high.price:
                    # LONG SIGNAL
                    entry_price = current_close
                    stop_price = current_low - (self.atr_padding * atr)
                    risk = entry_price - stop_price
                    
                    if risk > 0:
                        tp_price = entry_price + (2.0 * risk)
                        
                        contracts, actual_risk = self._calculate_position_size(entry_price, stop_price)
                        if contracts > 0:
                            self._state.last_trigger_bar = features.bar_idx
                            
                            return ScannerResult(
                                scanner_id=self.scanner_id,
                                triggered=True,
                                context={
                                    'direction': 'LONG',
                                    'entry_price': entry_price,
                                    'stop_price': stop_price,
                                    'tp_price': tp_price,
                                    'trigger_high': current_high,
                                    'trigger_low': current_low,
                                    'broken_level': second_prev_high.price,
                                    'new_low_price': most_recent_low.price,
                                    'risk_points': risk,
                                    'reward_points': 2.0 * risk,
                                    'contracts': contracts,
                                    'risk_dollars': actual_risk,
                                    'reward_dollars': actual_risk * 2.0,
                                    'rr_ratio': 2.0
                                },
                                score=1.0
                            )
        
        return ScannerResult(scanner_id=self.scanner_id, triggered=False)

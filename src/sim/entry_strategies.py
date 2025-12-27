"""
Entry Strategies
Decoupled logic for calculating entry prices.

This module provides a registry of entry strategies that can be used
by the OCO Engine to determine where to place limit/stop orders.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

from src.sim.costs import CostModel

class EntryStrategy(ABC):
    """Base class for entry price calculation strategies."""
    
    @abstractmethod
    def calculate_entry(
        self,
        base_price: float,
        direction: str,
        bar: pd.Series,
        atr: float,
        params: Dict[str, Any],
        costs: CostModel,
        context: Dict[str, Any] = None
    ) -> float:
        """
        Calculate entry price.
        
        Args:
            base_price: Current/Signal price (often Close of signal bar)
            direction: 'LONG' or 'SHORT'
            bar: The signal bar (current completed bar)
            atr: Current ATR
            params: Strategy-specific parameters (e.g. {'pct': 0.5})
            costs: Cost model for tick rounding
            context: Optional context (e.g. htf_bars, indicators)
            
        Returns:
            Calculated entry price
        """
        pass


class MarketEntry(EntryStrategy):
    """Enter at market (Open of next bar, effectively)."""
    
    def calculate_entry(self, base_price, direction, bar, atr, params, costs, context=None) -> float:
        return base_price


class LimitOffsetEntry(EntryStrategy):
    """Legacy: Limit at Base +/- ATR offset."""
    
    def calculate_entry(self, base_price, direction, bar, atr, params, costs, context=None) -> float:
        offset = params.get('offset_atr', 0.0)
        if direction == 'LONG':
            price = base_price - (offset * atr)
            return costs.round_to_tick(price, 'down')
        else:
            price = base_price + (offset * atr)
            return costs.round_to_tick(price, 'up')


class SignalRetraceEntry(EntryStrategy):
    """
    Limit at X% retrace of the signal bar range.
    
    Long: Low + (Range * pct) 
    Short: High - (Range * pct)
    """
    
    def calculate_entry(self, base_price, direction, bar, atr, params, costs, context=None) -> float:
        pct = params.get('pct', 0.5) # Default 50%
        bar_range = bar['high'] - bar['low']
        
        if direction == 'LONG':
            # Buy limit below close, ideally. 
            # Logic: We want to buy at the bottom X% of the candle
            price = bar['low'] + (bar_range * pct)
            return costs.round_to_tick(price, 'down')
        else:
            # Sell limit above close
            price = bar['high'] - (bar_range * pct)
            return costs.round_to_tick(price, 'up')


class TimeframeRetraceEntry(EntryStrategy):
    """
    Limit at X% retrace of the last completed bar on a SPECIFIC timeframe.
    
    Requires 'df_context' in context with keys like '5m', '15m'.
    User Request: '50 percent of the last 15m'
    """
    
    def calculate_entry(self, base_price, direction, bar, atr, params, costs, context=None) -> float:
        tf = params.get('timeframe', '15m')
        pct = params.get('pct', 0.5)
        
        if not context or tf not in context:
            # Fallback to current bar if HTF data missing
            return SignalRetraceEntry().calculate_entry(base_price, direction, bar, atr, params, costs, context)
            
        # Get last closed bar for timeframe
        htf_bar = context[tf].iloc[-1]
        range_val = htf_bar['high'] - htf_bar['low']
        
        if direction == 'LONG':
            price = htf_bar['low'] + (range_val * pct)
            return costs.round_to_tick(price, 'down')
        else:
            price = htf_bar['high'] - (range_val * pct)
            return costs.round_to_tick(price, 'up')


class BreakoutEntry(EntryStrategy):
    """
    Stop-Limit entry at signal bar High/Low + Offset.
    
    Long: High + Offset
    Short: Low - Offset
    """
    
    def calculate_entry(self, base_price, direction, bar, atr, params, costs, context=None) -> float:
        offset_atr = params.get('offset_atr', 0.1)
        offset = offset_atr * atr
        
        if direction == 'LONG':
            price = bar['high'] + offset
            return costs.round_to_tick(price, 'up')
        else:
            price = bar['low'] - offset
            return costs.round_to_tick(price, 'down')


class RangeBreakoutEntry(EntryStrategy):
    """
    Breakout of the last N bars High/Low.
    """
    
    def calculate_entry(self, base_price, direction, bar, atr, params, costs, context=None) -> float:
        lookback = int(params.get('lookback', 5))
        
        # Need history from context or assume current bar is end of history?
        # Ideally context has 'history_1m'
        history = context.get('history', pd.DataFrame()) if context else pd.DataFrame()
        
        if len(history) < lookback:
            # Fallback to single bar breakout
            return BreakoutEntry().calculate_entry(base_price, direction, bar, atr, params, costs, context)
            
        recent = history.iloc[-lookback:]
        
        if direction == 'LONG':
            price = recent['high'].max()
            return costs.round_to_tick(price, 'up')
        else:
            price = recent['low'].min()
            return costs.round_to_tick(price, 'down')


class VwapReversionEntry(EntryStrategy):
    """
    Limit entry at VWAP.
    """
    def calculate_entry(self, base_price, direction, bar, atr, params, costs, context=None) -> float:
        vwap = context.get('vwap') if context else None
        
        if vwap is None:
             # Fallback to market if no VWAP
             return base_price
             
        if direction == 'LONG':
            return costs.round_to_tick(vwap, 'down')
        else:
            return costs.round_to_tick(vwap, 'up')


class EntryRegistry:
    """Registry for entry strategies."""
    
    _strategies = {
        'market': MarketEntry(),
        'limit_offset': LimitOffsetEntry(), # Legacy 'limit'
        'retrace_signal': SignalRetraceEntry(),
        'retrace_timeframe': TimeframeRetraceEntry(),
        'breakout': BreakoutEntry(),
        'breakout_range': RangeBreakoutEntry(),
        'vwap': VwapReversionEntry(),
    }
    
    @classmethod
    def get(cls, name: str) -> EntryStrategy:
        return cls._strategies.get(name, cls._strategies['market']) # Default to market

    @classmethod
    def list_strategies(cls):
        return list(cls._strategies.keys())

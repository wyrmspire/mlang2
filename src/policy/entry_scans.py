"""
Entry Scans - Modular Entry Order Modification

Entry Scans refine HOW a trade enters after a trigger fires.
They don't replace scanners/models, they MODIFY the entry order.

Categories (exclusive within each):
- Entry Type: Market vs Limit
- Stop Method: ATR, Behind Swing, Fixed Bars
- TP Method: ATR, R-Multiple

Usage:
    from src.policy.entry_scans import apply_entry_scans, EntryConfig
    
    config = EntryConfig(
        entry_type='limit',
        stop_method='swing',
        tp_method='r_multiple'
    )
    modified_order = apply_entry_scans(base_order, df_history, config)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np


@dataclass
class EntryOrder:
    """Represents a trade entry order."""
    direction: str  # 'LONG' or 'SHORT'
    entry_price: float
    stop_price: float
    tp_price: float
    entry_type: str = 'market'  # 'market' or 'limit'
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EntryConfig:
    """Configuration for entry scan processing."""
    entry_type: str = 'market'  # 'market' or 'limit'
    stop_method: str = 'atr'    # 'atr', 'swing', 'fixed_bars'
    tp_method: str = 'atr'      # 'atr', 'r_multiple'
    
    # ATR parameters
    stop_atr_multiple: float = 1.0
    tp_atr_multiple: float = 2.0
    
    # Swing parameters
    swing_lookback: int = 5
    swing_buffer_atr: float = 0.1  # Buffer beyond swing level
    
    # Fixed bars parameters
    fixed_bars_lookback: int = 3
    
    # R-multiple parameters
    tp_r_multiple: float = 2.0


# =============================================================================
# Entry Type Modifiers
# =============================================================================

class EntryTypeModifier(ABC):
    """Base class for entry type modification."""
    
    @abstractmethod
    def modify(self, order: EntryOrder, bar: pd.Series) -> EntryOrder:
        pass


class MarketEntry(EntryTypeModifier):
    """Enter at current market price (close of bar)."""
    
    def modify(self, order: EntryOrder, bar: pd.Series) -> EntryOrder:
        order.entry_type = 'market'
        order.entry_price = float(bar['close'])
        return order


class LimitAtDecision(EntryTypeModifier):
    """Place limit order at the decision point price."""
    
    def modify(self, order: EntryOrder, bar: pd.Series) -> EntryOrder:
        order.entry_type = 'limit'
        # Keep entry_price as-is (from signal)
        return order


# =============================================================================
# Stop Placement Modifiers
# =============================================================================

class StopModifier(ABC):
    """Base class for stop loss modification."""
    
    @abstractmethod
    def modify(self, order: EntryOrder, df_history: pd.DataFrame, config: EntryConfig) -> EntryOrder:
        pass


class StopAtATR(StopModifier):
    """Place stop at N×ATR from entry."""
    
    def modify(self, order: EntryOrder, df_history: pd.DataFrame, config: EntryConfig) -> EntryOrder:
        # Calculate ATR
        high = df_history['high']
        low = df_history['low']
        close = df_history['close'].shift(1)
        tr = pd.concat([high - low, abs(high - close), abs(low - close)], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])
        
        if order.direction == 'LONG':
            order.stop_price = order.entry_price - (atr * config.stop_atr_multiple)
        else:
            order.stop_price = order.entry_price + (atr * config.stop_atr_multiple)
        
        return order


class StopBehindSwing(StopModifier):
    """Place stop behind the most recent swing high/low."""
    
    def modify(self, order: EntryOrder, df_history: pd.DataFrame, config: EntryConfig) -> EntryOrder:
        lookback = config.swing_lookback
        recent = df_history.tail(lookback * 2)
        
        # Calculate small ATR for buffer
        tr = recent['high'] - recent['low']
        atr = float(tr.mean())
        buffer = atr * config.swing_buffer_atr
        
        if order.direction == 'LONG':
            # Find swing low (lowest low in lookback)
            swing_low = float(recent['low'].min())
            order.stop_price = swing_low - buffer
        else:
            # Find swing high (highest high in lookback)
            swing_high = float(recent['high'].max())
            order.stop_price = swing_high + buffer
        
        return order


class StopAtFixedBars(StopModifier):
    """Place stop at high/low of N bars ago."""
    
    def modify(self, order: EntryOrder, df_history: pd.DataFrame, config: EntryConfig) -> EntryOrder:
        lookback = config.fixed_bars_lookback
        ref_bar = df_history.iloc[-lookback] if len(df_history) >= lookback else df_history.iloc[0]
        
        if order.direction == 'LONG':
            order.stop_price = float(ref_bar['low'])
        else:
            order.stop_price = float(ref_bar['high'])
        
        return order


# =============================================================================
# Take Profit Modifiers
# =============================================================================

class TPModifier(ABC):
    """Base class for take profit modification."""
    
    @abstractmethod
    def modify(self, order: EntryOrder, df_history: pd.DataFrame, config: EntryConfig) -> EntryOrder:
        pass


class TPAtATR(TPModifier):
    """Place TP at N×ATR from entry."""
    
    def modify(self, order: EntryOrder, df_history: pd.DataFrame, config: EntryConfig) -> EntryOrder:
        # Calculate ATR
        high = df_history['high']
        low = df_history['low']
        close = df_history['close'].shift(1)
        tr = pd.concat([high - low, abs(high - close), abs(low - close)], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])
        
        if order.direction == 'LONG':
            order.tp_price = order.entry_price + (atr * config.tp_atr_multiple)
        else:
            order.tp_price = order.entry_price - (atr * config.tp_atr_multiple)
        
        return order


class TPAtRMultiple(TPModifier):
    """Place TP at R×risk from entry (R-multiple)."""
    
    def modify(self, order: EntryOrder, df_history: pd.DataFrame, config: EntryConfig) -> EntryOrder:
        # Calculate risk (distance to stop)
        risk = abs(order.entry_price - order.stop_price)
        
        if order.direction == 'LONG':
            order.tp_price = order.entry_price + (risk * config.tp_r_multiple)
        else:
            order.tp_price = order.entry_price - (risk * config.tp_r_multiple)
        
        return order


# =============================================================================
# Entry Scan Registry
# =============================================================================

ENTRY_TYPE_MODIFIERS = {
    'market': MarketEntry(),
    'limit': LimitAtDecision(),
}

STOP_MODIFIERS = {
    'atr': StopAtATR(),
    'swing': StopBehindSwing(),
    'fixed_bars': StopAtFixedBars(),
}

TP_MODIFIERS = {
    'atr': TPAtATR(),
    'r_multiple': TPAtRMultiple(),
}


# =============================================================================
# Main Entry Point
# =============================================================================

def apply_entry_scans(
    base_order: EntryOrder,
    df_history: pd.DataFrame,
    config: EntryConfig,
    current_bar: Optional[pd.Series] = None
) -> EntryOrder:
    """
    Apply entry scans to modify the base order.
    
    Args:
        base_order: Initial order from signal
        df_history: Historical OHLCV data
        config: Entry scan configuration
        current_bar: Current bar data (for entry type)
        
    Returns:
        Modified EntryOrder with final levels
    """
    order = base_order
    
    # Apply entry type modifier
    if current_bar is not None and config.entry_type in ENTRY_TYPE_MODIFIERS:
        order = ENTRY_TYPE_MODIFIERS[config.entry_type].modify(order, current_bar)
    
    # Apply stop modifier (must be before TP if using R-multiple)
    if config.stop_method in STOP_MODIFIERS:
        order = STOP_MODIFIERS[config.stop_method].modify(order, df_history, config)
    
    # Apply TP modifier
    if config.tp_method in TP_MODIFIERS:
        order = TP_MODIFIERS[config.tp_method].modify(order, df_history, config)
    
    return order


def create_default_config() -> EntryConfig:
    """Create default entry config (current behavior)."""
    return EntryConfig(
        entry_type='market',
        stop_method='atr',
        tp_method='atr',
        stop_atr_multiple=1.0,
        tp_atr_multiple=2.0
    )

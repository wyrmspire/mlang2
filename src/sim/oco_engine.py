"""
OCO Engine - Unified One-Cancels-Other Order Management

This is the SINGLE authoritative implementation for OCO bracket logic.
All other implementations should use this engine.

Key Features:
- Standardized stop/TP priority rules
- Tick size rounding
- Consistent bars_held calculation
- Flat oco_results output format
- Integration with stop_calculator for smart stops
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List
from enum import Enum
import pandas as pd

from src.sim.execution import Order, OrderType, OrderStatus, Fill
from src.sim.bar_fill_model import BarFillEngine, BarFillConfig, DEFAULT_FILL_ENGINE
from src.sim.costs import CostModel, DEFAULT_COSTS
from src.sim.stop_calculator import StopType, StopConfig, calculate_stop, calculate_tp_from_risk


class OCOStatus(Enum):
    """OCO bracket status."""
    PENDING_ENTRY = "PENDING_ENTRY"   # Waiting for entry fill
    ACTIVE = "ACTIVE"                  # Entry filled, SL/TP pending
    CLOSED_TP = "CLOSED_TP"           # Closed by take profit
    CLOSED_SL = "CLOSED_SL"           # Closed by stop loss
    CLOSED_TIMEOUT = "CLOSED_TIMEOUT" # Closed by max bars
    CANCELLED = "CANCELLED"           # Entry expired/cancelled


class ExitPriority(Enum):
    """
    Priority rule when both SL and TP would trigger in same bar.
    
    According to ARCHITECTURE_AGREEMENT.md:
    - STOP_FIRST: Conservative - assume worst case (default)
    - TP_FIRST: Optimistic - assume best case
    - RANDOM: Random selection (for sensitivity analysis)
    - INTRABAR_MODEL: Use bar fill model (if available)
    """
    STOP_FIRST = "STOP_FIRST"
    TP_FIRST = "TP_FIRST"
    RANDOM = "RANDOM"
    INTRABAR_MODEL = "INTRABAR_MODEL"


@dataclass
class OCOConfig:
    """
    Unified OCO bracket configuration.
    
    Supports both legacy ATR-based stops and modern smart stops.
    """
    direction: str = "LONG"         # 'LONG' or 'SHORT'
    
    # Entry
    entry_type: str = "LIMIT"       # 'MARKET', 'LIMIT', 'RETRACE', etc.
    entry_params: Dict[str, Any] = field(default_factory=dict)  # Params for entry strategy
    entry_offset_atr: float = 0.25  # Legacy support (maps to limit_offset params)
    
    # Stop configuration
    stop_config: Optional[StopConfig] = None  # Use smart stops if provided
    stop_atr: float = 1.0                     # Legacy: ATR-based stop (if stop_config is None)
    
    # Take profit
    tp_multiple: float = 1.4        # Take profit as multiple of risk
    
    # Limits
    max_bars: int = 200             # Max bars in trade
    
    # Exit priority
    exit_priority: ExitPriority = ExitPriority.STOP_FIRST
    
    # Unique ID
    name: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'direction': self.direction,
            'entry_type': self.entry_type,
            'entry_params': self.entry_params,
            'entry_offset_atr': self.entry_offset_atr,
            'stop_atr': self.stop_atr if self.stop_config is None else None,
            'stop_config': self.stop_config.to_dict() if self.stop_config else None,
            'tp_multiple': self.tp_multiple,
            'max_bars': self.max_bars,
            'exit_priority': self.exit_priority.value,
            'name': self.name,
        }


@dataclass
class OCOBracket:
    """
    Active OCO bracket state.
    
    This is the runtime state of an OCO order.
    """
    config: OCOConfig
    
    # Computed prices (rounded to tick size)
    entry_price: float = 0.0
    stop_price: float = 0.0
    tp_price: float = 0.0
    
    # State
    status: OCOStatus = OCOStatus.PENDING_ENTRY
    entry_bar: int = 0
    entry_fill: Optional[Fill] = None
    exit_fill: Optional[Fill] = None
    
    # Reference data for logging
    atr_at_creation: float = 0.0
    
    # Tracking (for analytics)
    bars_in_trade: int = 0          # Bars AFTER entry (not including entry bar)
    mae: float = 0.0                # Max Adverse Excursion (points)
    mfe: float = 0.0                # Max Favorable Excursion (points)
    
    def to_flat_dict(self) -> Dict[str, Any]:
        """
        Convert to flat dictionary for oco_results.
        
        This is the AUTHORITATIVE format for UI consumption.
        According to ARCHITECTURE_AGREEMENT.md, oco_results MUST be flat.
        """
        return {
            'direction': self.config.direction,
            'entry_price': self.entry_price,
            'stop_price': self.stop_price,
            'tp_price': self.tp_price,
            'status': self.status.value,
            'entry_bar': self.entry_bar,
            'bars_held': self.bars_in_trade,  # UI reads this field
            'mae': self.mae,
            'mfe': self.mfe,
            'filled': self.status != OCOStatus.CANCELLED,
            'outcome': self._get_outcome(),
            'exit_price': self.exit_fill.fill_price if self.exit_fill else 0.0,
        }
    
    def _get_outcome(self) -> str:
        """Get outcome string for oco_results."""
        if self.status == OCOStatus.CLOSED_TP:
            return "TP"
        elif self.status == OCOStatus.CLOSED_SL:
            return "SL"
        elif self.status == OCOStatus.CLOSED_TIMEOUT:
            return "TIMEOUT"
        elif self.status == OCOStatus.CANCELLED:
            return "CANCELLED"
        elif self.status == OCOStatus.ACTIVE:
            return "ACTIVE"
        else:
            return "PENDING"


class OCOEngine:
    """
    Unified OCO Engine.
    
    This is the single source of truth for OCO bracket creation and processing.
    """
    
    def __init__(
        self,
        fill_engine: BarFillEngine = None,
        costs: CostModel = None
    ):
        self.fill_engine = fill_engine or DEFAULT_FILL_ENGINE
        self.costs = costs or DEFAULT_COSTS
    
    def create_bracket(
        self,
        config: OCOConfig,
        base_price: float,
        atr: float,
        df_1m: Optional[pd.DataFrame] = None,
        df_htf: Optional[pd.DataFrame] = None,
        current_idx: int = 0,
        range_high: float = 0.0,
        range_low: float = 0.0,
        direction_override: Optional[str] = None,
    ) -> OCOBracket:
        """
        Create OCO bracket with computed price levels.
        
        Args:
            config: OCO configuration
            base_price: Current price or signal bar close
            atr: Current ATR for offset calculations
            df_1m: 1-minute data (for smart stops)
            df_htf: Higher timeframe data (for smart stops)
            current_idx: Current bar index
            range_high: Pre-calculated range high (for range-based stops)
            range_low: Pre-calculated range low (for range-based stops)
            direction_override: Override config.direction (for dynamic scanners)
            
        Returns:
            OCOBracket with rounded prices
        """
        from src.sim.entry_strategies import EntryRegistry

        # Use override direction if provided, else use config
        direction = direction_override or config.direction
        
        # Prepare context for entry strategy
        entry_context = {
            '5m': None, # TODO: Pass these in more reliably
            '15m': df_htf if df_htf is not None else None, # Assuming htf is the needed one for now
            'vwap': None # TODO: Pass in
        }
        
        # Helper params merge (legacy support)
        entry_params = config.entry_params.copy()
        if 'offset_atr' not in entry_params:
            entry_params['offset_atr'] = config.entry_offset_atr
            
        # Calculate entry price using strategy
        strategy = EntryRegistry.get(config.entry_type.lower())
        entry_price = strategy.calculate_entry(
            base_price=base_price,
            direction=direction,
            bar=df_1m.iloc[current_idx] if df_1m is not None else pd.Series({'high': base_price, 'low': base_price, 'close': base_price}), 
            atr=atr,
            params=entry_params,
            costs=self.costs,
            context=entry_context
        )
        
        # Calculate stop price
        if config.stop_config is not None:
            # Use smart stop calculator
            stop_price, stop_reason = calculate_stop(
                direction=direction,
                entry_price=entry_price,
                atr=atr,
                config=config.stop_config,
                df_1m=df_1m,
                df_htf=df_htf,
                current_idx=current_idx,
                range_high=range_high,
                range_low=range_low,
            )
            # Round to tick
            if direction == 'LONG':
                stop_price = self.costs.round_to_tick(stop_price, 'down')
            else:
                stop_price = self.costs.round_to_tick(stop_price, 'up')
        else:
            # Legacy ATR-based stop
            if direction == 'LONG':
                stop_price = self.costs.round_to_tick(
                    entry_price - config.stop_atr * atr, 'down'
                )
            else:
                stop_price = self.costs.round_to_tick(
                    entry_price + config.stop_atr * atr, 'up'
                )
        
        # Calculate TP from risk
        tp_price = calculate_tp_from_risk(
            entry_price=entry_price,
            stop_price=stop_price,
            direction=direction,
            r_multiple=config.tp_multiple
        )
        
        # Round TP to tick
        if direction == 'LONG':
            tp_price = self.costs.round_to_tick(tp_price, 'up')
        else:
            tp_price = self.costs.round_to_tick(tp_price, 'down')
        
        return OCOBracket(
            config=config,
            entry_price=entry_price,
            stop_price=stop_price,
            tp_price=tp_price,
            atr_at_creation=atr,
        )
    
    def process_bar(
        self,
        bracket: OCOBracket,
        bar: pd.Series,
        bar_idx: int
    ) -> Tuple[OCOBracket, Optional[str]]:
        """
        Process one bar for an OCO bracket.
        
        Returns:
            (Updated bracket, event string or None)
            
        Events:
            'ENTRY': Entry filled
            'SL': Stop loss hit
            'TP': Take profit hit
            'TIMEOUT': Max bars reached
        """
        if bracket.status == OCOStatus.PENDING_ENTRY:
            return self._process_entry(bracket, bar, bar_idx)
        elif bracket.status == OCOStatus.ACTIVE:
            return self._process_active(bracket, bar, bar_idx)
        
        # Already closed
        return (bracket, None)
    
    def _process_entry(
        self,
        bracket: OCOBracket,
        bar: pd.Series,
        bar_idx: int
    ) -> Tuple[OCOBracket, Optional[str]]:
        """Process entry fill attempt."""
        if bracket.config.entry_type == 'MARKET':
            # Market entry fills at open (with slippage)
            fill_price = self.fill_engine.costs.apply_slippage(
                bar['open'], bracket.config.direction, 'MARKET'
            )
            bracket.entry_fill = Fill(
                order=Order(OrderType.MARKET, bracket.config.direction, None),
                fill_price=fill_price,
                fill_bar=bar_idx
            )
            bracket.entry_bar = bar_idx
            bracket.status = OCOStatus.ACTIVE
            bracket.entry_price = fill_price  # Update actual entry
            return (bracket, 'ENTRY')
        
        else:  # LIMIT
            fill_price = self.fill_engine.get_limit_entry_fill_price(
                bracket.entry_price,
                bracket.config.direction,
                bar
            )
            if fill_price is not None:
                bracket.entry_fill = Fill(
                    order=Order(OrderType.LIMIT, bracket.config.direction, bracket.entry_price),
                    fill_price=fill_price,
                    fill_bar=bar_idx
                )
                bracket.entry_bar = bar_idx
                bracket.status = OCOStatus.ACTIVE
                bracket.entry_price = fill_price
                return (bracket, 'ENTRY')
        
        return (bracket, None)
    
    def _process_active(
        self,
        bracket: OCOBracket,
        bar: pd.Series,
        bar_idx: int
    ) -> Tuple[OCOBracket, Optional[str]]:
        """Process active bracket for exit."""
        # Increment bars_in_trade (counts bars AFTER entry)
        bracket.bars_in_trade += 1
        
        # Update MAE/MFE
        if bracket.config.direction == 'LONG':
            adverse = bracket.entry_price - bar['low']
            favorable = bar['high'] - bracket.entry_price
        else:
            adverse = bar['high'] - bracket.entry_price
            favorable = bracket.entry_price - bar['low']
        
        bracket.mae = max(bracket.mae, adverse)
        bracket.mfe = max(bracket.mfe, favorable)
        
        # Check timeout (bars_held + 1 to account for entry bar)
        if bracket.bars_in_trade >= bracket.config.max_bars:
            bracket.status = OCOStatus.CLOSED_TIMEOUT
            # Exit at close
            bracket.exit_fill = Fill(
                order=Order(OrderType.MARKET, bracket.config.direction, None),
                fill_price=bar['close'],
                fill_bar=bar_idx
            )
            return (bracket, 'TIMEOUT')
        
        # Check for SL/TP
        result, fill_price = self.fill_engine.check_exit(
            bracket.config.direction,
            bracket.stop_price,
            bracket.tp_price,
            bar,
            bracket.entry_bar,
            bar_idx
        )
        
        if result == 'SL':
            bracket.status = OCOStatus.CLOSED_SL
            bracket.exit_fill = Fill(
                order=Order(OrderType.STOP, bracket.config.direction, bracket.stop_price),
                fill_price=fill_price,
                fill_bar=bar_idx
            )
            return (bracket, 'SL')
        
        elif result == 'TP':
            bracket.status = OCOStatus.CLOSED_TP
            bracket.exit_fill = Fill(
                order=Order(OrderType.LIMIT, bracket.config.direction, bracket.tp_price),
                fill_price=fill_price,
                fill_bar=bar_idx
            )
            return (bracket, 'TP')
        
        return (bracket, None)


# Global engine instance (for backward compatibility)
DEFAULT_OCO_ENGINE = OCOEngine()


def create_oco_bracket(
    config: OCOConfig,
    base_price: float,
    atr: float,
    reference_value: Optional[float] = None,
    costs: CostModel = None,
    **kwargs
) -> OCOBracket:
    """
    Legacy compatibility wrapper for create_oco_bracket.
    
    New code should use OCOEngine directly.
    """
    engine = OCOEngine(costs=costs or DEFAULT_COSTS)
    return engine.create_bracket(config, base_price, atr, **kwargs)


def process_oco_bar(
    bracket: OCOBracket,
    bar: pd.Series,
    bar_idx: int,
    fill_engine: BarFillEngine = None
) -> Tuple[OCOBracket, Optional[str]]:
    """
    Legacy compatibility wrapper for process_oco_bar.
    
    New code should use OCOEngine directly.
    """
    engine = OCOEngine(fill_engine=fill_engine or DEFAULT_FILL_ENGINE)
    return engine.process_bar(bracket, bar, bar_idx)

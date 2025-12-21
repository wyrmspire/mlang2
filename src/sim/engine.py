"""
Simulation Engine - Core OMS and Strategy Runner

This module implements the backend simulation engine with:
- DataStream: Historical bar iterator
- StrategyRunner: Wrapper for model/strategy execution
- OMS: Order Management System with matching logic
- Full market simulation for interactive trading lab
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Iterator, Tuple
from enum import Enum
import pandas as pd

from src.sim.execution import Order, OrderType, OrderStatus, Fill, process_order
from src.sim.oco import OCOBracket, OCOStatus, OCOConfig, create_oco_bracket, process_oco_bar
from src.sim.costs import CostModel, DEFAULT_COSTS
from src.sim.bar_fill_model import BarFillEngine, DEFAULT_FILL_ENGINE


# =============================================================================
# Data Stream
# =============================================================================

class DataStream:
    """
    Historical bar iterator for simulation.
    
    Yields bars one-by-one from a DataFrame.
    """
    
    def __init__(self, df: pd.DataFrame, start_idx: int = 0, end_idx: Optional[int] = None):
        """
        Initialize data stream.
        
        Args:
            df: OHLCV DataFrame with 'time', 'open', 'high', 'low', 'close', 'volume'
            start_idx: Starting index
            end_idx: Ending index (None = end of df)
        """
        self.df = df
        self.start_idx = start_idx
        self.end_idx = end_idx or len(df)
        self.current_idx = start_idx
    
    def __iter__(self):
        """Iterator protocol."""
        self.current_idx = self.start_idx
        return self
    
    def __next__(self) -> Tuple[int, pd.Series]:
        """
        Get next bar.
        
        Returns:
            Tuple of (bar_idx, bar_series)
        """
        if self.current_idx >= self.end_idx:
            raise StopIteration
        
        bar = self.df.iloc[self.current_idx]
        idx = self.current_idx
        self.current_idx += 1
        return (idx, bar)
    
    def peek(self) -> Optional[pd.Series]:
        """Peek at next bar without advancing."""
        if self.current_idx >= self.end_idx:
            return None
        return self.df.iloc[self.current_idx]
    
    def reset(self):
        """Reset to start."""
        self.current_idx = self.start_idx
    
    @property
    def is_done(self) -> bool:
        """Check if stream is exhausted."""
        return self.current_idx >= self.end_idx
    
    @property
    def progress(self) -> float:
        """Progress as fraction (0.0 to 1.0)."""
        total = self.end_idx - self.start_idx
        if total == 0:
            return 1.0
        return (self.current_idx - self.start_idx) / total


# =============================================================================
# Strategy Runner
# =============================================================================

@dataclass
class StrategySignal:
    """Signal output from a strategy."""
    direction: Optional[str] = None  # 'LONG', 'SHORT', or None
    confidence: float = 0.0          # 0.0 to 1.0
    setup_detected: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class StrategyRunner:
    """
    Wrapper for strategy/model execution.
    
    Takes current bar + history, returns signal.
    """
    
    def __init__(self, strategy_name: str, config: Dict[str, Any] = None):
        """
        Initialize strategy runner.
        
        Args:
            strategy_name: Name of strategy (e.g., 'ifvg_cnn', 'always_long')
            config: Strategy configuration
        """
        self.strategy_name = strategy_name
        self.config = config or {}
        self.model = None
        self._setup_strategy()
    
    def _setup_strategy(self):
        """Setup the strategy/model."""
        # For now, placeholder - will integrate with model registry
        if self.strategy_name == "always_long":
            # Simple test strategy
            self.model = lambda bar, history: StrategySignal(
                direction='LONG',
                confidence=1.0,
                setup_detected=True
            )
        elif self.strategy_name == "random":
            # Random strategy for testing
            import random
            self.model = lambda bar, history: StrategySignal(
                direction='LONG' if random.random() > 0.5 else None,
                confidence=random.random(),
                setup_detected=random.random() > 0.8
            )
        else:
            # Default: no signals
            self.model = lambda bar, history: StrategySignal()
    
    def evaluate(self, bar: pd.Series, history: pd.DataFrame) -> StrategySignal:
        """
        Evaluate strategy on current bar.
        
        Args:
            bar: Current bar
            history: Historical bars up to (but not including) current bar
            
        Returns:
            StrategySignal with direction, confidence, etc.
        """
        if self.model is None:
            return StrategySignal()
        
        try:
            return self.model(bar, history)
        except Exception as e:
            print(f"Strategy error: {e}")
            return StrategySignal()


# =============================================================================
# Order Management System (OMS)
# =============================================================================

@dataclass
class Position:
    """Active position."""
    position_id: str
    direction: str          # 'LONG' or 'SHORT'
    size: int
    entry_price: float
    entry_bar: int
    entry_time: pd.Timestamp
    stop_price: Optional[float] = None
    tp_price: Optional[float] = None
    current_pnl: float = 0.0
    mae: float = 0.0  # Max Adverse Excursion
    mfe: float = 0.0  # Max Favorable Excursion


class OrderManagementSystem:
    """
    Order Management System (OMS).
    
    Maintains:
    - Active orders (pending limit/stop orders)
    - Open positions
    - OCO brackets
    
    On each bar:
    - Checks if orders should fill
    - Updates position P&L
    - Checks if stops/TPs are hit
    """
    
    def __init__(
        self,
        costs: CostModel = None,
        fill_engine: BarFillEngine = None
    ):
        """
        Initialize OMS.
        
        Args:
            costs: Cost model for slippage/commissions
            fill_engine: Bar fill engine for realistic fills
        """
        self.costs = costs or DEFAULT_COSTS
        self.fill_engine = fill_engine or DEFAULT_FILL_ENGINE
        
        # Active orders
        self.pending_orders: List[Order] = []
        
        # Open positions
        self.open_positions: List[Position] = []
        
        # OCO brackets
        self.active_ocos: List[OCOBracket] = []
        
        # Tracking
        self._order_counter = 0
        self._position_counter = 0
        self._oco_counter = 0
        
        # History
        self.filled_orders: List[Fill] = []
        self.closed_positions: List[Position] = []
        self.completed_ocos: List[OCOBracket] = []
    
    def submit_order(self, order: Order) -> str:
        """
        Submit a new order.
        
        Returns:
            Order ID
        """
        self._order_counter += 1
        order.order_id = f"ord_{self._order_counter}"
        self.pending_orders.append(order)
        return order.order_id
    
    def submit_oco(
        self,
        config: OCOConfig,
        base_price: float,
        atr: float,
        reference_value: Optional[float] = None,
        bar_idx: int = 0
    ) -> str:
        """
        Submit OCO bracket order.
        
        Returns:
            OCO ID
        """
        bracket = create_oco_bracket(config, base_price, atr, reference_value, self.costs)
        bracket.entry_bar = bar_idx
        
        self._oco_counter += 1
        oco_id = f"oco_{self._oco_counter}"
        bracket.config.name = oco_id
        
        self.active_ocos.append(bracket)
        return oco_id
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        for order in self.pending_orders:
            if order.order_id == order_id:
                order.status = OrderStatus.CANCELLED
                self.pending_orders.remove(order)
                return True
        return False
    
    def process_bar(self, bar: pd.Series, bar_idx: int) -> List[str]:
        """
        Process a bar through the OMS.
        
        Returns list of events:
        - 'ORDER_FILLED:<order_id>'
        - 'POSITION_OPENED:<pos_id>'
        - 'POSITION_CLOSED:<pos_id>:<reason>'
        - 'OCO_ENTRY:<oco_id>'
        - 'OCO_SL:<oco_id>'
        - 'OCO_TP:<oco_id>'
        """
        events = []
        
        # 1. Process pending orders
        to_remove = []
        for order in self.pending_orders:
            fill = process_order(order, bar, bar_idx, self.costs)
            if fill:
                events.append(f"ORDER_FILLED:{order.order_id}")
                self.filled_orders.append(fill)
                to_remove.append(order)
                
                # Create position from fill
                pos_id = self._create_position_from_fill(fill, bar)
                if pos_id:
                    events.append(f"POSITION_OPENED:{pos_id}")
        
        # Remove filled/expired orders
        for order in to_remove:
            if order in self.pending_orders:
                self.pending_orders.remove(order)
        
        # 2. Process OCO brackets
        to_remove_oco = []
        for oco in self.active_ocos:
            bracket, event = process_oco_bar(oco, bar, bar_idx, self.fill_engine)
            
            if event == 'ENTRY':
                events.append(f"OCO_ENTRY:{bracket.config.name}")
                # Create position from OCO entry
                if bracket.entry_fill:
                    pos_id = self._create_position_from_oco(bracket, bar)
                    if pos_id:
                        events.append(f"POSITION_OPENED:{pos_id}")
            
            elif event in ['SL', 'TP', 'TIMEOUT']:
                events.append(f"OCO_{event}:{bracket.config.name}")
                # Close position
                self._close_position_from_oco(bracket, bar, event)
                to_remove_oco.append(bracket)
                self.completed_ocos.append(bracket)
        
        # Remove completed OCOs
        for oco in to_remove_oco:
            if oco in self.active_ocos:
                self.active_ocos.remove(oco)
        
        # 3. Update open positions P&L
        current_price = float(bar['close'])
        for pos in self.open_positions:
            self._update_position_pnl(pos, bar, current_price)
        
        return events
    
    def _create_position_from_fill(self, fill: Fill, bar: pd.Series) -> Optional[str]:
        """Create position from order fill."""
        self._position_counter += 1
        pos_id = f"pos_{self._position_counter}"
        
        position = Position(
            position_id=pos_id,
            direction=fill.direction,
            size=fill.size,
            entry_price=fill.fill_price,
            entry_bar=fill.fill_bar,
            entry_time=bar['time'] if 'time' in bar else pd.Timestamp.now()
        )
        
        self.open_positions.append(position)
        return pos_id
    
    def _create_position_from_oco(self, bracket: OCOBracket, bar: pd.Series) -> Optional[str]:
        """Create position from OCO entry."""
        if not bracket.entry_fill:
            return None
        
        self._position_counter += 1
        pos_id = f"pos_{self._position_counter}"
        
        position = Position(
            position_id=pos_id,
            direction=bracket.config.direction,
            size=1,
            entry_price=bracket.entry_price,
            entry_bar=bracket.entry_bar,
            entry_time=bar['time'] if 'time' in bar else pd.Timestamp.now(),
            stop_price=bracket.stop_price,
            tp_price=bracket.tp_price
        )
        
        self.open_positions.append(position)
        return pos_id
    
    def _close_position_from_oco(self, bracket: OCOBracket, bar: pd.Series, reason: str):
        """Close position associated with OCO."""
        # Find and close matching position
        for pos in self.open_positions[:]:
            if (pos.entry_price == bracket.entry_price and 
                pos.entry_bar == bracket.entry_bar):
                
                # Calculate final P&L
                exit_price = bracket.exit_fill.fill_price if bracket.exit_fill else float(bar['close'])
                if pos.direction == 'LONG':
                    pnl = exit_price - pos.entry_price
                else:
                    pnl = pos.entry_price - exit_price
                
                pos.current_pnl = pnl * pos.size
                
                self.open_positions.remove(pos)
                self.closed_positions.append(pos)
                break
    
    def _update_position_pnl(self, position: Position, bar: pd.Series, current_price: float):
        """Update position P&L and MAE/MFE."""
        if position.direction == 'LONG':
            pnl = current_price - position.entry_price
            adverse = position.entry_price - float(bar['low'])
            favorable = float(bar['high']) - position.entry_price
        else:
            pnl = position.entry_price - current_price
            adverse = float(bar['high']) - position.entry_price
            favorable = position.entry_price - float(bar['low'])
        
        position.current_pnl = pnl * position.size
        position.mae = max(position.mae, adverse * position.size)
        position.mfe = max(position.mfe, favorable * position.size)
    
    def get_state(self) -> Dict[str, Any]:
        """Get current OMS state."""
        return {
            'pending_orders': [
                {
                    'order_id': o.order_id,
                    'type': o.order_type.value,
                    'direction': o.direction,
                    'price': o.price,
                    'status': o.status.value
                }
                for o in self.pending_orders
            ],
            'open_positions': [
                {
                    'position_id': p.position_id,
                    'direction': p.direction,
                    'size': p.size,
                    'entry_price': p.entry_price,
                    'stop_price': p.stop_price,
                    'tp_price': p.tp_price,
                    'current_pnl': p.current_pnl,
                    'mae': p.mae,
                    'mfe': p.mfe
                }
                for p in self.open_positions
            ],
            'active_ocos': [
                {
                    'name': oco.config.name,
                    'status': oco.status.value,
                    'entry_price': oco.entry_price,
                    'stop_price': oco.stop_price,
                    'tp_price': oco.tp_price,
                    'bars_in_trade': oco.bars_in_trade
                }
                for oco in self.active_ocos
            ]
        }


# =============================================================================
# Simulation Engine
# =============================================================================

class SimulationEngine:
    """
    Main simulation engine.
    
    Coordinates:
    - DataStream (bars)
    - StrategyRunner (signals)
    - OMS (orders/positions)
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        strategy_name: str,
        config: Dict[str, Any] = None,
        start_idx: int = 0,
        end_idx: Optional[int] = None
    ):
        """
        Initialize simulation engine.
        
        Args:
            df: OHLCV DataFrame
            strategy_name: Strategy to run
            config: Strategy configuration
            start_idx: Starting bar index
            end_idx: Ending bar index
        """
        self.df = df
        self.stream = DataStream(df, start_idx, end_idx)
        self.strategy = StrategyRunner(strategy_name, config)
        self.oms = OrderManagementSystem()
        
        self.current_bar_idx: Optional[int] = None
        self.current_bar: Optional[pd.Series] = None
        
        # Configuration
        self.config = config or {}
        self.auto_submit_ocos = self.config.get('auto_submit_ocos', True)
        self.oco_config = self._build_oco_config()
    
    def _build_oco_config(self) -> OCOConfig:
        """Build OCO config from engine config."""
        return OCOConfig(
            direction=self.config.get('direction', 'LONG'),
            entry_type=self.config.get('entry_type', 'MARKET'),
            entry_offset_atr=self.config.get('entry_offset_atr', 0.25),
            stop_atr=self.config.get('stop_atr', 1.0),
            tp_multiple=self.config.get('tp_multiple', 1.4),
            max_bars=self.config.get('max_bars', 200)
        )
    
    def step(self) -> Dict[str, Any]:
        """
        Step forward one bar.
        
        Returns dict with:
        - bar: Current bar data
        - events: List of event strings
        - oms_state: Current OMS state
        - signal: Strategy signal
        """
        try:
            bar_idx, bar = next(self.stream)
        except StopIteration:
            return {'done': True, 'events': []}
        
        self.current_bar_idx = bar_idx
        self.current_bar = bar
        
        events = []
        
        # 1. Process OMS (check fills, update positions)
        oms_events = self.oms.process_bar(bar, bar_idx)
        events.extend(oms_events)
        
        # 2. Get history for strategy
        history = self.df.iloc[:bar_idx] if bar_idx > 0 else pd.DataFrame()
        
        # 3. Evaluate strategy
        signal = self.strategy.evaluate(bar, history)
        
        # 4. Act on signal (if auto-submit enabled)
        if self.auto_submit_ocos and signal.setup_detected and signal.direction:
            # Calculate ATR (simple approximation)
            atr = self._calculate_atr(history, window=14)
            
            # Update OCO config direction
            self.oco_config.direction = signal.direction
            
            # Submit OCO
            oco_id = self.oms.submit_oco(
                self.oco_config,
                float(bar['close']),
                atr,
                bar_idx=bar_idx
            )
            events.append(f"OCO_SUBMITTED:{oco_id}")
        
        return {
            'done': False,
            'bar_idx': bar_idx,
            'bar': {
                'time': bar['time'].isoformat() if 'time' in bar else '',
                'open': float(bar['open']),
                'high': float(bar['high']),
                'low': float(bar['low']),
                'close': float(bar['close']),
                'volume': float(bar['volume']) if 'volume' in bar else 0
            },
            'events': events,
            'signal': {
                'direction': signal.direction,
                'confidence': signal.confidence,
                'setup_detected': signal.setup_detected,
                'metadata': signal.metadata
            },
            'oms_state': self.oms.get_state(),
            'progress': self.stream.progress
        }
    
    def _calculate_atr(self, history: pd.DataFrame, window: int = 14) -> float:
        """Calculate ATR from historical data."""
        if len(history) < window:
            # Fallback: simple range
            if len(history) > 0:
                return float(history['high'].iloc[-1] - history['low'].iloc[-1])
            return 10.0  # Default fallback
        
        # True Range
        high_low = history['high'] - history['low']
        high_close = abs(history['high'] - history['close'].shift(1))
        low_close = abs(history['low'] - history['close'].shift(1))
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean().iloc[-1]
        
        return float(atr) if pd.notna(atr) else 10.0
    
    def update_params(self, params: Dict[str, Any]):
        """Update execution parameters mid-simulation."""
        if 'entry_type' in params:
            self.oco_config.entry_type = params['entry_type']
        if 'stop_atr' in params:
            self.oco_config.stop_atr = params['stop_atr']
        if 'tp_multiple' in params:
            self.oco_config.tp_multiple = params['tp_multiple']
        if 'auto_submit_ocos' in params:
            self.auto_submit_ocos = params['auto_submit_ocos']
        
        # Update config dict
        self.config.update(params)
    
    def get_state(self) -> Dict[str, Any]:
        """Get complete engine state."""
        return {
            'current_bar_idx': self.current_bar_idx,
            'progress': self.stream.progress,
            'config': self.config,
            'oms': self.oms.get_state(),
            'stats': {
                'total_ocos': len(self.oms.completed_ocos) + len(self.oms.active_ocos),
                'completed_ocos': len(self.oms.completed_ocos),
                'active_ocos': len(self.oms.active_ocos),
                'open_positions': len(self.oms.open_positions),
                'closed_positions': len(self.oms.closed_positions)
            }
        }

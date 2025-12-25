"""
Market Session
Backend-owned bar-by-bar stepping with events.

This is the unified session manager for both historical (backtest)
and live (streaming) market data. Instead of the frontend doing stepping,
the backend owns:
- MarketStepper (historical or live)
- Indicator pipeline
- Policies/models
- Accounts
- OCO engine

Frontend becomes: renderer + controls + config UI
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Iterator
from enum import Enum
import pandas as pd
from pathlib import Path

from src.sim.causal_runner import CausalExecutor, StepResult
from src.sim.stepper import MarketStepper
from src.sim.account_manager import AccountManager
from src.policy.scanners import Scanner
from src.features.pipeline import FeatureConfig
from src.core.enums import RunMode
from src.core.registries import IndicatorSeries, IndicatorRegistry


class SimEventType(Enum):
    """Types of events during simulation."""
    BAR = "BAR"                          # New bar arrived
    INDICATORS = "INDICATORS"            # Indicators computed
    DECISION = "DECISION"                # Decision point triggered
    ORDER_SUBMIT = "ORDER_SUBMIT"        # Order submitted
    FILL = "FILL"                        # Order filled
    POSITION_OPEN = "POSITION_OPEN"      # Position opened
    POSITION_CLOSE = "POSITION_CLOSE"    # Position closed
    ACCOUNT_UPDATE = "ACCOUNT_UPDATE"    # Account state changed
    SESSION_START = "SESSION_START"      # Session started
    SESSION_END = "SESSION_END"          # Session ended


@dataclass
class SimEvent:
    """Single event during simulation."""
    type: SimEventType
    timestamp: pd.Timestamp
    bar_idx: int
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type.value,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'bar_idx': self.bar_idx,
            'data': self.data,
        }


class MarketSession:
    """
    Backend market session - the unified stepping engine.
    
    Owns all session state:
    - Market stepper (historical or live)
    - Indicator cache
    - Active accounts
    - Active policies (scanners/models)
    - Emits structured events
    
    Frontend subscribes to events via SSE.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        session_id: str = "default",
        start_idx: int = 0,
        end_idx: Optional[int] = None,
    ):
        """
        Initialize market session.
        
        Args:
            df: Full OHLCV DataFrame
            session_id: Unique session identifier
            start_idx: Starting bar index
            end_idx: Ending bar index (None = end of df)
        """
        self.session_id = session_id
        self.df = df
        self.start_idx = start_idx
        self.end_idx = end_idx or len(df)
        
        # Market stepper
        self.stepper = MarketStepper(df, start_idx=start_idx, end_idx=self.end_idx)
        
        # Account manager
        self.account_manager = AccountManager()
        
        # Executor (Lazy init in start or setup)
        self.executor: Optional[CausalExecutor] = None
        
        # Strategy components
        self.scanner: Optional[Scanner] = None
        self.scanner_config: Dict[str, Any] = {}
        self.feature_config: FeatureConfig = FeatureConfig()
        
        # Data caches
        self.df_5m = None
        self.df_15m = None
        
        # Event log
        self.events: List[SimEvent] = []
        
        # Session state
        self.is_running = False
        self.is_paused = False
        self.current_bar_idx: Optional[int] = None
        self.current_timestamp: Optional[pd.Timestamp] = None
        
    def setup_strategy(self, scanner: Scanner, feature_config: FeatureConfig, df_5m=None, df_15m=None):
        """Configure strategy for the session."""
        self.scanner = scanner
        self.feature_config = feature_config
        self.df_5m = df_5m
        self.df_15m = df_15m
    
    def add_account(self, account_id: str, starting_balance: float = 50000.0):
        """Add an account to the session."""
        self.account_manager.create_account(account_id, starting_balance)
    
    def enable_indicator(self, indicator_id: str):
        """Enable an indicator for computation."""
        if indicator_id not in self.enabled_indicators:
            self.enabled_indicators.append(indicator_id)
    
    def disable_indicator(self, indicator_id: str):
        """Disable an indicator."""
        if indicator_id in self.enabled_indicators:
            self.enabled_indicators.remove(indicator_id)
    
    def add_scanner(self, scanner: Any):
        """Add a scanner to active scanners."""
        self.active_scanners.append(scanner)
    
    def add_model(self, model: Any):
        """Add a model to active models."""
        self.active_models.append(model)
    
    def start(self):
        """Start the session."""
        self.is_running = True
        self.is_paused = False
        
        # Initialize executor if needed
        if not self.executor:
            self.executor = CausalExecutor(
                df=self.df,
                stepper=self.stepper,
                account_manager=self.account_manager,
                scanner=self.scanner,
                feature_config=self.feature_config,
                df_5m=self.df_5m,
                df_15m=self.df_15m
            )
        
        # Emit session start event
        
        # Emit session start event
        event = SimEvent(
            type=SimEventType.SESSION_START,
            timestamp=self.df.iloc[self.start_idx]['time'],
            bar_idx=self.start_idx,
            data={
                'session_id': self.session_id,
                'accounts': self.account_manager.list_accounts(),
                'indicators': self.enabled_indicators,
            }
        )
        self.events.append(event)
        return event
    
    def stop(self):
        """Stop the session."""
        self.is_running = False
        
        # Emit session end event
        if self.current_timestamp and self.current_bar_idx is not None:
            event = SimEvent(
                type=SimEventType.SESSION_END,
                timestamp=self.current_timestamp,
                bar_idx=self.current_bar_idx,
                data={
                    'session_id': self.session_id,
                    'total_events': len(self.events),
                    'stats': self.account_manager.get_aggregate_stats(),
                }
            )
            self.events.append(event)
            return event
    
    def pause(self):
        """Pause the session."""
        self.is_paused = True
    
    def resume(self):
        """Resume the session."""
        self.is_paused = False
    
    def step_once(self) -> Optional[List[SimEvent]]:
        """
        Step forward by one bar using CausalExecutor.
        """
        if not self.is_running or self.is_paused or not self.executor:
            return None
        
        result = self.executor.step()
        if not result:
            return None
        
        self.current_bar_idx = result.bar_idx
        self.current_timestamp = result.timestamp
        
        events = []
        
        # 1. BAR event
        bar_event = SimEvent(
            type=SimEventType.BAR,
            timestamp=result.timestamp,
            bar_idx=result.bar_idx,
            data={
                'open': float(result.bar['open']),
                'high': float(result.bar['high']),
                'low': float(result.bar['low']),
                'close': float(result.bar['close']),
                'volume': float(result.bar['volume']),
            }
        )
        events.append(bar_event)
        
        # 2. Fills (Exits/Entries)
        for bracket, event_type in result.fills:
            sim_type = SimEventType(event_type) if event_type in [e.value for e in SimEventType] else SimEventType.FILL
            events.append(SimEvent(
                type=sim_type,
                timestamp=result.timestamp,
                bar_idx=result.bar_idx,
                data={'bracket_id': id(bracket), 'status': bracket.status.value, 'event': event_type}
            ))

        # 3. New Orders (Decisions)
        for bracket in result.new_orders:
            # Emit Decision
            events.append(SimEvent(
                type=SimEventType.DECISION,
                timestamp=result.timestamp,
                bar_idx=result.bar_idx,
                data={
                    'scanner_id': self.scanner.__class__.__name__ if self.scanner else "unknown",
                    'triggered': True,
                    'price': bracket.entry_price, # Use bracket price which is set
                    'atr': bracket.atr_at_creation
                }
            ))
            # Emit Order Submit
            events.append(SimEvent(
                type=SimEventType.ORDER_SUBMIT,
                timestamp=result.timestamp,
                bar_idx=result.bar_idx,
                data=bracket.to_flat_dict()
            ))

        # 4. Account Updates
        for acc_id, snapshot in result.account_snapshots.items():
            events.append(SimEvent(
                type=SimEventType.ACCOUNT_UPDATE,
                timestamp=result.timestamp,
                bar_idx=result.bar_idx,
                data=snapshot.to_dict()
            ))
        
        # Store events
        self.events.extend(events)
        return events
    
    def play(self) -> Iterator[SimEvent]:
        """
        Play through the session, yielding events.
        
        This is the main simulation loop for SSE streaming.
        """
        self.start()
        
        while self.is_running:
            # Wait if paused
            if self.is_paused:
                break
            
            events = self.step_once()
            if not events:
                # Session is done
                break
            
            for event in events:
                yield event
        
        # Emit end event
        end_event = self.stop()
        if end_event:
            yield end_event
    
    def get_state(self) -> Dict[str, Any]:
        """Get current session state."""
        return {
            'session_id': self.session_id,
            'is_running': self.is_running,
            'is_paused': self.is_paused,
            'current_bar_idx': self.current_bar_idx,
            'current_timestamp': self.current_timestamp.isoformat() if self.current_timestamp else None,
            'total_events': len(self.events),
            'accounts': self.account_manager.list_accounts(),
            'enabled_indicators': self.enabled_indicators,
            'stats': self.account_manager.get_aggregate_stats(),
        }

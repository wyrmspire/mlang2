"""
Simulation Session
Backend-owned simulation stepping with events.

This is the backend counterpart to the frontend SimulationView.
Instead of the frontend doing stepping, the backend owns:
- MarketStepper
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

from src.sim.stepper import MarketStepper
from src.sim.account_manager import AccountManager
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


class SimulationSession:
    """
    Backend simulation session.
    
    Owns all simulation state:
    - Market stepper
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
        Initialize simulation session.
        
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
        
        # Indicators (computed on demand, cached)
        self.indicator_cache: Dict[str, IndicatorSeries] = {}
        self.enabled_indicators: List[str] = []
        
        # Policies (scanners/models)
        self.active_scanners: List[Any] = []
        self.active_models: List[Any] = []
        
        # Event log
        self.events: List[SimEvent] = []
        
        # Session state
        self.is_running = False
        self.is_paused = False
        self.current_bar_idx: Optional[int] = None
        self.current_timestamp: Optional[pd.Timestamp] = None
    
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
        Step forward by one bar.
        
        Returns list of events generated by this step.
        """
        if not self.is_running or self.is_paused:
            return None
        
        step = self.stepper.step()
        if step.is_done:
            return None
        
        self.current_bar_idx = step.bar_idx
        self.current_timestamp = step.bar['time']
        
        events = []
        
        # 1. BAR event
        bar_event = SimEvent(
            type=SimEventType.BAR,
            timestamp=self.current_timestamp,
            bar_idx=self.current_bar_idx,
            data={
                'open': float(step.bar['open']),
                'high': float(step.bar['high']),
                'low': float(step.bar['low']),
                'close': float(step.bar['close']),
                'volume': float(step.bar['volume']),
            }
        )
        events.append(bar_event)
        
        # 2. INDICATORS event (if any enabled)
        if self.enabled_indicators:
            # Compute indicators (in real implementation, would use IndicatorRegistry)
            indicators_data = {}
            for ind_id in self.enabled_indicators:
                # Placeholder - real implementation would compute
                indicators_data[ind_id] = None
            
            ind_event = SimEvent(
                type=SimEventType.INDICATORS,
                timestamp=self.current_timestamp,
                bar_idx=self.current_bar_idx,
                data={'indicators': indicators_data}
            )
            events.append(ind_event)
        
        # 3. Check scanners/policies for DECISION events
        # (Placeholder - real implementation would run scanners)
        
        # 4. Update accounts with current price
        current_price = float(step.bar['close'])
        for account_id in self.account_manager.list_accounts():
            snapshot = self.account_manager.take_snapshot(
                account_id,
                current_price,
                self.current_timestamp
            )
            if snapshot:
                acc_event = SimEvent(
                    type=SimEventType.ACCOUNT_UPDATE,
                    timestamp=self.current_timestamp,
                    bar_idx=self.current_bar_idx,
                    data=snapshot.to_dict()
                )
                events.append(acc_event)
        
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

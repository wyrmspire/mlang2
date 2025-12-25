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

from src.sim.stepper import MarketStepper
from src.sim.account_manager import AccountManager
from src.sim.oco_engine import OCOEngine, OCOStatus
from src.sim.sizing import calculate_contracts, calculate_pnl_dollars
from src.features.pipeline import compute_features, FeatureConfig, precompute_indicators
from src.policy.scanners import Scanner, ScanResult
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
        
        # OCO Engine
        self.oco_engine = OCOEngine()
        self.active_brackets = []  # List[OCOBracket]
        
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
        
        # 2. Update active OCO brackets (Exits/Fills)
        # This MUST happen before new entries to free up capital/slots
        completed_brackets = []
        for bracket in self.active_brackets:
            updated_bracket, event_type = self.oco_engine.process_bar(
                bracket, step.bar, self.current_bar_idx
            )
            
            if event_type:
                # Emit event
                events.append(SimEvent(
                    type=SimEventType(event_type) if event_type in [e.value for e in SimEventType] else SimEventType.FILL,
                    timestamp=self.current_timestamp,
                    bar_idx=self.current_bar_idx,
                    data={'bracket_id': id(bracket), 'status': bracket.status.value, 'event': event_type}
                ))
                
                # Check for completion
                if bracket.status in [OCOStatus.CLOSED_TP, OCOStatus.CLOSED_SL, OCOStatus.CLOSED_TIMEOUT, OCOStatus.CANCELLED]:
                    completed_brackets.append(bracket)
                    
                    # Update Account
                    if bracket.exit_fill:
                        # Calculate PnL
                        entry_price = bracket.entry_price
                        exit_price = bracket.exit_fill.fill_price
                        direction = bracket.config.direction
                        
                        contracts = getattr(bracket, 'contracts', 1) 
                        
                        # Use AccountManager to handle the close
                        # We need to find the specific position. 
                        # Limitation: The current AccountManager tracks strict FIFO/Position objects.
                        # OCOBracket tracks a "Trade" lifecycle.
                        # We need to bridge them.
                        
                        # For this refactor, we will simpler update the default account balance
                        # directly to ensure stats work, assuming single account 'default'.
                        default_account = self.account_manager.get_account('default')
                        if default_account:
                            # We construct a fill and 'close_position' on the account.
                            # But wait, did we 'open' it on the account? 
                            # We missed the ENTRY fill event in the loop above!
                            pass # Handled below in separate check
                            
                # Check for Entry Fill strictly (State transition PENDING -> ACTIVE)
                if event_type == 'ENTRY':
                     default_account = self.account_manager.get_account('default')
                     if default_account:
                         # Register the position
                         # We need the contracts/size.
                         contracts = getattr(bracket, 'contracts', 1) 
                         
                         default_account.open_position(
                             fill=bracket.entry_fill,
                             stop_loss=bracket.stop_price,
                             take_profit=bracket.tp_price,
                             time=self.current_timestamp
                         )
                
                # Check for Exit Fill (State transition ACTIVE -> CLOSED_*)
                if bracket.exit_fill and bracket.status in [OCOStatus.CLOSED_TP, OCOStatus.CLOSED_SL, OCOStatus.CLOSED_TIMEOUT]:
                    default_account = self.account_manager.get_account('default')
                    if default_account:
                        # Find matching position (Approximation for now: assume last or matching direction)
                        # AccountManager.close_position wants the Position object.
                        # We'll search for one matching entry price/time.
                        
                        # Robust matching
                        matching_pos = None
                        for pos in default_account.positions:
                            if pos.entry_price == bracket.entry_price and pos.direction == bracket.config.direction:
                                matching_pos = pos
                                break
                        
                        if matching_pos:
                            default_account.close_position(
                                position=matching_pos,
                                fill=bracket.exit_fill,
                                outcome=bracket._get_outcome(),
                                time=self.current_timestamp
                            )

        
        # Remove completed
        for b in completed_brackets:
            self.active_brackets.remove(b)

        # 3. Strategy / Scanner (Entries)
        if self.scanner:
            features = compute_features(
                self.stepper, 
                self.feature_config, 
                df_5m=self.df_5m, 
                df_15m=self.df_15m
            )
            
            scan_result = self.scanner.scan(None, features) # MarketState None for now
            
            if scan_result.triggered:
                # Create Decision Event
                events.append(SimEvent(
                    type=SimEventType.DECISION,
                    timestamp=self.current_timestamp,
                    bar_idx=self.current_bar_idx,
                    data={
                        'scanner_id': self.scanner.__class__.__name__,
                        'triggered': True,
                        'price': features.current_price,
                        'atr': features.atr
                    }
                ))
                
                # Create Bracket from ScanResult
                # We need a proper OCOConfig. In a real app, this comes from the Strategy class.
                # For now, we construct one based on the scanner's direction or defaults.
                # Attempt to get direction from result or feature set
                direction = getattr(scan_result, 'direction', "LONG")
                
                # Construct OCO Config (Defaults if not provided by strategy)
                # TODO: Retrieve this from self.strategy_config or similar
                from src.sim.oco_engine import OCOConfig
                oco_config = OCOConfig(
                    direction=direction,
                    entry_type="MARKET", # Default to Market for immediate entry if triggered? Or Limit?
                    stop_atr=2.0,       # Default
                    tp_multiple=2.0,    # Default
                    name=f"Sim_{self.scanner.__class__.__name__}"
                )
                
                # Create and register the bracket
                bracket = self.oco_engine.create_bracket(
                    config=oco_config,
                    base_price=features.current_price,
                    atr=features.atr,
                    current_idx=self.current_bar_idx
                )
                
                # Note: create_bracket calculates prices but doesn't "place" it unless we track it
                # We add to active_brackets. 
                # Ideally, we should have an 'Order' abstraction, but OCOBracket handles the lifecycle well enough here.
                # However, OCOEngine.create_bracket returns a bracket in PENDING_ENTRY state.
                self.active_brackets.append(bracket)
                
                # Emit OCO Created Event
                events.append(SimEvent(
                    type=SimEventType.ORDER_SUBMIT, # Logic maps creation to submission
                    timestamp=self.current_timestamp,
                    bar_idx=self.current_bar_idx,
                    data=bracket.to_flat_dict()
                ))

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

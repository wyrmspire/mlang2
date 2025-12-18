"""
Replay Session
Real-time simulation of historical data with strict causality.

This module enables:
- Simulated real-time stepping through historical data
- Model/policy triggering at each bar
- Event streaming (decisions, orders, fills, exits)
- Agent speed/pause/resume control
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Iterator
from enum import Enum
import pandas as pd
from pathlib import Path

from src.sim.stepper import MarketStepper, StepResult
from src.experiments.config import ReplayConfig, RunMode
from src.policy.actions import Action, SkipReason


class ReplayEventType(Enum):
    """Types of events during replay."""
    BAR_UPDATE = "BAR_UPDATE"           # New bar arrived
    DECISION = "DECISION"                # Decision point triggered
    ORDER_PLACED = "ORDER_PLACED"        # Order placed
    ORDER_FILLED = "ORDER_FILLED"        # Order filled
    OCO_UPDATE = "OCO_UPDATE"            # OCO bracket updated
    EXIT = "EXIT"                        # Position exited
    TIMEOUT = "TIMEOUT"                  # OCO timed out


@dataclass
class ReplayEvent:
    """Single event during replay."""
    type: ReplayEventType
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


class ReplaySession:
    """
    Replay session manager.
    
    Enables simulated real-time stepping through historical data
    with model/policy evaluation at each bar.
    
    Usage:
        session = ReplaySession(df, config)
        for event in session.play():
            # Process event
            if event.type == ReplayEventType.DECISION:
                # Handle decision point
                pass
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        config: ReplayConfig,
        run_mode: RunMode = RunMode.REPLAY
    ):
        """
        Initialize replay session.
        
        Args:
            df: Full OHLCV DataFrame to replay
            config: Replay configuration
            run_mode: Should always be REPLAY for safety
        """
        if run_mode != RunMode.REPLAY:
            raise ValueError("ReplaySession must use RunMode.REPLAY")
        
        self.df = df
        self.config = config
        self.run_mode = run_mode
        
        # Initialize stepper
        start_idx = config.start_bar
        end_idx = config.end_bar if config.end_bar else len(df)
        self.stepper = MarketStepper(df, start_idx=start_idx, end_idx=end_idx)
        
        # Replay state
        self.current_bar_idx: Optional[int] = None
        self.current_timestamp: Optional[pd.Timestamp] = None
        self.is_playing = config.auto_play
        self.is_paused = False
        self.events: List[ReplayEvent] = []
        
        # Position tracking
        self.in_position = False
        self.position_entry_bar: Optional[int] = None
        self.current_oco: Optional[Dict[str, Any]] = None
    
    def reset(self):
        """Reset session to beginning."""
        start_idx = self.config.start_bar
        end_idx = self.config.end_bar if self.config.end_bar else len(self.df)
        self.stepper = MarketStepper(self.df, start_idx=start_idx, end_idx=end_idx)
        self.current_bar_idx = None
        self.current_timestamp = None
        self.events = []
        self.in_position = False
        self.position_entry_bar = None
        self.current_oco = None
    
    def play(self) -> Iterator[ReplayEvent]:
        """
        Play through the session, yielding events.
        
        This is the main replay loop. It steps through each bar and
        yields events as they occur.
        
        Yields:
            ReplayEvent objects
        """
        self.is_playing = True
        
        while not self.stepper.is_done() and self.is_playing:
            # Wait if paused
            while self.is_paused and self.is_playing:
                # In a real implementation, this would check pause state periodically
                # For now, just break if paused
                break
            
            if self.is_paused:
                break
            
            # Step forward
            step = self.stepper.step()
            
            if step.is_done:
                break
            
            self.current_bar_idx = step.bar_idx
            self.current_timestamp = step.current_bar['time']
            
            # Emit bar update event
            bar_event = ReplayEvent(
                type=ReplayEventType.BAR_UPDATE,
                timestamp=self.current_timestamp,
                bar_idx=self.current_bar_idx,
                data={
                    'open': float(step.current_bar['open']),
                    'high': float(step.current_bar['high']),
                    'low': float(step.current_bar['low']),
                    'close': float(step.current_bar['close']),
                    'volume': float(step.current_bar['volume']),
                }
            )
            self.events.append(bar_event)
            yield bar_event
            
            # Check for decision points, orders, fills, etc.
            # This would be integrated with scanner/policy/model in real use
            # For now, this is a skeleton that can be extended
    
    def step_once(self) -> Optional[ReplayEvent]:
        """
        Step forward by one bar (manual control).
        
        Returns:
            The bar update event, or None if done
        """
        if self.stepper.is_done():
            return None
        
        step = self.stepper.step()
        
        if step.is_done:
            return None
        
        self.current_bar_idx = step.bar_idx
        self.current_timestamp = step.current_bar['time']
        
        event = ReplayEvent(
            type=ReplayEventType.BAR_UPDATE,
            timestamp=self.current_timestamp,
            bar_idx=self.current_bar_idx,
            data={
                'open': float(step.current_bar['open']),
                'high': float(step.current_bar['high']),
                'low': float(step.current_bar['low']),
                'close': float(step.current_bar['close']),
                'volume': float(step.current_bar['volume']),
            }
        )
        self.events.append(event)
        return event
    
    def pause(self):
        """Pause playback."""
        self.is_paused = True
    
    def resume(self):
        """Resume playback."""
        self.is_paused = False
    
    def stop(self):
        """Stop playback."""
        self.is_playing = False
    
    def seek(self, bar_idx: int):
        """
        Seek to a specific bar index.
        
        Note: This recreates the stepper at the target position.
        """
        if bar_idx < 0 or bar_idx >= len(self.df):
            raise ValueError(f"Invalid bar index: {bar_idx}")
        
        # Recreate stepper at new position
        end_idx = self.config.end_bar if self.config.end_bar else len(self.df)
        self.stepper = MarketStepper(self.df, start_idx=bar_idx, end_idx=end_idx)
        self.current_bar_idx = bar_idx
        self.current_timestamp = self.df.iloc[bar_idx]['time']
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        Get current replay state.
        
        Returns:
            Dictionary with current state information
        """
        return {
            'bar_idx': self.current_bar_idx,
            'timestamp': self.current_timestamp.isoformat() if self.current_timestamp else None,
            'is_playing': self.is_playing,
            'is_paused': self.is_paused,
            'is_done': self.stepper.is_done(),
            'in_position': self.in_position,
            'total_events': len(self.events),
        }
    
    def get_visible_window(self) -> pd.DataFrame:
        """
        Get the visible window of bars for display.
        
        Returns bars from (current - lookback) to (current + future_bars)
        based on config settings.
        """
        if self.current_bar_idx is None:
            return pd.DataFrame()
        
        # Default lookback
        lookback = 100
        future = self.config.show_future_bars
        
        start_idx = max(0, self.current_bar_idx - lookback)
        end_idx = min(len(self.df), self.current_bar_idx + future + 1)
        
        return self.df.iloc[start_idx:end_idx].copy()

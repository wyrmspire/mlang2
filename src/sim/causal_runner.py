"""
Causal Runner
Unified execution engine for bar-by-bar simulation.

This is the SINGLE SOURCE OF TRUTH for:
1. Stepping through market data
2. Computing features (causal)
3. Running Scanners (Signal Generation)
4. Managing OCO Brackets (Order Lifecycle)
5. Updating Accounts (Fills/PnL)

It is used by:
- MarketSession (for Live/Replay streaming)
- ExperimentRunner (for Backtesting/Training data generation)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
import uuid

from src.sim.stepper import MarketStepper
from src.sim.account_manager import AccountManager
from src.sim.oco_engine import OCOEngine, OCOStatus, OCOBracket, OCOConfig
from src.sim.sizing import calculate_contracts, calculate_pnl_dollars
from src.features.pipeline import compute_features, FeatureConfig
from src.policy.scanners import Scanner, ScanResult
from src.sim.execution import Fill


@dataclass
class StepResult:
    """Result of a single simulation step."""
    bar_idx: int
    timestamp: pd.Timestamp
    bar: pd.Series
    
    # State
    current_price: float
    atr: float
    features: Any  # FeatureBundle
    
    # Events
    scanner_triggers: List[Dict] = field(default_factory=list)
    new_orders: List[OCOBracket] = field(default_factory=list)
    fills: List[Tuple[OCOBracket, str]] = field(default_factory=list)  # (bracket, event_type)
    completed_brackets: List[OCOBracket] = field(default_factory=list)
    
    # Snapshot
    account_snapshots: Dict[str, Any] = field(default_factory=dict)


class CausalExecutor:
    """
    Executes the causal market loop.
    
    Does NOT know about:
    - Future outcomes (labels)
    - Training
    - Visualization/SSE protocols
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        stepper: MarketStepper,
        account_manager: AccountManager,
        scanner: Optional[Scanner] = None,
        feature_config: Optional[FeatureConfig] = None,
        df_5m: Optional[pd.DataFrame] = None,
        df_15m: Optional[pd.DataFrame] = None,
        precomputed_indicators: Optional[Dict[int, Any]] = None,
    ):
        self.df = df
        self.stepper = stepper
        self.account_manager = account_manager
        
        # Strategy (Optional)
        self.scanner = scanner
        self.feature_config = feature_config or FeatureConfig()
        self.df_5m = df_5m
        self.df_15m = df_15m
        self.precomputed_indicators = precomputed_indicators
        
        # Execution Engine
        self.oco_engine = OCOEngine()
        self.active_brackets: List[OCOBracket] = []
        
        # State
        self.current_bar_idx = 0
        self.current_timestamp = None
    
    def step(self) -> Optional[StepResult]:
        """Execute one bar step."""
        step = self.stepper.step()
        if step.is_done:
            return None
        
        self.current_bar_idx = step.bar_idx
        self.current_timestamp = step.bar['time']
        
        current_price = float(step.bar['close'])
        
        result = StepResult(
            bar_idx=self.current_bar_idx,
            timestamp=self.current_timestamp,
            bar=step.bar,
            current_price=current_price,
            atr=0.0, # Filled later
            features=None
        )
        
        # 1. Update Active OCO Brackets (Exits/Fills)
        # ---------------------------------------------------
        # Must run BEFORE entries to clear capital/slots
        completed = []
        for bracket in self.active_brackets:
            updated_bracket, event_type = self.oco_engine.process_bar(
                bracket, step.bar, self.current_bar_idx
            )
            
            if event_type:
                result.fills.append((bracket, event_type))
                
                # Handle ENTRY Fill -> Open Position
                if event_type == 'ENTRY':
                    self._handle_entry(bracket)
                
                # Handle EXIT Fill -> Close Position
                if bracket.status in [OCOStatus.CLOSED_TP, OCOStatus.CLOSED_SL, OCOStatus.CLOSED_TIMEOUT]:
                    self._handle_exit(bracket)
                    completed.append(bracket)
                
                # Handle CANCELLED
                if bracket.status == OCOStatus.CANCELLED:
                    completed.append(bracket)

        # Cleanup completed
        for b in completed:
            self.active_brackets.remove(b)
        result.completed_brackets = completed

        # 2. Strategy / Scanner (Entries)
        # ---------------------------------------------------
        features = None
        if self.scanner:
            features = compute_features(
                self.stepper,
                self.feature_config,
                df_5m=self.df_5m,
                df_15m=self.df_15m,
                precomputed_indicators=self.precomputed_indicators
            )
            result.features = features
            result.atr = features.atr
            
            # Run scan
            # Note: MarketState is passed as None for now, or extracted from features if available
            scan_result = self.scanner.scan(features.market_state, features)
            
            if scan_result.triggered:
                # 3. Signals -> Orders
                # -----------------------------------------------
                
                # Determine direction
                direction = getattr(scan_result, 'direction', "LONG")
                
                # Construct OCO Config
                # TODO: This should come from a Strategy Config object
                oco_config = OCOConfig(
                    direction=direction,
                    entry_type="MARKET", 
                    stop_atr=2.0,
                    tp_multiple=2.0,
                    name=f"Auto_{self.scanner.__class__.__name__}"
                )
                
                # Create Bracket
                bracket = self.oco_engine.create_bracket(
                    config=oco_config,
                    base_price=features.current_price,
                    atr=features.atr,
                    current_idx=self.current_bar_idx
                )
                
                # Calculate Contracts (Sizing) - REQUIRED
                # We default to max risk if not specified
                from src.config import DEFAULT_MAX_RISK_DOLLARS
                sizing = calculate_contracts(
                    entry_price=bracket.entry_price,
                    stop_price=bracket.stop_price,
                    max_risk_dollars=DEFAULT_MAX_RISK_DOLLARS
                )
                
                # Store contracts on the bracket for tracking
                # (Dynamically attaching for now, untyped)
                bracket.contracts = sizing.contracts
                
                # Register
                self.active_brackets.append(bracket)
                
                # Record event
                result.scanner_triggers.append({
                    'scanner': self.scanner.__class__.__name__,
                    'price': features.current_price,
                    'direction': direction
                })
                result.new_orders.append(bracket)

        # 4. Account Updates (Mark-to-Market)
        # ---------------------------------------------------
        for account_id in self.account_manager.list_accounts():
            snapshot = self.account_manager.take_snapshot(
                account_id,
                current_price,
                self.current_timestamp
            )
            if snapshot:
                result.account_snapshots[account_id] = snapshot

        return result

    def _handle_entry(self, bracket: OCOBracket):
        """Register entry fill with Account Manager."""
        # Assume 'default' account for now
        account = self.account_manager.get_account('default')
        if account:
            # Enforce contract size from sizing step
            contracts = getattr(bracket, 'contracts', 1)
            
            # Override fill size in case it was 1 default
            if bracket.entry_fill:
                bracket.entry_fill.size = contracts
            
            account.open_position(
                fill=bracket.entry_fill,
                stop_loss=bracket.stop_price,
                take_profit=bracket.tp_price,
                time=self.current_timestamp
            )

    def _handle_exit(self, bracket: OCOBracket):
        """Register exit fill with Account Manager."""
        account = self.account_manager.get_account('default')
        if account and bracket.exit_fill:
            # Find matching position
            # Robust matching by direction and approximately entry price
            matching_pos = None
            for pos in account.positions:
                if (pos.direction == bracket.config.direction and 
                    abs(pos.entry_price - bracket.entry_price) < 1e-4):
                    matching_pos = pos
                    break
            
            if matching_pos:
                # Ensure fill size matches position
                bracket.exit_fill.size = matching_pos.size
                
                account.close_position(
                    position=matching_pos,
                    fill=bracket.exit_fill,
                    outcome=bracket._get_outcome(),
                    time=self.current_timestamp
                )

# Git Diff Report

**Generated**: Thu, Dec 25, 2025 11:39:54 AM

**Local Branch**: master

**Comparing Against**: origin/master

---

## Uncommitted Changes (working directory)

### Modified/Staged Files

```
 M src/experiments/runner.py
 M src/sim/market_session.py
?? gitrdiff.md
```

### Uncommitted Diff

```diff
diff --git a/src/experiments/runner.py b/src/experiments/runner.py
index 38a2d27..8631c39 100644
--- a/src/experiments/runner.py
+++ b/src/experiments/runner.py
@@ -22,11 +22,13 @@ from src.data.resample import resample_all_timeframes
 
 from src.sim.stepper import MarketStepper
 from src.sim.oco_engine import create_oco_bracket
+from src.sim.sizing import calculate_contracts, calculate_pnl_dollars
 from src.features.pipeline import compute_features, precompute_indicators
 from src.policy.scanners import get_scanner
 from src.policy.filters import DEFAULT_FILTERS
 from src.policy.cooldown import CooldownManager
 from src.policy.actions import Action, SkipReason
+from src.viz.window_utils import enforce_2hour_window
 
 from src.labels.labeler import Labeler
 from src.datasets.decision_record import DecisionRecord
@@ -37,7 +39,7 @@ from src.datasets.reader import create_dataloader
 from src.models.fusion import FusionModel
 from src.models.train import train_model, TrainResult
 
-from src.config import PROCESSED_DIR, SHARDS_DIR, RESULTS_DIR
+from src.config import PROCESSED_DIR, SHARDS_DIR, RESULTS_DIR, DEFAULT_MAX_RISK_DOLLARS
 
 
 @dataclass
@@ -188,11 +190,19 @@ def run_experiment(
         if exporter:
             curr_idx = step.bar_idx
             
-            # Extract RAW OHLCV for chart: 60 bars before + 20 bars after
-            start_raw_idx = max(0, curr_idx - 60)
-            end_raw_idx = min(len(df), curr_idx + 21)  # +1 for current bar, +20 future
-            raw_slice = df.iloc[start_raw_idx : end_raw_idx]
-            raw_ohlcv = raw_slice[['open', 'high', 'low', 'close', 'volume']].values.tolist()
+            exit_time = None
+            if record.action == Action.PLACE_ORDER:
+                exit_time = features.timestamp + pd.Timedelta(minutes=record.cf_bars_held)
+
+            raw_ohlcv, window_warning = enforce_2hour_window(
+                df_1m=df,
+                entry_time=features.timestamp,
+                exit_time=exit_time,
+                bars_held=record.cf_bars_held
+            )
+
+            if window_warning:
+                exporter._window_warnings.append(window_warning)
             
             # Extract future bars separately (for compatibility)
             future_bars = []
@@ -226,7 +236,16 @@ def run_experiment(
                     base_price=features.current_price,
                     atr=features.atr
                 )
-                exporter.on_bracket_created(record.decision_id, bracket)
+                sizing_result = calculate_contracts(
+                    entry_price=bracket.entry_price,
+                    stop_price=bracket.stop_price,
+                    max_risk_dollars=DEFAULT_MAX_RISK_DOLLARS
+                )
+                exporter.on_bracket_created(
+                    record.decision_id,
+                    bracket,
+                    contracts=sizing_result.contracts
+                )
         
         # Update cooldown if trade placed
         if record.action == Action.PLACE_ORDER:
@@ -238,6 +257,26 @@ def run_experiment(
                 exit_bar = step.bar_idx + record.cf_bars_held
                 exit_time = features.timestamp + pd.Timedelta(minutes=record.cf_bars_held)
                 
+                bracket = create_oco_bracket(
+                    config=config.oco_config,
+                    base_price=features.current_price,
+                    atr=features.atr
+                )
+                sizing_result = calculate_contracts(
+                    entry_price=bracket.entry_price,
+                    stop_price=bracket.stop_price,
+                    max_risk_dollars=DEFAULT_MAX_RISK_DOLLARS
+                )
+                exit_price = features.current_price + (
+                    record.cf_pnl / (1 if config.oco_config.direction == "LONG" else -1)
+                )
+                pnl_points, pnl_dollars = calculate_pnl_dollars(
+                    entry_price=features.current_price,
+                    exit_price=exit_price,
+                    direction=config.oco_config.direction,
+                    contracts=sizing_result.contracts
+                )
+
                 trade = TradeRecord(
                     trade_id=str(uuid.uuid4())[:8],
                     decision_id=record.decision_id,
@@ -247,12 +286,12 @@ def run_experiment(
                     direction=config.oco_config.direction,
                     exit_time=exit_time,
                     exit_bar=exit_bar,
-                    exit_price=features.current_price + (record.cf_pnl / (1 if config.oco_config.direction=="LONG" else -1)),
+                    exit_price=exit_price,
                     exit_reason=record.cf_outcome,
                     outcome=record.cf_outcome,
-                    pnl_points=record.cf_pnl,
-                    pnl_dollars=record.cf_pnl_dollars,
-                    r_multiple=record.cf_pnl_dollars / (features.atr * config.oco_config.stop_atr * 50) if features.atr > 0 else 0,
+                    pnl_points=pnl_points,
+                    pnl_dollars=pnl_dollars,
+                    r_multiple=pnl_dollars / (features.atr * config.oco_config.stop_atr * 50) if features.atr > 0 else 0,
                     bars_held=record.cf_bars_held,
                     mae=record.cf_mae,
                     mfe=record.cf_mfe,
diff --git a/src/sim/market_session.py b/src/sim/market_session.py
index c41a214..b56ebf5 100644
--- a/src/sim/market_session.py
+++ b/src/sim/market_session.py
@@ -22,6 +22,10 @@ from pathlib import Path
 
 from src.sim.stepper import MarketStepper
 from src.sim.account_manager import AccountManager
+from src.sim.oco_engine import OCOEngine, OCOStatus
+from src.sim.sizing import calculate_contracts, calculate_pnl_dollars
+from src.features.pipeline import compute_features, FeatureConfig, precompute_indicators
+from src.policy.scanners import Scanner, ScanResult
 from src.core.enums import RunMode
 from src.core.registries import IndicatorSeries, IndicatorRegistry
 
@@ -98,13 +102,18 @@ class MarketSession:
         # Account manager
         self.account_manager = AccountManager()
         
-        # Indicators (computed on demand, cached)
-        self.indicator_cache: Dict[str, IndicatorSeries] = {}
-        self.enabled_indicators: List[str] = []
+        # OCO Engine
+        self.oco_engine = OCOEngine()
+        self.active_brackets = []  # List[OCOBracket]
         
-        # Policies (scanners/models)
-        self.active_scanners: List[Any] = []
-        self.active_models: List[Any] = []
+        # Strategy components
+        self.scanner: Optional[Scanner] = None
+        self.scanner_config: Dict[str, Any] = {}
+        self.feature_config: FeatureConfig = FeatureConfig()
+        
+        # Data caches
+        self.df_5m = None
+        self.df_15m = None
         
         # Event log
         self.events: List[SimEvent] = []
@@ -114,6 +123,13 @@ class MarketSession:
         self.is_paused = False
         self.current_bar_idx: Optional[int] = None
         self.current_timestamp: Optional[pd.Timestamp] = None
+        
+    def setup_strategy(self, scanner: Scanner, feature_config: FeatureConfig, df_5m=None, df_15m=None):
+        """Configure strategy for the session."""
+        self.scanner = scanner
+        self.feature_config = feature_config
+        self.df_5m = df_5m
+        self.df_15m = df_15m
     
     def add_account(self, account_id: str, starting_balance: float = 50000.0):
         """Add an account to the session."""
@@ -216,26 +232,158 @@ class MarketSession:
         )
         events.append(bar_event)
         
-        # 2. INDICATORS event (if any enabled)
-        if self.enabled_indicators:
-            # Compute indicators (in real implementation, would use IndicatorRegistry)
-            indicators_data = {}
-            for ind_id in self.enabled_indicators:
-                # Placeholder - real implementation would compute
-                indicators_data[ind_id] = None
-            
-            ind_event = SimEvent(
-                type=SimEventType.INDICATORS,
-                timestamp=self.current_timestamp,
-                bar_idx=self.current_bar_idx,
-                data={'indicators': indicators_data}
+        # 2. Update active OCO brackets (Exits/Fills)
+        # This MUST happen before new entries to free up capital/slots
+        completed_brackets = []
+        for bracket in self.active_brackets:
+            updated_bracket, event_type = self.oco_engine.process_bar(
+                bracket, step.bar, self.current_bar_idx
             )
-            events.append(ind_event)
-        
-        # 3. Check scanners/policies for DECISION events
-        # (Placeholder - real implementation would run scanners)
+            
+            if event_type:
+                # Emit event
+                events.append(SimEvent(
+                    type=SimEventType(event_type) if event_type in [e.value for e in SimEventType] else SimEventType.FILL,
+                    timestamp=self.current_timestamp,
+                    bar_idx=self.current_bar_idx,
+                    data={'bracket_id': id(bracket), 'status': bracket.status.value, 'event': event_type}
+                ))
+                
+                # Check for completion
+                if bracket.status in [OCOStatus.CLOSED_TP, OCOStatus.CLOSED_SL, OCOStatus.CLOSED_TIMEOUT, OCOStatus.CANCELLED]:
+                    completed_brackets.append(bracket)
+                    
+                    # Update Account
+                    if bracket.exit_fill:
+                        # Calculate PnL
+                        entry_price = bracket.entry_price
+                        exit_price = bracket.exit_fill.fill_price
+                        direction = bracket.config.direction
+                        
+                        contracts = getattr(bracket, 'contracts', 1) 
+                        
+                        # Use AccountManager to handle the close
+                        # We need to find the specific position. 
+                        # Limitation: The current AccountManager tracks strict FIFO/Position objects.
+                        # OCOBracket tracks a "Trade" lifecycle.
+                        # We need to bridge them.
+                        
+                        # For this refactor, we will simpler update the default account balance
+                        # directly to ensure stats work, assuming single account 'default'.
+                        default_account = self.account_manager.get_account('default')
+                        if default_account:
+                            # We construct a fill and 'close_position' on the account.
+                            # But wait, did we 'open' it on the account? 
+                            # We missed the ENTRY fill event in the loop above!
+                            pass # Handled below in separate check
+                            
+                # Check for Entry Fill strictly (State transition PENDING -> ACTIVE)
+                if event_type == 'ENTRY':
+                     default_account = self.account_manager.get_account('default')
+                     if default_account:
+                         # Register the position
+                         # We need the contracts/size.
+                         contracts = getattr(bracket, 'contracts', 1) 
+                         
+                         default_account.open_position(
+                             fill=bracket.entry_fill,
+                             stop_loss=bracket.stop_price,
+                             take_profit=bracket.tp_price,
+                             time=self.current_timestamp
+                         )
+                
+                # Check for Exit Fill (State transition ACTIVE -> CLOSED_*)
+                if bracket.exit_fill and bracket.status in [OCOStatus.CLOSED_TP, OCOStatus.CLOSED_SL, OCOStatus.CLOSED_TIMEOUT]:
+                    default_account = self.account_manager.get_account('default')
+                    if default_account:
+                        # Find matching position (Approximation for now: assume last or matching direction)
+                        # AccountManager.close_position wants the Position object.
+                        # We'll search for one matching entry price/time.
+                        
+                        # Robust matching
+                        matching_pos = None
+                        for pos in default_account.positions:
+                            if pos.entry_price == bracket.entry_price and pos.direction == bracket.config.direction:
+                                matching_pos = pos
+                                break
+                        
+                        if matching_pos:
+                            default_account.close_position(
+                                position=matching_pos,
+                                fill=bracket.exit_fill,
+                                outcome=bracket._get_outcome(),
+                                time=self.current_timestamp
+                            )
+
         
-        # 4. Update accounts with current price
+        # Remove completed
+        for b in completed_brackets:
+            self.active_brackets.remove(b)
+
+        # 3. Strategy / Scanner (Entries)
+        if self.scanner:
+            features = compute_features(
+                self.stepper, 
+                self.feature_config, 
+                df_5m=self.df_5m, 
+                df_15m=self.df_15m
+            )
+            
+            scan_result = self.scanner.scan(None, features) # MarketState None for now
+            
+            if scan_result.triggered:
+                # Create Decision Event
+                events.append(SimEvent(
+                    type=SimEventType.DECISION,
+                    timestamp=self.current_timestamp,
+                    bar_idx=self.current_bar_idx,
+                    data={
+                        'scanner_id': self.scanner.__class__.__name__,
+                        'triggered': True,
+                        'price': features.current_price,
+                        'atr': features.atr
+                    }
+                ))
+                
+                # Create Bracket from ScanResult
+                # We need a proper OCOConfig. In a real app, this comes from the Strategy class.
+                # For now, we construct one based on the scanner's direction or defaults.
+                # Attempt to get direction from result or feature set
+                direction = getattr(scan_result, 'direction', "LONG")
+                
+                # Construct OCO Config (Defaults if not provided by strategy)
+                # TODO: Retrieve this from self.strategy_config or similar
+                from src.sim.oco_engine import OCOConfig
+                oco_config = OCOConfig(
+                    direction=direction,
+                    entry_type="MARKET", # Default to Market for immediate entry if triggered? Or Limit?
+                    stop_atr=2.0,       # Default
+                    tp_multiple=2.0,    # Default
+                    name=f"Sim_{self.scanner.__class__.__name__}"
+                )
+                
+                # Create and register the bracket
+                bracket = self.oco_engine.create_bracket(
+                    config=oco_config,
+                    base_price=features.current_price,
+                    atr=features.atr,
+                    current_idx=self.current_bar_idx
+                )
+                
+                # Note: create_bracket calculates prices but doesn't "place" it unless we track it
+                # We add to active_brackets. 
+                # Ideally, we should have an 'Order' abstraction, but OCOBracket handles the lifecycle well enough here.
+                # However, OCOEngine.create_bracket returns a bracket in PENDING_ENTRY state.
+                self.active_brackets.append(bracket)
+                
+                # Emit OCO Created Event
+                events.append(SimEvent(
+                    type=SimEventType.ORDER_SUBMIT, # Logic maps creation to submission
+                    timestamp=self.current_timestamp,
+                    bar_idx=self.current_bar_idx,
+                    data=bracket.to_flat_dict()
+                ))
+
         current_price = float(step.bar['close'])
         for account_id in self.account_manager.list_accounts():
             snapshot = self.account_manager.take_snapshot(
```

### New Untracked Files

#### `gitrdiff.md`

```
```

---

## Commits Ahead (local changes not on remote)

```
```

## Commits Behind (remote changes not pulled)

```
```

---

## File Changes (YOUR UNPUSHED CHANGES)

```
```

---

## Full Diff of Your Unpushed Changes

Green (+) = lines you ADDED locally
Red (-) = lines you REMOVED locally

```diff
```

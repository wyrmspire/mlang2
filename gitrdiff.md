# Git Diff Report

**Generated**: Thu, Dec 25, 2025 12:00:21 PM

**Local Branch**: master

**Comparing Against**: origin/master

---

## Uncommitted Changes (working directory)

### Modified/Staged Files

```
 M gitrdiff.md
 M src/experiments/runner.py
 M src/policy/library/delayed_breakout.py
 M src/policy/library/ict_fvg.py
 M src/policy/library/mean_reversion.py
 M src/policy/library/mid_day_reversal.py
 M src/policy/library/opening_range.py
 M src/policy/library/simple_time.py
 M src/policy/library/structure_break.py
 M src/policy/library/swing_breakout.py
 M src/policy/modular_scanner.py
 M src/policy/scanner_registry_init.py
 M src/policy/scanners.py
 M src/sim/market_session.py
?? src/sim/causal_runner.py
?? tests/test_causal_runner.py
```

### Uncommitted Diff

```diff
diff --git a/gitrdiff.md b/gitrdiff.md
index d069163..f02482d 100644
--- a/gitrdiff.md
+++ b/gitrdiff.md
@@ -1,6 +1,6 @@
 # Git Diff Report
 
-**Generated**: Thu, Dec 25, 2025 11:39:54 AM
+**Generated**: Thu, Dec 25, 2025 12:00:21 PM
 
 **Local Branch**: master
 
@@ -13,393 +13,24 @@
 ### Modified/Staged Files
 
 ```
+ M gitrdiff.md
  M src/experiments/runner.py
+ M src/policy/library/delayed_breakout.py
+ M src/policy/library/ict_fvg.py
+ M src/policy/library/mean_reversion.py
+ M src/policy/library/mid_day_reversal.py
+ M src/policy/library/opening_range.py
+ M src/policy/library/simple_time.py
+ M src/policy/library/structure_break.py
+ M src/policy/library/swing_breakout.py
+ M src/policy/modular_scanner.py
+ M src/policy/scanner_registry_init.py
+ M src/policy/scanners.py
  M src/sim/market_session.py
-?? gitrdiff.md
+?? src/sim/causal_runner.py
+?? tests/test_causal_runner.py
 ```
 
 ### Uncommitted Diff
 
 ```diff
-diff --git a/src/experiments/runner.py b/src/experiments/runner.py
-index 38a2d27..8631c39 100644
---- a/src/experiments/runner.py
-+++ b/src/experiments/runner.py
-@@ -22,11 +22,13 @@ from src.data.resample import resample_all_timeframes
- 
- from src.sim.stepper import MarketStepper
- from src.sim.oco_engine import create_oco_bracket
-+from src.sim.sizing import calculate_contracts, calculate_pnl_dollars
- from src.features.pipeline import compute_features, precompute_indicators
- from src.policy.scanners import get_scanner
- from src.policy.filters import DEFAULT_FILTERS
- from src.policy.cooldown import CooldownManager
- from src.policy.actions import Action, SkipReason
-+from src.viz.window_utils import enforce_2hour_window
- 
- from src.labels.labeler import Labeler
- from src.datasets.decision_record import DecisionRecord
-@@ -37,7 +39,7 @@ from src.datasets.reader import create_dataloader
- from src.models.fusion import FusionModel
- from src.models.train import train_model, TrainResult
- 
--from src.config import PROCESSED_DIR, SHARDS_DIR, RESULTS_DIR
-+from src.config import PROCESSED_DIR, SHARDS_DIR, RESULTS_DIR, DEFAULT_MAX_RISK_DOLLARS
- 
- 
- @dataclass
-@@ -188,11 +190,19 @@ def run_experiment(
-         if exporter:
-             curr_idx = step.bar_idx
-             
--            # Extract RAW OHLCV for chart: 60 bars before + 20 bars after
--            start_raw_idx = max(0, curr_idx - 60)
--            end_raw_idx = min(len(df), curr_idx + 21)  # +1 for current bar, +20 future
--            raw_slice = df.iloc[start_raw_idx : end_raw_idx]
--            raw_ohlcv = raw_slice[['open', 'high', 'low', 'close', 'volume']].values.tolist()
-+            exit_time = None
-+            if record.action == Action.PLACE_ORDER:
-+                exit_time = features.timestamp + pd.Timedelta(minutes=record.cf_bars_held)
-+
-+            raw_ohlcv, window_warning = enforce_2hour_window(
-+                df_1m=df,
-+                entry_time=features.timestamp,
-+                exit_time=exit_time,
-+                bars_held=record.cf_bars_held
-+            )
-+
-+            if window_warning:
-+                exporter._window_warnings.append(window_warning)
-             
-             # Extract future bars separately (for compatibility)
-             future_bars = []
-@@ -226,7 +236,16 @@ def run_experiment(
-                     base_price=features.current_price,
-                     atr=features.atr
-                 )
--                exporter.on_bracket_created(record.decision_id, bracket)
-+                sizing_result = calculate_contracts(
-+                    entry_price=bracket.entry_price,
-+                    stop_price=bracket.stop_price,
-+                    max_risk_dollars=DEFAULT_MAX_RISK_DOLLARS
-+                )
-+                exporter.on_bracket_created(
-+                    record.decision_id,
-+                    bracket,
-+                    contracts=sizing_result.contracts
-+                )
-         
-         # Update cooldown if trade placed
-         if record.action == Action.PLACE_ORDER:
-@@ -238,6 +257,26 @@ def run_experiment(
-                 exit_bar = step.bar_idx + record.cf_bars_held
-                 exit_time = features.timestamp + pd.Timedelta(minutes=record.cf_bars_held)
-                 
-+                bracket = create_oco_bracket(
-+                    config=config.oco_config,
-+                    base_price=features.current_price,
-+                    atr=features.atr
-+                )
-+                sizing_result = calculate_contracts(
-+                    entry_price=bracket.entry_price,
-+                    stop_price=bracket.stop_price,
-+                    max_risk_dollars=DEFAULT_MAX_RISK_DOLLARS
-+                )
-+                exit_price = features.current_price + (
-+                    record.cf_pnl / (1 if config.oco_config.direction == "LONG" else -1)
-+                )
-+                pnl_points, pnl_dollars = calculate_pnl_dollars(
-+                    entry_price=features.current_price,
-+                    exit_price=exit_price,
-+                    direction=config.oco_config.direction,
-+                    contracts=sizing_result.contracts
-+                )
-+
-                 trade = TradeRecord(
-                     trade_id=str(uuid.uuid4())[:8],
-                     decision_id=record.decision_id,
-@@ -247,12 +286,12 @@ def run_experiment(
-                     direction=config.oco_config.direction,
-                     exit_time=exit_time,
-                     exit_bar=exit_bar,
--                    exit_price=features.current_price + (record.cf_pnl / (1 if config.oco_config.direction=="LONG" else -1)),
-+                    exit_price=exit_price,
-                     exit_reason=record.cf_outcome,
-                     outcome=record.cf_outcome,
--                    pnl_points=record.cf_pnl,
--                    pnl_dollars=record.cf_pnl_dollars,
--                    r_multiple=record.cf_pnl_dollars / (features.atr * config.oco_config.stop_atr * 50) if features.atr > 0 else 0,
-+                    pnl_points=pnl_points,
-+                    pnl_dollars=pnl_dollars,
-+                    r_multiple=pnl_dollars / (features.atr * config.oco_config.stop_atr * 50) if features.atr > 0 else 0,
-                     bars_held=record.cf_bars_held,
-                     mae=record.cf_mae,
-                     mfe=record.cf_mfe,
-diff --git a/src/sim/market_session.py b/src/sim/market_session.py
-index c41a214..b56ebf5 100644
---- a/src/sim/market_session.py
-+++ b/src/sim/market_session.py
-@@ -22,6 +22,10 @@ from pathlib import Path
- 
- from src.sim.stepper import MarketStepper
- from src.sim.account_manager import AccountManager
-+from src.sim.oco_engine import OCOEngine, OCOStatus
-+from src.sim.sizing import calculate_contracts, calculate_pnl_dollars
-+from src.features.pipeline import compute_features, FeatureConfig, precompute_indicators
-+from src.policy.scanners import Scanner, ScanResult
- from src.core.enums import RunMode
- from src.core.registries import IndicatorSeries, IndicatorRegistry
- 
-@@ -98,13 +102,18 @@ class MarketSession:
-         # Account manager
-         self.account_manager = AccountManager()
-         
--        # Indicators (computed on demand, cached)
--        self.indicator_cache: Dict[str, IndicatorSeries] = {}
--        self.enabled_indicators: List[str] = []
-+        # OCO Engine
-+        self.oco_engine = OCOEngine()
-+        self.active_brackets = []  # List[OCOBracket]
-         
--        # Policies (scanners/models)
--        self.active_scanners: List[Any] = []
--        self.active_models: List[Any] = []
-+        # Strategy components
-+        self.scanner: Optional[Scanner] = None
-+        self.scanner_config: Dict[str, Any] = {}
-+        self.feature_config: FeatureConfig = FeatureConfig()
-+        
-+        # Data caches
-+        self.df_5m = None
-+        self.df_15m = None
-         
-         # Event log
-         self.events: List[SimEvent] = []
-@@ -114,6 +123,13 @@ class MarketSession:
-         self.is_paused = False
-         self.current_bar_idx: Optional[int] = None
-         self.current_timestamp: Optional[pd.Timestamp] = None
-+        
-+    def setup_strategy(self, scanner: Scanner, feature_config: FeatureConfig, df_5m=None, df_15m=None):
-+        """Configure strategy for the session."""
-+        self.scanner = scanner
-+        self.feature_config = feature_config
-+        self.df_5m = df_5m
-+        self.df_15m = df_15m
-     
-     def add_account(self, account_id: str, starting_balance: float = 50000.0):
-         """Add an account to the session."""
-@@ -216,26 +232,158 @@ class MarketSession:
-         )
-         events.append(bar_event)
-         
--        # 2. INDICATORS event (if any enabled)
--        if self.enabled_indicators:
--            # Compute indicators (in real implementation, would use IndicatorRegistry)
--            indicators_data = {}
--            for ind_id in self.enabled_indicators:
--                # Placeholder - real implementation would compute
--                indicators_data[ind_id] = None
--            
--            ind_event = SimEvent(
--                type=SimEventType.INDICATORS,
--                timestamp=self.current_timestamp,
--                bar_idx=self.current_bar_idx,
--                data={'indicators': indicators_data}
-+        # 2. Update active OCO brackets (Exits/Fills)
-+        # This MUST happen before new entries to free up capital/slots
-+        completed_brackets = []
-+        for bracket in self.active_brackets:
-+            updated_bracket, event_type = self.oco_engine.process_bar(
-+                bracket, step.bar, self.current_bar_idx
-             )
--            events.append(ind_event)
--        
--        # 3. Check scanners/policies for DECISION events
--        # (Placeholder - real implementation would run scanners)
-+            
-+            if event_type:
-+                # Emit event
-+                events.append(SimEvent(
-+                    type=SimEventType(event_type) if event_type in [e.value for e in SimEventType] else SimEventType.FILL,
-+                    timestamp=self.current_timestamp,
-+                    bar_idx=self.current_bar_idx,
-+                    data={'bracket_id': id(bracket), 'status': bracket.status.value, 'event': event_type}
-+                ))
-+                
-+                # Check for completion
-+                if bracket.status in [OCOStatus.CLOSED_TP, OCOStatus.CLOSED_SL, OCOStatus.CLOSED_TIMEOUT, OCOStatus.CANCELLED]:
-+                    completed_brackets.append(bracket)
-+                    
-+                    # Update Account
-+                    if bracket.exit_fill:
-+                        # Calculate PnL
-+                        entry_price = bracket.entry_price
-+                        exit_price = bracket.exit_fill.fill_price
-+                        direction = bracket.config.direction
-+                        
-+                        contracts = getattr(bracket, 'contracts', 1) 
-+                        
-+                        # Use AccountManager to handle the close
-+                        # We need to find the specific position. 
-+                        # Limitation: The current AccountManager tracks strict FIFO/Position objects.
-+                        # OCOBracket tracks a "Trade" lifecycle.
-+                        # We need to bridge them.
-+                        
-+                        # For this refactor, we will simpler update the default account balance
-+                        # directly to ensure stats work, assuming single account 'default'.
-+                        default_account = self.account_manager.get_account('default')
-+                        if default_account:
-+                            # We construct a fill and 'close_position' on the account.
-+                            # But wait, did we 'open' it on the account? 
-+                            # We missed the ENTRY fill event in the loop above!
-+                            pass # Handled below in separate check
-+                            
-+                # Check for Entry Fill strictly (State transition PENDING -> ACTIVE)
-+                if event_type == 'ENTRY':
-+                     default_account = self.account_manager.get_account('default')
-+                     if default_account:
-+                         # Register the position
-+                         # We need the contracts/size.
-+                         contracts = getattr(bracket, 'contracts', 1) 
-+                         
-+                         default_account.open_position(
-+                             fill=bracket.entry_fill,
-+                             stop_loss=bracket.stop_price,
-+                             take_profit=bracket.tp_price,
-+                             time=self.current_timestamp
-+                         )
-+                
-+                # Check for Exit Fill (State transition ACTIVE -> CLOSED_*)
-+                if bracket.exit_fill and bracket.status in [OCOStatus.CLOSED_TP, OCOStatus.CLOSED_SL, OCOStatus.CLOSED_TIMEOUT]:
-+                    default_account = self.account_manager.get_account('default')
-+                    if default_account:
-+                        # Find matching position (Approximation for now: assume last or matching direction)
-+                        # AccountManager.close_position wants the Position object.
-+                        # We'll search for one matching entry price/time.
-+                        
-+                        # Robust matching
-+                        matching_pos = None
-+                        for pos in default_account.positions:
-+                            if pos.entry_price == bracket.entry_price and pos.direction == bracket.config.direction:
-+                                matching_pos = pos
-+                                break
-+                        
-+                        if matching_pos:
-+                            default_account.close_position(
-+                                position=matching_pos,
-+                                fill=bracket.exit_fill,
-+                                outcome=bracket._get_outcome(),
-+                                time=self.current_timestamp
-+                            )
-+
-         
--        # 4. Update accounts with current price
-+        # Remove completed
-+        for b in completed_brackets:
-+            self.active_brackets.remove(b)
-+
-+        # 3. Strategy / Scanner (Entries)
-+        if self.scanner:
-+            features = compute_features(
-+                self.stepper, 
-+                self.feature_config, 
-+                df_5m=self.df_5m, 
-+                df_15m=self.df_15m
-+            )
-+            
-+            scan_result = self.scanner.scan(None, features) # MarketState None for now
-+            
-+            if scan_result.triggered:
-+                # Create Decision Event
-+                events.append(SimEvent(
-+                    type=SimEventType.DECISION,
-+                    timestamp=self.current_timestamp,
-+                    bar_idx=self.current_bar_idx,
-+                    data={
-+                        'scanner_id': self.scanner.__class__.__name__,
-+                        'triggered': True,
-+                        'price': features.current_price,
-+                        'atr': features.atr
-+                    }
-+                ))
-+                
-+                # Create Bracket from ScanResult
-+                # We need a proper OCOConfig. In a real app, this comes from the Strategy class.
-+                # For now, we construct one based on the scanner's direction or defaults.
-+                # Attempt to get direction from result or feature set
-+                direction = getattr(scan_result, 'direction', "LONG")
-+                
-+                # Construct OCO Config (Defaults if not provided by strategy)
-+                # TODO: Retrieve this from self.strategy_config or similar
-+                from src.sim.oco_engine import OCOConfig
-+                oco_config = OCOConfig(
-+                    direction=direction,
-+                    entry_type="MARKET", # Default to Market for immediate entry if triggered? Or Limit?
-+                    stop_atr=2.0,       # Default
-+                    tp_multiple=2.0,    # Default
-+                    name=f"Sim_{self.scanner.__class__.__name__}"
-+                )
-+                
-+                # Create and register the bracket
-+                bracket = self.oco_engine.create_bracket(
-+                    config=oco_config,
-+                    base_price=features.current_price,
-+                    atr=features.atr,
-+                    current_idx=self.current_bar_idx
-+                )
-+                
-+                # Note: create_bracket calculates prices but doesn't "place" it unless we track it
-+                # We add to active_brackets. 
-+                # Ideally, we should have an 'Order' abstraction, but OCOBracket handles the lifecycle well enough here.
-+                # However, OCOEngine.create_bracket returns a bracket in PENDING_ENTRY state.
-+                self.active_brackets.append(bracket)
-+                
-+                # Emit OCO Created Event
-+                events.append(SimEvent(
-+                    type=SimEventType.ORDER_SUBMIT, # Logic maps creation to submission
-+                    timestamp=self.current_timestamp,
-+                    bar_idx=self.current_bar_idx,
-+                    data=bracket.to_flat_dict()
-+                ))
-+
-         current_price = float(step.bar['close'])
-         for account_id in self.account_manager.list_accounts():
-             snapshot = self.account_manager.take_snapshot(
-```
-
-### New Untracked Files
-
-#### `gitrdiff.md`
-
-```
-```
-
----
-
-## Commits Ahead (local changes not on remote)
-
-```
-```
-
-## Commits Behind (remote changes not pulled)
-
-```
-```
-
----
-
-## File Changes (YOUR UNPUSHED CHANGES)
-
-```
-```
-
----
-
-## Full Diff of Your Unpushed Changes
-
-Green (+) = lines you ADDED locally
-Red (-) = lines you REMOVED locally
-
-```diff
-```
diff --git a/src/experiments/runner.py b/src/experiments/runner.py
index 8631c39..53c00ba 100644
--- a/src/experiments/runner.py
+++ b/src/experiments/runner.py
@@ -23,6 +23,8 @@ from src.data.resample import resample_all_timeframes
 from src.sim.stepper import MarketStepper
 from src.sim.oco_engine import create_oco_bracket
 from src.sim.sizing import calculate_contracts, calculate_pnl_dollars
+from src.sim.causal_runner import CausalExecutor, StepResult
+from src.sim.account_manager import AccountManager
 from src.features.pipeline import compute_features, precompute_indicators
 from src.policy.scanners import get_scanner
 from src.policy.filters import DEFAULT_FILTERS
@@ -126,44 +128,86 @@ def run_experiment(
     labeler = Labeler(config.label_config)
     cooldown = CooldownManager()
     
+    # Initialize Causal Executor
+    # Note: Experiment runner uses its own strict stepper, so we pass it in.
+    # We also use a dummy AccountManager as we are just checking for signals/records here.
+    account_manager = AccountManager()
+    executor = CausalExecutor(
+        df=df,
+        stepper=stepper,
+        account_manager=account_manager,
+        scanner=scanner,
+        feature_config=config.feature_config,
+        df_5m=df_5m,
+        df_15m=df_15m
+    )
+    
     records: List[DecisionRecord] = []
     
     while True:
-        step = stepper.step()
-        if step.is_done:
+        # Step the unified executor
+        result = executor.step()
+        if not result:
             break
+            
+        # If scanner triggered, we have a potential record
+        # In CausalExecutor, triggers are in result.scanner_triggers and result.new_orders
+        # We need to map this back to DecisionRecord format.
         
-        # Compute features
-        features = compute_features(
-            stepper,
-            config.feature_config,
-            df_5m=df_5m,
-            df_15m=df_15m,
-            precomputed_indicators=indicators_map,
-        )
+        # We only care if meaningful decision occurred (scanner checked)
+        # CausalExecutor runs scanner every step if provided.
+        # But we only want to RECORD if it triggered or if we want negative samples?
+        # The original code recorded ONLY IF skip_reason != SKIP (or if it was filter blocked).
+        # Actually original code recorded ALL scan attempts that passed basic checks?
+        # Original: "if not scan_result.triggered: continue"
         
-        # Check if scanner triggers
-        scan_result = scanner.scan(features.market_state, features)
-        if not scan_result.triggered:
+        # So we check if triggered.
+        if not result.scanner_triggers:
             continue
+            
+        # Extract the first trigger (assuming one per bar for now)
+        trigger = result.scanner_triggers[0]
+        # And the bracket (order) if any
+        bracket_ref = result.new_orders[0] if result.new_orders else None
+        
+        # Features are available in result
+        features = result.features
+        
+        # Re-verify filters/cooldown using the centralized logic or here?
+        # CausalExecutor creates the order if triggered. It doesn't check "cooldown" from policy/cooldown.py
+        # because that's a higher-level policy. 
+        # Wait, if CausalExecutor creates the order, it implies it passed checks?
+        # The current CausalExecutor implementation is bare-bones: Trigger -> Order.
+        # It misses the Filter/Cooldown/Skip logic from the old runner.
+        
+        # To maintain exact parity, we should move Filter/Cooldown INTO CausalExecutor?
+        # Or check it here and "Cancel" the order?
+        # FOR NOW: We will re-implement the check here to decide SKIP vs PLACE, matching old runner.
+        # Ideally, CausalExecutor should take a 'Policy' object that handles this.
         
         # Check filters
         filter_result = DEFAULT_FILTERS.check(features)
         if not filter_result.passed:
             skip_reason = SkipReason.FILTER_BLOCK
         # Check cooldown
-        elif cooldown.is_on_cooldown(step.bar_idx, features.timestamp)[0]:
+        elif cooldown.is_on_cooldown(result.bar_idx, result.timestamp)[0]:
             skip_reason = SkipReason.COOLDOWN
         else:
             skip_reason = SkipReason.NOT_SKIPPED
+            
+        # Determine Action
+        action = Action.NO_TRADE if skip_reason != SkipReason.NOT_SKIPPED else Action.PLACE_ORDER
+        
+        # If we skipped, we technically "cancelled" the order the executor made.
+        # But for 'Generating Data', we just record the decision.
         
         # Create record
         record = DecisionRecord(
-            timestamp=features.timestamp,
-            bar_idx=step.bar_idx,
+            timestamp=result.timestamp,
+            bar_idx=result.bar_idx,
             decision_id=str(uuid.uuid4())[:8],
             scanner_id=config.scanner_id,
-            action=Action.NO_TRADE if skip_reason != SkipReason.NOT_SKIPPED else Action.PLACE_ORDER,
+            action=action,
             skip_reason=skip_reason,
             x_price_1m=features.x_price_1m,
             x_price_5m=features.x_price_5m,
@@ -173,8 +217,10 @@ def run_experiment(
             atr=features.atr,
         )
         
-        # 3. Label with counterfactual outcome
-        cf_label = labeler.label_decision_point(df, step.bar_idx, features.atr)
+        # 3. Label with counterfactual outcome (TRAINING/DATA GEN ONLY)
+        # This is the "Lookahead" step that we keep ONLY for data generation.
+        # It uses the Labeler to jump ahead and see what happened.
+        cf_label = labeler.label_decision_point(df, result.bar_idx, features.atr)
         record.cf_outcome = cf_label.outcome
         record.cf_pnl = cf_label.pnl
         record.cf_pnl_dollars = cf_label.pnl_dollars
@@ -186,9 +232,8 @@ def run_experiment(
         
         records.append(record)
         
-        # === VIZ EXPORT HOOK ===
         if exporter:
-            curr_idx = step.bar_idx
+            curr_idx = result.bar_idx
             
             exit_time = None
             if record.action == Action.PLACE_ORDER:
@@ -249,12 +294,12 @@ def run_experiment(
         
         # Update cooldown if trade placed
         if record.action == Action.PLACE_ORDER:
-            cooldown.record_trade(step.bar_idx, cf_label.outcome, features.timestamp)
+            cooldown.record_trade(result.bar_idx, cf_label.outcome, features.timestamp)
             
-            # Export Trade Record for Viz
+            # Export Trade Record for Viz (Constructed from CF outcome)
             if exporter:
                 # Approximate exit bar
-                exit_bar = step.bar_idx + record.cf_bars_held
+                exit_bar = result.bar_idx + record.cf_bars_held
                 exit_time = features.timestamp + pd.Timedelta(minutes=record.cf_bars_held)
                 
                 bracket = create_oco_bracket(
diff --git a/src/policy/library/delayed_breakout.py b/src/policy/library/delayed_breakout.py
index 6a9bfdf..538e73b 100644
--- a/src/policy/library/delayed_breakout.py
+++ b/src/policy/library/delayed_breakout.py
@@ -10,7 +10,7 @@ from typing import Optional
 from dataclasses import dataclass
 from datetime import time
 
-from src.policy.scanners import Scanner, ScannerResult
+from src.policy.scanners import Scanner, ScanResult
 from src.features.state import MarketState
 from src.features.pipeline import FeatureBundle
 
@@ -159,30 +159,30 @@ class DelayedBreakoutScanner(Scanner):
         features: FeatureBundle,
         df_15m: pd.DataFrame = None,
         df_5m: pd.DataFrame = None
-    ) -> ScannerResult:
+    ) -> ScanResult:
         """Check for delayed breakout."""
         t = features.timestamp
         if t is None:
-            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
+            return ScanResult(scanner_id=self.scanner_id, triggered=False)
         
         # 1. TIME FILTER
         current_t = t.time()
         if current_t < self.start_time or current_t > self.end_time:
-            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
+            return ScanResult(scanner_id=self.scanner_id, triggered=False)
             
         # Cooldown check
         if features.bar_idx - self._state.last_trigger_bar < self.cooldown_bars:
-            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
+            return ScanResult(scanner_id=self.scanner_id, triggered=False)
         
         # Need data
         if df_15m is None or df_15m.empty:
-            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
+            return ScanResult(scanner_id=self.scanner_id, triggered=False)
         
         # Compute swings (for Entry)
         swing_high, swing_low = self._compute_swing_levels(df_15m, t)
         
         if swing_high == 0 or swing_low == 0:
-            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
+            return ScanResult(scanner_id=self.scanner_id, triggered=False)
         
         current_price = features.current_price
         atr = features.atr if features.atr > 0 else 1.0
@@ -211,7 +211,7 @@ class DelayedBreakoutScanner(Scanner):
                     stop_price = swing_low - padding
                 
                 risk = current_price - stop_price
-                if risk <= 0: return ScannerResult(scanner_id=self.scanner_id, triggered=False)
+                if risk <= 0: return ScanResult(scanner_id=self.scanner_id, triggered=False)
                 tp_price = current_price + (1.4 * risk)
                 
             else: # SHORT
@@ -221,17 +221,17 @@ class DelayedBreakoutScanner(Scanner):
                     stop_price = swing_high + padding
                 
                 risk = stop_price - current_price
-                if risk <= 0: return ScannerResult(scanner_id=self.scanner_id, triggered=False)
+                if risk <= 0: return ScanResult(scanner_id=self.scanner_id, triggered=False)
                 tp_price = current_price - (1.4 * risk)
             
             # 3. Position Sizing
             contracts, actual_risk = self._calculate_position_size(current_price, stop_price)
             if contracts == 0:
-                 return ScannerResult(scanner_id=self.scanner_id, triggered=False)
+                 return ScanResult(scanner_id=self.scanner_id, triggered=False)
 
             self._state.last_trigger_bar = features.bar_idx
             
-            return ScannerResult(
+            return ScanResult(
                 scanner_id=self.scanner_id,
                 triggered=True,
                 context={
@@ -252,4 +252,4 @@ class DelayedBreakoutScanner(Scanner):
                 score=1.0
             )
             
-        return ScannerResult(scanner_id=self.scanner_id, triggered=False)
+        return ScanResult(scanner_id=self.scanner_id, triggered=False)
diff --git a/src/policy/library/ict_fvg.py b/src/policy/library/ict_fvg.py
index 6d0edec..fbefcef 100644
--- a/src/policy/library/ict_fvg.py
+++ b/src/policy/library/ict_fvg.py
@@ -20,7 +20,7 @@ from typing import Optional, Dict, Any
 from dataclasses import dataclass, field
 from datetime import time
 
-from src.policy.scanners import Scanner, ScannerResult
+from src.policy.scanners import Scanner, ScanResult
 from src.features.state import MarketState
 from src.features.pipeline import FeatureBundle
 from src.features.session_levels import (
@@ -216,11 +216,11 @@ class ICTFVGScanner(Scanner):
         features: FeatureBundle,
         df_5m: pd.DataFrame = None,
         df_1m: pd.DataFrame = None
-    ) -> ScannerResult:
+    ) -> ScanResult:
         """Check for ICT FVG setup."""
         t = features.timestamp
         if t is None:
-            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
+            return ScanResult(scanner_id=self.scanner_id, triggered=False)
         
         # 1. Check if we're in the trade window (9:30 - 11:30 NY)
         if not is_in_trade_window(t, NY_TZ):
@@ -245,7 +245,7 @@ class ICTFVGScanner(Scanner):
         
         # 5. Need data
         if df_5m is None or df_5m.empty or df_1m is None or df_1m.empty:
-            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
+            return ScanResult(scanner_id=self.scanner_id, triggered=False)
         
         current_price = features.current_price
         atr = features.atr if features.atr > 0 else 5.0
@@ -358,7 +358,7 @@ class ICTFVGScanner(Scanner):
                 self._state.active_fvg = None
                 self._state.pending_direction = None
                 
-                return ScannerResult(
+                return ScanResult(
                     scanner_id=self.scanner_id,
                     triggered=True,
                     context={
@@ -386,4 +386,4 @@ class ICTFVGScanner(Scanner):
                     score=1.0
                 )
         
-        return ScannerResult(scanner_id=self.scanner_id, triggered=False)
+        return ScanResult(scanner_id=self.scanner_id, triggered=False)
diff --git a/src/policy/library/mean_reversion.py b/src/policy/library/mean_reversion.py
index a2c37ca..b58b939 100644
--- a/src/policy/library/mean_reversion.py
+++ b/src/policy/library/mean_reversion.py
@@ -4,7 +4,7 @@ Triggers when price extends beyond Keltner Channels (EMA +/- ATR bands).
 """
 
 from typing import Dict, Any
-from src.policy.scanners import Scanner, ScannerResult
+from src.policy.scanners import Scanner, ScanResult
 from src.features.state import MarketState
 from src.features.pipeline import FeatureBundle
 
@@ -39,9 +39,9 @@ class MeanReversionScanner(Scanner):
         self,
         state: MarketState,
         features: FeatureBundle
-    ) -> ScannerResult:
+    ) -> ScanResult:
         if features.indicators is None:
-            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
+            return ScanResult(scanner_id=self.scanner_id, triggered=False)
         
         ind = features.indicators
         
@@ -54,10 +54,10 @@ class MeanReversionScanner(Scanner):
             atr = ind.atr_15m_14
             rsi = ind.rsi_15m_14
         else:
-            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
+            return ScanResult(scanner_id=self.scanner_id, triggered=False)
             
         if ema == 0 or atr == 0:
-            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
+            return ScanResult(scanner_id=self.scanner_id, triggered=False)
             
         current_price = features.current_price
         
@@ -93,7 +93,7 @@ class MeanReversionScanner(Scanner):
             # Reset state when condition is lost
             self._was_triggered = False
             
-        return ScannerResult(
+        return ScanResult(
             scanner_id=self.scanner_id,
             triggered=triggered,
             context={
diff --git a/src/policy/library/mid_day_reversal.py b/src/policy/library/mid_day_reversal.py
index fb49679..6c88ee6 100644
--- a/src/policy/library/mid_day_reversal.py
+++ b/src/policy/library/mid_day_reversal.py
@@ -3,7 +3,7 @@ Mid-Day Reversal Strategy
 Modular scanner that looks for reversals during lunch/mid-day.
 """
 
-from src.policy.scanners import Scanner, ScannerResult
+from src.policy.scanners import Scanner, ScanResult
 from src.features.state import MarketState
 from src.features.pipeline import FeatureBundle
 
@@ -35,19 +35,19 @@ class MidDayReversalScanner(Scanner):
         self,
         state: MarketState,
         features: FeatureBundle
-    ) -> ScannerResult:
+    ) -> ScanResult:
         t = features.time_features
         if not t or not t.is_rth:
-            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
+            return ScanResult(scanner_id=self.scanner_id, triggered=False)
         
         # 1. Check time window
         is_midday = self.start_hour <= t.hour_ny <= self.end_hour
         if not is_midday:
-            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
+            return ScanResult(scanner_id=self.scanner_id, triggered=False)
         
         # 2. Check for reversal signal (Simple RSI extreme for now)
         if features.indicators is None:
-            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
+            return ScanResult(scanner_id=self.scanner_id, triggered=False)
         
         rsi = features.indicators.rsi_5m_14
         oversold = rsi <= self.rsi_extreme
@@ -55,7 +55,7 @@ class MidDayReversalScanner(Scanner):
         
         triggered = oversold or overbought
         
-        return ScannerResult(
+        return ScanResult(
             scanner_id=self.scanner_id,
             triggered=triggered,
             context={
diff --git a/src/policy/library/opening_range.py b/src/policy/library/opening_range.py
index badd961..036827b 100644
--- a/src/policy/library/opening_range.py
+++ b/src/policy/library/opening_range.py
@@ -8,7 +8,7 @@ from typing import Optional, Dict, Any
 from dataclasses import dataclass, field
 from zoneinfo import ZoneInfo
 
-from src.policy.scanners import Scanner, ScannerResult
+from src.policy.scanners import Scanner, ScanResult
 from src.features.state import MarketState
 from src.features.pipeline import FeatureBundle
 from src.config import NY_TZ
@@ -94,10 +94,10 @@ class OpeningRangeScanner(Scanner):
         self,
         state: MarketState,
         features: FeatureBundle
-    ) -> ScannerResult:
+    ) -> ScanResult:
         t = features.timestamp
         if t is None:
-            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
+            return ScanResult(scanner_id=self.scanner_id, triggered=False)
         
         # Reset on new day
         if self._is_new_day(t):
@@ -111,7 +111,7 @@ class OpeningRangeScanner(Scanner):
                 'low': state.current_low if hasattr(state, 'current_low') else features.current_price - 1,
             }
             self._or_bars.append(bar_data)
-            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
+            return ScanResult(scanner_id=self.scanner_id, triggered=False)
         
         # Establish OR after period ends
         if self._is_after_or(t) and not self._state.or_established and len(self._or_bars) > 0:
@@ -121,11 +121,11 @@ class OpeningRangeScanner(Scanner):
         
         # Can't trigger if OR not established
         if not self._state.or_established:
-            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
+            return ScanResult(scanner_id=self.scanner_id, triggered=False)
         
         # Cooldown check
         if features.bar_idx - self._state.last_trigger_bar < self.cooldown_bars:
-            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
+            return ScanResult(scanner_id=self.scanner_id, triggered=False)
         
         # Check for retest
         price = features.current_price
@@ -149,7 +149,7 @@ class OpeningRangeScanner(Scanner):
         
         if long_triggered or short_triggered:
             self._state.last_trigger_bar = features.bar_idx
-            return ScannerResult(
+            return ScanResult(
                 scanner_id=self.scanner_id,
                 triggered=True,
                 context={
@@ -162,4 +162,4 @@ class OpeningRangeScanner(Scanner):
                 score=1.0
             )
         
-        return ScannerResult(scanner_id=self.scanner_id, triggered=False)
+        return ScanResult(scanner_id=self.scanner_id, triggered=False)
diff --git a/src/policy/library/simple_time.py b/src/policy/library/simple_time.py
index 54c1c03..f87e600 100644
--- a/src/policy/library/simple_time.py
+++ b/src/policy/library/simple_time.py
@@ -8,7 +8,7 @@ from typing import Optional, Dict, Any
 from dataclasses import dataclass
 from zoneinfo import ZoneInfo
 
-from src.policy.scanners import Scanner, ScannerResult
+from src.policy.scanners import Scanner, ScanResult
 from src.features.state import MarketState
 from src.features.pipeline import FeatureBundle
 from src.config import NY_TZ
@@ -49,10 +49,10 @@ class SimpleTimeScanner(Scanner):
         self,
         state: MarketState,
         features: FeatureBundle
-    ) -> ScannerResult:
+    ) -> ScanResult:
         t = features.timestamp
         if t is None:
-            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
+            return ScanResult(scanner_id=self.scanner_id, triggered=False)
         
         ny_time = t.astimezone(NY_TZ)
         
@@ -60,7 +60,7 @@ class SimpleTimeScanner(Scanner):
         if ny_time.hour == self.hour and ny_time.minute == self.minute:
             # Check if we already traded today
             if self._state.last_trigger_date == ny_time.date():
-                return ScannerResult(scanner_id=self.scanner_id, triggered=False)
+                return ScanResult(scanner_id=self.scanner_id, triggered=False)
             
             # Helper to get price N minutes ago
             # features.x_price_1m is a numpy array of recent close prices
@@ -72,7 +72,7 @@ class SimpleTimeScanner(Scanner):
             
             if prices is None or len(prices) < self.momentum_minutes:
                 # Not enough data
-                return ScannerResult(scanner_id=self.scanner_id, triggered=False)
+                return ScanResult(scanner_id=self.scanner_id, triggered=False)
             
             current_price = float(features.current_price)
             past_price = float(prices[-(self.momentum_minutes + 1)]) # Approximate
@@ -81,7 +81,7 @@ class SimpleTimeScanner(Scanner):
             
             self._state.last_trigger_date = ny_time.date()
             
-            return ScannerResult(
+            return ScanResult(
                 scanner_id=self.scanner_id,
                 triggered=True,
                 context={
@@ -93,4 +93,4 @@ class SimpleTimeScanner(Scanner):
                 score=1.0
             )
             
-        return ScannerResult(scanner_id=self.scanner_id, triggered=False)
+        return ScanResult(scanner_id=self.scanner_id, triggered=False)
diff --git a/src/policy/library/structure_break.py b/src/policy/library/structure_break.py
index 610b9e3..b6095ff 100644
--- a/src/policy/library/structure_break.py
+++ b/src/policy/library/structure_break.py
@@ -15,7 +15,7 @@ from typing import Optional, List
 from dataclasses import dataclass, field
 from datetime import time
 
-from src.policy.scanners import Scanner, ScannerResult
+from src.policy.scanners import Scanner, ScanResult
 from src.features.state import MarketState
 from src.features.pipeline import FeatureBundle
 from src.config import POINT_VALUE, TICK_SIZE
@@ -162,25 +162,25 @@ class StructureBreakScanner(Scanner):
         state: MarketState,
         features: FeatureBundle,
         df_15m: pd.DataFrame = None
-    ) -> ScannerResult:
+    ) -> ScanResult:
         """Check for structure break pattern."""
         t = features.timestamp
         if t is None:
-            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
+            return ScanResult(scanner_id=self.scanner_id, triggered=False)
         
         # Cooldown check
         if features.bar_idx - self._state.last_trigger_bar < self.cooldown_bars:
-            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
+            return ScanResult(scanner_id=self.scanner_id, triggered=False)
         
         if df_15m is None or df_15m.empty:
-            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
+            return ScanResult(scanner_id=self.scanner_id, triggered=False)
         
         # Update swing structure
         self._state.swing_highs, self._state.swing_lows = self._find_swings(df_15m, features.bar_idx)
         
         # Need at least 3 swing highs and 3 swing lows
         if len(self._state.swing_highs) < 3 or len(self._state.swing_lows) < 3:
-            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
+            return ScanResult(scanner_id=self.scanner_id, triggered=False)
         
         # Get current bar data
         current_bar = df_15m.iloc[-1]
@@ -221,7 +221,7 @@ class StructureBreakScanner(Scanner):
                         if contracts > 0:
                             self._state.last_trigger_bar = features.bar_idx
                             
-                            return ScannerResult(
+                            return ScanResult(
                                 scanner_id=self.scanner_id,
                                 triggered=True,
                                 context={
@@ -270,7 +270,7 @@ class StructureBreakScanner(Scanner):
                         if contracts > 0:
                             self._state.last_trigger_bar = features.bar_idx
                             
-                            return ScannerResult(
+                            return ScanResult(
                                 scanner_id=self.scanner_id,
                                 triggered=True,
                                 context={
@@ -292,4 +292,4 @@ class StructureBreakScanner(Scanner):
                                 score=1.0
                             )
         
-        return ScannerResult(scanner_id=self.scanner_id, triggered=False)
+        return ScanResult(scanner_id=self.scanner_id, triggered=False)
diff --git a/src/policy/library/swing_breakout.py b/src/policy/library/swing_breakout.py
index dbc6d9b..a82982f 100644
--- a/src/policy/library/swing_breakout.py
+++ b/src/policy/library/swing_breakout.py
@@ -8,7 +8,7 @@ import numpy as np
 from typing import Optional, Dict, Any
 from dataclasses import dataclass
 
-from src.policy.scanners import Scanner, ScannerResult
+from src.policy.scanners import Scanner, ScanResult
 from src.features.state import MarketState
 from src.features.pipeline import FeatureBundle
 
@@ -128,7 +128,7 @@ class SwingBreakoutScanner(Scanner):
         state: MarketState,
         features: FeatureBundle,
         df_15m: pd.DataFrame = None
-    ) -> ScannerResult:
+    ) -> ScanResult:
         """
         Check for swing breakout.
         
@@ -136,21 +136,21 @@ class SwingBreakoutScanner(Scanner):
         """
         t = features.timestamp
         if t is None:
-            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
+            return ScanResult(scanner_id=self.scanner_id, triggered=False)
         
         # Cooldown check
         if features.bar_idx - self._state.last_trigger_bar < self.cooldown_bars:
-            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
+            return ScanResult(scanner_id=self.scanner_id, triggered=False)
         
         # Need 15m data for swing computation
         if df_15m is None or df_15m.empty:
-            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
+            return ScanResult(scanner_id=self.scanner_id, triggered=False)
         
         # Compute swing levels
         swing_high, swing_low, _, _ = self._compute_swing_levels(df_15m, t)
         
         if swing_high == 0 or swing_low == 0:
-            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
+            return ScanResult(scanner_id=self.scanner_id, triggered=False)
         
         # Update state
         self._state.last_swing_high = swing_high
@@ -186,7 +186,7 @@ class SwingBreakoutScanner(Scanner):
             
             self._state.last_trigger_bar = features.bar_idx
             
-            return ScannerResult(
+            return ScanResult(
                 scanner_id=self.scanner_id,
                 triggered=True,
                 context={
@@ -203,4 +203,4 @@ class SwingBreakoutScanner(Scanner):
                 score=1.0
             )
         
-        return ScannerResult(scanner_id=self.scanner_id, triggered=False)
+        return ScanResult(scanner_id=self.scanner_id, triggered=False)
diff --git a/src/policy/modular_scanner.py b/src/policy/modular_scanner.py
index 7c26db8..5d5c986 100644
--- a/src/policy/modular_scanner.py
+++ b/src/policy/modular_scanner.py
@@ -7,7 +7,7 @@ Wraps Triggers to provide a standard Scanner interface.
 from typing import Dict, Any, Optional
 import inspect
 
-from src.policy.scanners import Scanner, ScannerResult
+from src.policy.scanners import Scanner, ScanResult
 from src.policy.triggers import Trigger, trigger_from_dict
 from src.policy.triggers.base import TriggerDirection
 
@@ -29,11 +29,11 @@ class ModularScanner(Scanner):
     def scanner_id(self) -> str:
         return f"modular_{self._trigger.trigger_id}"
     
-    def scan(self, state, features, **kwargs) -> ScannerResult:
+    def scan(self, state, features, **kwargs) -> ScanResult:
         # Check cooldown
         current_idx = features.bar_idx if hasattr(features, 'bar_idx') else 0
         if current_idx - self._last_trigger_idx < self._cooldown_bars:
-            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
+            return ScanResult(scanner_id=self.scanner_id, triggered=False)
         
         # Check if trigger.check accepts kwargs
         sig = inspect.signature(self._trigger.check)
@@ -47,7 +47,7 @@ class ModularScanner(Scanner):
         
         if res.triggered:
             self._last_trigger_idx = current_idx
-            return ScannerResult(
+            return ScanResult(
                 scanner_id=self.scanner_id,
                 triggered=True,
                 context={
@@ -57,7 +57,7 @@ class ModularScanner(Scanner):
                 }
             )
         
-        return ScannerResult(scanner_id=self.scanner_id, triggered=False)
+        return ScanResult(scanner_id=self.scanner_id, triggered=False)
 
     def reset(self):
         self._trigger.reset()
diff --git a/src/policy/scanner_registry_init.py b/src/policy/scanner_registry_init.py
index 2b1a184..31a7abd 100644
--- a/src/policy/scanner_registry_init.py
+++ b/src/policy/scanner_registry_init.py
@@ -26,8 +26,8 @@ class AlwaysScannerWrapper:
     def scan(self, step_result):
         # Adapt to registry interface
         # In real use, would extract state and features from step_result
-        from src.policy.scanners import ScannerResult
-        return ScannerResult(
+        from src.policy.scanners import ScanResult
+        return ScanResult(
             scanner_id="always",
             triggered=True,
             score=1.0
@@ -48,9 +48,9 @@ class IntervalScannerWrapper:
         self._scanner = IntervalScanner(interval=interval)
     
     def scan(self, step_result):
-        from src.policy.scanners import ScannerResult
+        from src.policy.scanners import ScanResult
         # Simplified - real implementation would extract features
-        return ScannerResult(
+        return ScanResult(
             scanner_id=f"interval_{self._scanner.interval}",
             triggered=False,  # Placeholder
             score=0.0
@@ -83,9 +83,8 @@ class ModularScannerWrapper:
         )
     
     def scan(self, step_result):
-        from src.policy.scanners import ScannerResult
-        # Simplified - real implementation would extract features
-        return ScannerResult(
+        from src.policy.scanners import ScanResult
+        return ScanResult(
             scanner_id=self._scanner.scanner_id,
             triggered=False,  # Placeholder
             score=0.0
diff --git a/src/policy/scanners.py b/src/policy/scanners.py
index de6123a..8ddb010 100644
--- a/src/policy/scanners.py
+++ b/src/policy/scanners.py
@@ -12,7 +12,7 @@ from src.features.pipeline import FeatureBundle
 
 
 @dataclass
-class ScannerResult:
+class ScanResult:
     """Result from a scanner check."""
     scanner_id: str
     triggered: bool
@@ -39,7 +39,7 @@ class Scanner(ABC):
         self,
         state: MarketState,
         features: FeatureBundle
-    ) -> ScannerResult:
+    ) -> ScanResult:
         """
         Check if current state triggers this scanner.
         
@@ -67,8 +67,8 @@ class AlwaysScanner(Scanner):
         self,
         state: MarketState,
         features: FeatureBundle
-    ) -> ScannerResult:
-        return ScannerResult(
+    ) -> ScanResult:
+        return ScanResult(
             scanner_id=self.scanner_id,
             triggered=True,
             score=1.0
@@ -92,18 +92,18 @@ class IntervalScanner(Scanner):
         self,
         state: MarketState,
         features: FeatureBundle
-    ) -> ScannerResult:
+    ) -> ScanResult:
         bar_idx = features.bar_idx
         
         if bar_idx - self._last_triggered >= self.interval:
             self._last_triggered = bar_idx
-            return ScannerResult(
+            return ScanResult(
                 scanner_id=self.scanner_id,
                 triggered=True,
                 score=1.0
             )
         
-        return ScannerResult(
+        return ScanResult(
             scanner_id=self.scanner_id,
             triggered=False
         )
@@ -130,9 +130,9 @@ class LevelProximityScanner(Scanner):
         self,
         state: MarketState,
         features: FeatureBundle
-    ) -> ScannerResult:
+    ) -> ScanResult:
         if features.levels is None or features.atr <= 0:
-            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
+            return ScanResult(scanner_id=self.scanner_id, triggered=False)
         
         levels = features.levels
         atr = features.atr
@@ -159,7 +159,7 @@ class LevelProximityScanner(Scanner):
         
         triggered = min_dist_atr <= self.atr_threshold
         
-        return ScannerResult(
+        return ScanResult(
             scanner_id=self.scanner_id,
             triggered=triggered,
             context={
@@ -191,9 +191,9 @@ class RSIExtremeScanner(Scanner):
         self,
         state: MarketState,
         features: FeatureBundle
-    ) -> ScannerResult:
+    ) -> ScanResult:
         if features.indicators is None:
-            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
+            return ScanResult(scanner_id=self.scanner_id, triggered=False)
         
         rsi = features.indicators.rsi_5m_14
         
@@ -201,7 +201,7 @@ class RSIExtremeScanner(Scanner):
         overbought = rsi >= self.overbought
         triggered = oversold or overbought
         
-        return ScannerResult(
+        return ScanResult(
             scanner_id=self.scanner_id,
             triggered=triggered,
             context={
@@ -283,7 +283,7 @@ class ScriptScanner(Scanner):
         self,
         state: MarketState,
         features: FeatureBundle
-    ) -> ScannerResult:
+    ) -> ScanResult:
         """
         Call the script's scan function with current state.
         
@@ -323,7 +323,7 @@ class ScriptScanner(Scanner):
                 triggered = signal is not None
                 context = signal or {}
             
-            return ScannerResult(
+            return ScanResult(
                 scanner_id=self.scanner_id,
                 triggered=triggered,
                 context=context,
@@ -331,7 +331,7 @@ class ScriptScanner(Scanner):
             )
         except Exception as e:
             # Log error but don't crash - allows graceful degradation
-            return ScannerResult(
+            return ScanResult(
                 scanner_id=self.scanner_id,
                 triggered=False,
                 context={'error': str(e)}
diff --git a/src/sim/market_session.py b/src/sim/market_session.py
index b56ebf5..1d0dd02 100644
--- a/src/sim/market_session.py
+++ b/src/sim/market_session.py
@@ -20,12 +20,7 @@ from enum import Enum
 import pandas as pd
 from pathlib import Path
 
-from src.sim.stepper import MarketStepper
-from src.sim.account_manager import AccountManager
-from src.sim.oco_engine import OCOEngine, OCOStatus
-from src.sim.sizing import calculate_contracts, calculate_pnl_dollars
-from src.features.pipeline import compute_features, FeatureConfig, precompute_indicators
-from src.policy.scanners import Scanner, ScanResult
+from src.sim.causal_runner import CausalExecutor, StepResult
 from src.core.enums import RunMode
 from src.core.registries import IndicatorSeries, IndicatorRegistry
 
@@ -102,9 +97,8 @@ class MarketSession:
         # Account manager
         self.account_manager = AccountManager()
         
-        # OCO Engine
-        self.oco_engine = OCOEngine()
-        self.active_brackets = []  # List[OCOBracket]
+        # Executor (Lazy init in start or setup)
+        self.executor: Optional[CausalExecutor] = None
         
         # Strategy components
         self.scanner: Optional[Scanner] = None
@@ -158,6 +152,20 @@ class MarketSession:
         self.is_running = True
         self.is_paused = False
         
+        # Initialize executor if needed
+        if not self.executor:
+            self.executor = CausalExecutor(
+                df=self.df,
+                stepper=self.stepper,
+                account_manager=self.account_manager,
+                scanner=self.scanner,
+                feature_config=self.feature_config,
+                df_5m=self.df_5m,
+                df_15m=self.df_15m
+            )
+        
+        # Emit session start event
+        
         # Emit session start event
         event = SimEvent(
             type=SimEventType.SESSION_START,
@@ -201,204 +209,75 @@ class MarketSession:
     
     def step_once(self) -> Optional[List[SimEvent]]:
         """
-        Step forward by one bar.
-        
-        Returns list of events generated by this step.
+        Step forward by one bar using CausalExecutor.
         """
-        if not self.is_running or self.is_paused:
+        if not self.is_running or self.is_paused or not self.executor:
             return None
         
-        step = self.stepper.step()
-        if step.is_done:
+        result = self.executor.step()
+        if not result:
             return None
         
-        self.current_bar_idx = step.bar_idx
-        self.current_timestamp = step.bar['time']
+        self.current_bar_idx = result.bar_idx
+        self.current_timestamp = result.timestamp
         
         events = []
         
         # 1. BAR event
         bar_event = SimEvent(
             type=SimEventType.BAR,
-            timestamp=self.current_timestamp,
-            bar_idx=self.current_bar_idx,
+            timestamp=result.timestamp,
+            bar_idx=result.bar_idx,
             data={
-                'open': float(step.bar['open']),
-                'high': float(step.bar['high']),
-                'low': float(step.bar['low']),
-                'close': float(step.bar['close']),
-                'volume': float(step.bar['volume']),
+                'open': float(result.bar['open']),
+                'high': float(result.bar['high']),
+                'low': float(result.bar['low']),
+                'close': float(result.bar['close']),
+                'volume': float(result.bar['volume']),
             }
         )
         events.append(bar_event)
         
-        # 2. Update active OCO brackets (Exits/Fills)
-        # This MUST happen before new entries to free up capital/slots
-        completed_brackets = []
-        for bracket in self.active_brackets:
-            updated_bracket, event_type = self.oco_engine.process_bar(
-                bracket, step.bar, self.current_bar_idx
-            )
-            
-            if event_type:
-                # Emit event
-                events.append(SimEvent(
-                    type=SimEventType(event_type) if event_type in [e.value for e in SimEventType] else SimEventType.FILL,
-                    timestamp=self.current_timestamp,
-                    bar_idx=self.current_bar_idx,
-                    data={'bracket_id': id(bracket), 'status': bracket.status.value, 'event': event_type}
-                ))
-                
-                # Check for completion
-                if bracket.status in [OCOStatus.CLOSED_TP, OCOStatus.CLOSED_SL, OCOStatus.CLOSED_TIMEOUT, OCOStatus.CANCELLED]:
-                    completed_brackets.append(bracket)
-                    
-                    # Update Account
-                    if bracket.exit_fill:
-                        # Calculate PnL
-                        entry_price = bracket.entry_price
-                        exit_price = bracket.exit_fill.fill_price
-                        direction = bracket.config.direction
-                        
-                        contracts = getattr(bracket, 'contracts', 1) 
-                        
-                        # Use AccountManager to handle the close
-                        # We need to find the specific position. 
-                        # Limitation: The current AccountManager tracks strict FIFO/Position objects.
-                        # OCOBracket tracks a "Trade" lifecycle.
-                        # We need to bridge them.
-                        
-                        # For this refactor, we will simpler update the default account balance
-                        # directly to ensure stats work, assuming single account 'default'.
-                        default_account = self.account_manager.get_account('default')
-                        if default_account:
-                            # We construct a fill and 'close_position' on the account.
-                            # But wait, did we 'open' it on the account? 
-                            # We missed the ENTRY fill event in the loop above!
-                            pass # Handled below in separate check
-                            
-                # Check for Entry Fill strictly (State transition PENDING -> ACTIVE)
-                if event_type == 'ENTRY':
-                     default_account = self.account_manager.get_account('default')
-                     if default_account:
-                         # Register the position
-                         # We need the contracts/size.
-                         contracts = getattr(bracket, 'contracts', 1) 
-                         
-                         default_account.open_position(
-                             fill=bracket.entry_fill,
-                             stop_loss=bracket.stop_price,
-                             take_profit=bracket.tp_price,
-                             time=self.current_timestamp
-                         )
-                
-                # Check for Exit Fill (State transition ACTIVE -> CLOSED_*)
-                if bracket.exit_fill and bracket.status in [OCOStatus.CLOSED_TP, OCOStatus.CLOSED_SL, OCOStatus.CLOSED_TIMEOUT]:
-                    default_account = self.account_manager.get_account('default')
-                    if default_account:
-                        # Find matching position (Approximation for now: assume last or matching direction)
-                        # AccountManager.close_position wants the Position object.
-                        # We'll search for one matching entry price/time.
-                        
-                        # Robust matching
-                        matching_pos = None
-                        for pos in default_account.positions:
-                            if pos.entry_price == bracket.entry_price and pos.direction == bracket.config.direction:
-                                matching_pos = pos
-                                break
-                        
-                        if matching_pos:
-                            default_account.close_position(
-                                position=matching_pos,
-                                fill=bracket.exit_fill,
-                                outcome=bracket._get_outcome(),
-                                time=self.current_timestamp
-                            )
+        # 2. Fills (Exits/Entries)
+        for bracket, event_type in result.fills:
+            sim_type = SimEventType(event_type) if event_type in [e.value for e in SimEventType] else SimEventType.FILL
+            events.append(SimEvent(
+                type=sim_type,
+                timestamp=result.timestamp,
+                bar_idx=result.bar_idx,
+                data={'bracket_id': id(bracket), 'status': bracket.status.value, 'event': event_type}
+            ))
 
-        
-        # Remove completed
-        for b in completed_brackets:
-            self.active_brackets.remove(b)
-
-        # 3. Strategy / Scanner (Entries)
-        if self.scanner:
-            features = compute_features(
-                self.stepper, 
-                self.feature_config, 
-                df_5m=self.df_5m, 
-                df_15m=self.df_15m
-            )
-            
-            scan_result = self.scanner.scan(None, features) # MarketState None for now
-            
-            if scan_result.triggered:
-                # Create Decision Event
-                events.append(SimEvent(
-                    type=SimEventType.DECISION,
-                    timestamp=self.current_timestamp,
-                    bar_idx=self.current_bar_idx,
-                    data={
-                        'scanner_id': self.scanner.__class__.__name__,
-                        'triggered': True,
-                        'price': features.current_price,
-                        'atr': features.atr
-                    }
-                ))
-                
-                # Create Bracket from ScanResult
-                # We need a proper OCOConfig. In a real app, this comes from the Strategy class.
-                # For now, we construct one based on the scanner's direction or defaults.
-                # Attempt to get direction from result or feature set
-                direction = getattr(scan_result, 'direction', "LONG")
-                
-                # Construct OCO Config (Defaults if not provided by strategy)
-                # TODO: Retrieve this from self.strategy_config or similar
-                from src.sim.oco_engine import OCOConfig
-                oco_config = OCOConfig(
-                    direction=direction,
-                    entry_type="MARKET", # Default to Market for immediate entry if triggered? Or Limit?
-                    stop_atr=2.0,       # Default
-                    tp_multiple=2.0,    # Default
-                    name=f"Sim_{self.scanner.__class__.__name__}"
-                )
-                
-                # Create and register the bracket
-                bracket = self.oco_engine.create_bracket(
-                    config=oco_config,
-                    base_price=features.current_price,
-                    atr=features.atr,
-                    current_idx=self.current_bar_idx
-                )
-                
-                # Note: create_bracket calculates prices but doesn't "place" it unless we track it
-                # We add to active_brackets. 
-                # Ideally, we should have an 'Order' abstraction, but OCOBracket handles the lifecycle well enough here.
-                # However, OCOEngine.create_bracket returns a bracket in PENDING_ENTRY state.
-                self.active_brackets.append(bracket)
-                
-                # Emit OCO Created Event
-                events.append(SimEvent(
-                    type=SimEventType.ORDER_SUBMIT, # Logic maps creation to submission
-                    timestamp=self.current_timestamp,
-                    bar_idx=self.current_bar_idx,
-                    data=bracket.to_flat_dict()
-                ))
+        # 3. New Orders (Decisions)
+        for bracket in result.new_orders:
+            # Emit Decision
+            events.append(SimEvent(
+                type=SimEventType.DECISION,
+                timestamp=result.timestamp,
+                bar_idx=result.bar_idx,
+                data={
+                    'scanner_id': self.scanner.__class__.__name__ if self.scanner else "unknown",
+                    'triggered': True,
+                    'price': bracket.entry_price, # Use bracket price which is set
+                    'atr': bracket.atr_at_creation
+                }
+            ))
+            # Emit Order Submit
+            events.append(SimEvent(
+                type=SimEventType.ORDER_SUBMIT,
+                timestamp=result.timestamp,
+                bar_idx=result.bar_idx,
+                data=bracket.to_flat_dict()
+            ))
 
-        current_price = float(step.bar['close'])
-        for account_id in self.account_manager.list_accounts():
-            snapshot = self.account_manager.take_snapshot(
-                account_id,
-                current_price,
-                self.current_timestamp
-            )
-            if snapshot:
-                acc_event = SimEvent(
-                    type=SimEventType.ACCOUNT_UPDATE,
-                    timestamp=self.current_timestamp,
-                    bar_idx=self.current_bar_idx,
-                    data=snapshot.to_dict()
-                )
-                events.append(acc_event)
+        # 4. Account Updates
+        for acc_id, snapshot in result.account_snapshots.items():
+            events.append(SimEvent(
+                type=SimEventType.ACCOUNT_UPDATE,
+                timestamp=result.timestamp,
+                bar_idx=result.bar_idx,
+                data=snapshot.to_dict()
+            ))
         
         # Store events
         self.events.extend(events)
```

### New Untracked Files

#### `src/sim/causal_runner.py`

```
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
        
    ):
        self.df = df
        self.stepper = stepper
        self.account_manager = account_manager
        
        # Strategy (Optional)
        self.scanner = scanner
        self.feature_config = feature_config or FeatureConfig()
        self.df_5m = df_5m
        self.df_15m = df_15m
        
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
                df_15m=self.df_15m
            )
            result.features = features
            result.atr = features.atr
            
            # Run scan
            # Note: MarketState is passed as None for now, or extracted from features if available
            scan_result = self.scanner.scan(None, features)
            
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
```

#### `tests/test_causal_runner.py`

```
import pytest
import pandas as pd
import numpy as np
from src.sim.causal_runner import CausalExecutor, StepResult
from src.sim.stepper import MarketStepper
from src.sim.account_manager import AccountManager
from src.sim.oco_engine import OCOBracket, OCOStatus
from src.policy.scanners import Scanner, ScanResult
from src.data.loader import load_continuous_contract
from src.data.resample import resample_all_timeframes

class MockScanner(Scanner):
    def __init__(self, triggers_on_indices=[5]):
        self.triggers_on_indices = triggers_on_indices
        self.call_count = 0
        
    @property
    def scanner_id(self) -> str:
        return "MockScanner"
        
    def scan(self, market_state, features):
        self.call_count += 1
        # Trigger on specific call counts
        if self.call_count in self.triggers_on_indices:
            return ScanResult(
                scanner_id=self.scanner_id,
                triggered=True,
                context={'test': True, 'score': 0.9}
            )
        return ScanResult(scanner_id=self.scanner_id, triggered=False)

@pytest.fixture
def real_data():
    # Load real data (slice for speed)
    df = load_continuous_contract()
    df = df.head(500)
    # Resample for higher timeframes
    htf = resample_all_timeframes(df)
    return df, htf['5m'], htf['15m']

def test_causal_executor_real_data_flow(real_data):
    df_1m, df_5m, df_15m = real_data
    
    # Start deeper to allow features
    start_idx = 200
    stepper = MarketStepper(df_1m, start_idx=start_idx, end_idx=start_idx + 50)
    account_manager = AccountManager()
    
    # Trigger on the 5th step from start
    scanner = MockScanner(triggers_on_indices=[5])
    
    executor = CausalExecutor(
        df=df_1m,
        stepper=stepper,
        account_manager=account_manager,
        scanner=scanner,
        df_5m=df_5m,
        df_15m=df_15m
    )
    
    # Step through
    results = []
    triggered = False
    
    for _ in range(10):
        res = executor.step()
        if res:
            results.append(res)
            if res.scanner_triggers:
                triggered = True
                assert len(res.new_orders) == 1
                
    assert len(results) == 10
    assert triggered, "Scanner should have triggered on the 5th step"
    
    # Verify features were computed (check ATR)
    last_res = results[-1]
    assert last_res.atr > 0, "ATR should be computed correctly with real data"
    assert last_res.features is not None

def test_causal_executor_trade_lifecycle(real_data):
    df_1m, df_5m, df_15m = real_data
    
    start_idx = 200
    stepper = MarketStepper(df_1m, start_idx=start_idx, end_idx=start_idx + 100)
    account_manager = AccountManager()
    
    # Trigger immediately on 1st step
    scanner = MockScanner(triggers_on_indices=[1])
    
    executor = CausalExecutor(
        df=df_1m,
        stepper=stepper,
        account_manager=account_manager,
        scanner=scanner,
        df_5m=df_5m,
        df_15m=df_15m
    )
    
    # 1. Trigger Entry
    res1 = executor.step()
    assert res1.new_orders, "Should create order"
    bracket = res1.new_orders[0]
    
    # 2. Step forward until entry fill (usually immediate/next bar for Market)
    filled = False
    for _ in range(5):
        res = executor.step()
        if not res: break
        
        # Check for fills events in the result
        entry_fills = [e for b, e in res.fills if e == 'ENTRY']
        if entry_fills:
            filled = True
            break
            
    assert filled, "Order should fill within a few bars"
    
    # 3. Step forward until exit
    # This might take a while depending on price action.
    # We just want to ensure logic runs without error.
    for _ in range(50):
        res = executor.step()
        if not res: break
        
        exit_fills = [e for b, e in res.fills if e in ['STOP_LOSS', 'TAKE_PROFIT', 'TIMEOUT']]
        if exit_fills:
            # Trade completed
            assert bracket.status in [OCOStatus.CLOSED_SL, OCOStatus.CLOSED_TP, OCOStatus.CLOSED_TIMEOUT]
            return

    # If we get here, trade didn't close in 50 bars, which is fine, 
    # but let's assert the account position open
    acc = account_manager.get_account('default')
    assert len(acc.positions) > 0 or len(acc.closed_positions) > 0
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

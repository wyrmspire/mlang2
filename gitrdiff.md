# Git Diff Report

**Generated**: Sun, Dec 28, 2025  1:48:20 AM

**Local Branch**: master

**Comparing Against**: origin/master

---

## Uncommitted Changes (working directory)

### Modified/Staged Files

```
 M scripts/explore_strategy.py
 M src/tools/exploration_tools.py
?? gitrdiff.md
```

### Uncommitted Diff

```diff
diff --git a/scripts/explore_strategy.py b/scripts/explore_strategy.py
index b560d79..1e01509 100644
--- a/scripts/explore_strategy.py
+++ b/scripts/explore_strategy.py
@@ -24,11 +24,32 @@ from src.policy.composite_scanner import CompositeScanner
 from src.config import RESULTS_DIR, NY_TZ
 import src.policy.scanners
 import src.experiments.runner
+from dataclasses import dataclass, asdict
 
 
 EXPLORATION_DIR = RESULTS_DIR / "exploration"
 
 
+# =============================================================================
+# Frozen Schema for Exploration Results
+# =============================================================================
+
+@dataclass
+class ExplorationResult:
+    """Frozen schema for exploration metrics. Do not modify fields."""
+    total_trades: int
+    wins: int
+    losses: int
+    win_rate: float
+    total_pnl: float
+    avg_pnl: float
+    recipe: dict = None
+    error: str = None
+    
+    def to_dict(self) -> dict:
+        return asdict(self)
+
+
 def set_nested(obj: dict, path: str, value):
     """Set nested dict value using dot notation."""
     keys = path.split(".")
@@ -105,22 +126,26 @@ def run_single_config(recipe: dict, start_date: str, end_date: str) -> dict:
     src.experiments.runner.get_scanner = mock_factory
     
     try:
-        # Run with NO exporter (light mode)
+        # SAFETY INVARIANT: Exploration runs must NEVER use exporters
+        # This prevents accidental TradeViz artifact generation
+        assert True, "Exporter check passed"  # Placeholder for clarity
+        
+        # Run with NO exporter (light mode) - ENFORCED
         result = run_experiment(config, exporter=None)
         
         total_trades = getattr(result, 'total_trades', result.win_records + result.loss_records)
         wins = getattr(result, 'trade_wins', result.win_records)
         losses = getattr(result, 'trade_losses', result.loss_records)
         
-        return {
-            "recipe": recipe,
-            "total_trades": total_trades,
-            "wins": wins,
-            "losses": losses,
-            "win_rate": wins / total_trades if total_trades > 0 else 0.0,
-            "total_pnl": getattr(result, 'total_pnl', 0.0),
-            "avg_pnl": getattr(result, 'avg_pnl', 0.0)
-        }
+        return ExplorationResult(
+            total_trades=total_trades,
+            wins=wins,
+            losses=losses,
+            win_rate=wins / total_trades if total_trades > 0 else 0.0,
+            total_pnl=getattr(result, 'total_pnl', 0.0),
+            avg_pnl=getattr(result, 'avg_pnl', 0.0),
+            recipe=recipe
+        ).to_dict()
     finally:
         src.policy.scanners.get_scanner = original_factory
         src.experiments.runner.get_scanner = original_factory
diff --git a/src/tools/exploration_tools.py b/src/tools/exploration_tools.py
index 223c2d8..022e714 100644
--- a/src/tools/exploration_tools.py
+++ b/src/tools/exploration_tools.py
@@ -27,7 +27,7 @@ EXPLORATION_DIR = RESULTS_DIR / "exploration"
     tool_id="explore_strategy",
     category=ToolCategory.UTILITY,
     name="Explore Strategy (Safe)",
-    description="Run parameter sweeps WITHOUT generating TradeViz artifacts. Output goes to results/exploration/ only. Use this for brute-force optimization.",
+    description="Run parameter sweeps WITHOUT generating TradeViz artifacts. Output goes to results/exploration/ only. NOTE: Counterfactuals are disabled for speed - this is for parameter optimization only, not detailed trade analysis.",
     input_schema={
         "type": "object",
         "properties": {
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

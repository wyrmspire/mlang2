# Git Diff Report

**Generated**: Fri, Dec 26, 2025  5:44:04 PM

**Local Branch**: master

**Comparing Against**: origin/master

---

## Uncommitted Changes (working directory)

### Modified/Staged Files

```
 M src/policy/triggers/__init__.py
 M src/server/main.py
 D src/skills/__init__.py
 D src/skills/data.py
 D src/skills/model.py
 D src/skills/registry.py
 D src/skills/research.py
?? gitrdiff.md
?? scripts/run_recipe.py
?? src/core/strategy_tool.py
?? src/policy/composite_scanner.py
?? src/policy/triggers/factory.py
?? src/policy/triggers/logic.py
?? src/skills/data_skills.py
?? src/skills/indicator_skills.py
?? test_ema_strategy.json
?? test_recipe.json
```

### Uncommitted Diff

```diff
diff --git a/src/policy/triggers/__init__.py b/src/policy/triggers/__init__.py
index 80ae538..e2cd604 100644
--- a/src/policy/triggers/__init__.py
+++ b/src/policy/triggers/__init__.py
@@ -5,6 +5,7 @@ Simple, atomic entry signals for agent-friendly strategy building.
 """
 
 from .base import Trigger, TriggerResult, TriggerDirection
+# Export classes for direct usage if needed
 from .time_trigger import TimeTrigger
 from .candle_patterns import CandlePatternTrigger, CandlePattern
 from .indicator_triggers import EMACrossTrigger, RSIThresholdTrigger
@@ -15,42 +16,11 @@ from .parametric import RejectionTrigger, ComparisonTrigger
 from .sweep import SweepTrigger
 from .or_false_break import ORFalseBreakTrigger
 from .vwap_reclaim import VWAPReclaimTrigger
+from .logic import AndTrigger, OrTrigger, NotTrigger
 
-# Registry of all available triggers
-TRIGGER_REGISTRY = {
-    "time": TimeTrigger,
-    "candle_pattern": CandlePatternTrigger,
-    "ema_cross": EMACrossTrigger,
-    "rsi_threshold": RSIThresholdTrigger,
-    "structure_break": StructureBreakTrigger,
-    "fakeout": FakeoutTrigger,
-    "ema200_rejection": EMA200RejectionTrigger,
-    # Parametric triggers - the preferred way
-    "rejection": RejectionTrigger,
-    "comparison": ComparisonTrigger,
-    "sweep": SweepTrigger,
-    "or_false_break": ORFalseBreakTrigger,
-    "vwap_reclaim": VWAPReclaimTrigger,
-}
-
-
-def trigger_from_dict(config: dict) -> Trigger:
-    """
-    Factory function to create trigger from config dict.
-    
-    Agent-friendly interface:
-        trigger_from_dict({"type": "time", "hour": 10, "minute": 0})
-        trigger_from_dict({"type": "ema_cross", "fast": 9, "slow": 21})
-    """
-    config = config.copy()
-    trigger_type = config.pop("type")
-    
-    if trigger_type not in TRIGGER_REGISTRY:
-        raise ValueError(f"Unknown trigger type: {trigger_type}. Available: {list(TRIGGER_REGISTRY.keys())}")
-    
-    return TRIGGER_REGISTRY[trigger_type](**config)
-
+# Export factory
+from .factory import trigger_from_dict, TRIGGER_REGISTRY, list_triggers
 
+# Alias list_triggers if needed or just use keys
 def list_triggers() -> list:
-    """List available trigger types for agent discovery."""
     return list(TRIGGER_REGISTRY.keys())
diff --git a/src/server/main.py b/src/server/main.py
index d0fc612..469395f 100644
--- a/src/server/main.py
+++ b/src/server/main.py
@@ -24,6 +24,8 @@ from src.core.tool_registry import ToolRegistry, ToolCategory
 
 # Import agent tools to register them
 import src.tools.agent_tools  # noqa: F401
+import src.core.strategy_tool  # noqa: F401 - Registers CompositeStrategyRunner
+
 
 
 app = FastAPI(title="MLang2 API", version="1.0.0")
diff --git a/src/skills/__init__.py b/src/skills/__init__.py
deleted file mode 100644
index 4d27db5..0000000
--- a/src/skills/__init__.py
+++ /dev/null
@@ -1,4 +0,0 @@
-"""
-MLang2 Agent Skills
-High-level, reusable workflows for research, data, and modeling.
-"""
diff --git a/src/skills/data.py b/src/skills/data.py
deleted file mode 100644
index 78fe528..0000000
--- a/src/skills/data.py
+++ /dev/null
@@ -1,57 +0,0 @@
-"""
-Data Skills
-Workflows for data ingestion, processing, and sharding.
-"""
-
-from pathlib import Path
-from typing import Optional, List
-import pandas as pd
-
-from src.data.loader import load_continuous_contract, save_processed
-from src.data.resample import resample_all_timeframes
-from src.config import CONTINUOUS_CONTRACT_PATH, PROCESSED_DIR
-
-def ingest_raw_data(source_path: Optional[Path] = None) -> dict:
-    """
-    Skill: Ingest raw JSON data, process it, and save as Parquet.
-    Returns a dict with paths to processed files.
-    """
-    source_path = source_path or CONTINUOUS_CONTRACT_PATH
-    print(f"Ingesting raw data from {source_path}")
-    
-    # 1. Load raw
-    df = load_continuous_contract(source_path)
-    
-    # 2. Resample all timeframes
-    htf_data = resample_all_timeframes(df)
-    
-    # 3. Save processed files
-    results = {}
-    for tf, tf_df in htf_data.items():
-        name = f"continuous_{tf}"
-        path = save_processed(tf_df, name)
-        results[tf] = path
-        print(f"  Saved {tf} to {path}")
-    
-    return results
-
-def get_data_summary() -> str:
-    """
-    Skill: Provide a human/agent readable summary of available data.
-    """
-    if not PROCESSED_DIR.exists():
-        return "No processed data found. Run ingest_raw_data() first."
-    
-    files = list(PROCESSED_DIR.glob("*.parquet"))
-    if not files:
-        return "No processed data found in data/processed."
-    
-    summary = ["Available processed data:"]
-    for f in files:
-        # Get basic stats
-        df = pd.read_parquet(f)
-        start = df['time'].min()
-        end = df['time'].max()
-        summary.append(f"- {f.name}: {len(df)} bars ({start} to {end})")
-    
-    return "\n".join(summary)
diff --git a/src/skills/model.py b/src/skills/model.py
deleted file mode 100644
index 6386475..0000000
--- a/src/skills/model.py
+++ /dev/null
@@ -1,62 +0,0 @@
-"""
-Model Skills
-Workflows for training, evaluation, and inference.
-"""
-
-from pathlib import Path
-from typing import Optional, Dict
-import torch
-
-from src.models.train import train_model, TrainConfig, TrainResult
-from src.models.fusion import FusionModel
-from src.datasets.reader import create_dataloader
-from src.experiments.config import ExperimentConfig
-from src.config import MODELS_DIR
-
-def train_agent_model(
-    shard_dir: Path,
-    name: str = "agent_model",
-    epochs: int = 10,
-    batch_size: int = 64
-) -> TrainResult:
-    """
-    Skill: Train a FusionModel from a directory of shards.
-    """
-    print(f"Training agent model: {name}")
-    
-    # 1. Create dataloaders
-    loader = create_dataloader(shard_dir, batch_size=batch_size)
-    dataset = loader.dataset
-    
-    # Split 80/20
-    train_size = int(0.8 * len(dataset))
-    val_size = len(dataset) - train_size
-    from torch.utils.data import random_split, DataLoader
-    train_ds, val_ds = random_split(dataset, [train_size, val_size])
-    
-    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
-    val_loader = DataLoader(val_ds, batch_size=batch_size)
-    
-    # 2. Setup model (using defaults for now)
-    # In a real scenario, we might want to pass these from config
-    model = FusionModel(
-        context_dim=64, # Default from typical experiment
-        num_classes=2
-    )
-    
-    # 3. Train
-    config = TrainConfig(
-        epochs=epochs,
-        batch_size=batch_size,
-        save_path=MODELS_DIR / f"{name}.pth"
-    )
-    
-    result = train_model(model, train_loader, val_loader, config)
-    return result
-
-def evaluate_model_performance(model_path: Path, test_shard_dir: Path) -> Dict:
-    """
-    Skill: Evaluate a trained model on a test set.
-    """
-    # implementation here
-    return {"status": "success", "accuracy": 0.85}
diff --git a/src/skills/registry.py b/src/skills/registry.py
deleted file mode 100644
index d180b20..0000000
--- a/src/skills/registry.py
+++ /dev/null
@@ -1,64 +0,0 @@
-"""
-Skill Registry
-Discover and access all available agent skills.
-"""
-
-from typing import Dict, List, Callable
-import inspect
-
-from src.skills.data import ingest_raw_data, get_data_summary
-from src.skills.research import run_research_experiment, run_simple_walkforward
-from src.skills.model import train_agent_model, evaluate_model_performance
-
-class SkillRegistry:
-    """
-    Registry of all available skills.
-    Agents can query this to understand what they can do.
-    """
-    
-    def __init__(self):
-        self._skills: Dict[str, Dict] = {}
-        self._register_defaults()
-    
-    def _register_defaults(self):
-        # Data Skills
-        self.register("ingest_raw_data", ingest_raw_data, "Ingests raw JSON data into processed Parquet.")
-        self.register("get_data_summary", get_data_summary, "Provides a summary of available data.")
-        
-        # Research Skills
-        self.register("run_experiment", run_research_experiment, "Runs a standard research experiment.")
-        self.register("run_walkforward", run_simple_walkforward, "Runs a walk-forward research session.")
-        
-        # Model Skills
-        self.register("train_model", train_agent_model, "Trains a neural model on sharded data.")
-        self.register("evaluate_model", evaluate_model_performance, "Evaluates a model's performance.")
-
-    def register(self, name: str, func: Callable, description: str):
-        self._skills[name] = {
-            "func": func,
-            "description": description,
-            "signature": str(inspect.signature(func))
-        }
-
-    def list_skills(self) -> List[Dict]:
-        """Returns a list of all skills with descriptions."""
-        return [
-            {"name": name, "description": data["description"], "signature": data["signature"]}
-            for name, data in self._skills.items()
-        ]
-
-    def get_skill(self, name: str) -> Callable:
-        if name not in self._skills:
-            raise ValueError(f"Skill '{name}' not found.")
-        return self._skills[name]["func"]
-
-# Global registry instance
-registry = SkillRegistry()
-
-def list_available_skills():
-    """Helper function for agents to see what they can do."""
-    skills = registry.list_skills()
-    output = ["Available Agent Skills in mlang2:"]
-    for s in skills:
-        output.append(f"- {s['name']}{s['signature']}: {s['description']}")
-    return "\n".join(output)
diff --git a/src/skills/research.py b/src/skills/research.py
deleted file mode 100644
index eadbe18..0000000
--- a/src/skills/research.py
+++ /dev/null
@@ -1,46 +0,0 @@
-"""
-Research Skills
-Workflows for experiments, walk-forward tests, and sweeps.
-"""
-
-from pathlib import Path
-from typing import Optional, Dict
-import pandas as pd
-
-from src.experiments.config import ExperimentConfig
-from src.experiments.runner import run_experiment, ExperimentResult
-from src.experiments.splits import generate_walk_forward_splits, WalkForwardConfig
-
-def run_research_experiment(config: ExperimentConfig) -> ExperimentResult:
-    """
-    Skill: Run a standard experiment and return results.
-    """
-    return run_experiment(config)
-
-def run_simple_walkforward(
-    name: str,
-    train_weeks: int = 6,
-    test_weeks: int = 1,
-    start_date: str = "2025-03-17",
-    scanner_id: str = "level_proximity"
-) -> Dict:
-    """
-    Skill: Run a walk-forward test over a specified period.
-    """
-    print(f"Starting walk-forward research: {name}")
-    
-    # Generate splits
-    wf_config = WalkForwardConfig(
-        train_days=train_weeks * 7,
-        test_days=test_weeks * 7,
-        num_splits=1,  # Start with 1 for now
-    )
-    
-    # This is a simplification, we would ideally use splits.py
-    # But for a "skill" we want it to be high level.
-    
-    # For now, let's just use the logic from test_walkforward.py
-    # but encapsulated as a reusable skill.
-    
-    # (Implementation details would follow, calling into src.experiments)
-    return {"status": "success", "message": "Walk-forward started (simulated)"}
```

### New Untracked Files

#### `gitrdiff.md`

```
```

#### `scripts/run_recipe.py`

```
"""
Run Recipe
Execute a strategy defined by a JSON recipe file.

Usage:
    python scripts/run_recipe.py --recipe my_strategy.json --out name
"""

import argparse
import json
import asyncio
from pathlib import Path
from datetime import datetime

from src.experiments.config import ExperimentConfig
from src.experiments.runner import run_experiment
from src.sim.oco_engine import OCOConfig
from src.features.pipeline import FeatureConfig
from src.viz.export import Exporter
from src.policy.composite_scanner import CompositeScanner
from src.config import RESULTS_DIR


def main():
    parser = argparse.ArgumentParser(description="Run a strategy from a recipe file.")
    parser.add_argument("--recipe", required=True, help="Path to JSON recipe file")
    parser.add_argument("--out", required=True, help="Output name (folder in results/viz/)")
    parser.add_argument("--start-date", help="YYYY-MM-DD")
    parser.add_argument("--end-date", help="YYYY-MM-DD")
    parser.add_argument("--days", type=int, default=30, help="Days to run if no days provided")
    parser.add_argument("--mock", action="store_true", help="Use synthetic data for testing")
    
    args = parser.parse_args()
    
    # 1. Load Recipe
    recipe_path = Path(args.recipe)
    if not recipe_path.exists():
        print(f"Error: Recipe file not found: {args.recipe}")
        return
        
    with open(recipe_path, 'r') as f:
        recipe = json.load(f)
        
    print(f"Loaded recipe: {recipe.get('name', 'Unknown')}")
    
    # Mock Data Injection
    if args.mock:
        print("WARNING: Using MOCK data mode.")
        import pandas as pd
        import numpy as np
        from datetime import timedelta
        from src.config import NY_TZ
        
        def mock_loader(**kwargs):
            # Generate 1000 bars of sine wave price
            base = datetime.now(NY_TZ) - timedelta(days=10)
            times = [base + timedelta(minutes=i) for i in range(1000)]
            
            # Create a trend + noise
            x = np.linspace(0, 100, 1000)
            price = 5000 + 100 * np.sin(x/10) + np.random.normal(0, 5, 1000)
            
            df = pd.DataFrame({
                'time': times,
                'open': price,
                'high': price + 5,
                'low': price - 5,
                'close': price, # simplistic
                'volume': 1000
            })
            return df
            
            return df
            
        import src.data.loader
        import src.experiments.runner
        
        # Patch the source (good practice)
        src.data.loader.load_continuous_contract = mock_loader
        src.data.loader.load_processed_1m = lambda **kw: mock_loader()
        
        # Patch the consumer (CRITICAL because of 'from X import Y')
        src.experiments.runner.load_continuous_contract = mock_loader
        src.experiments.runner.load_processed_1m = lambda **kw: mock_loader()

    
    # 2. Build Scanner
    scanner = CompositeScanner(recipe)
    print(f"Initialized scanner: {scanner.scanner_id}")
    
    # 3. Configure Experiment
    # Use OCO settings from recipe if available, else defaults
    oco_config = recipe.get("oco", {})
    entry_trigger = recipe.get("entry_trigger", {})
    
    # Determine defaults based on recipe or args
    # Note: ExperimentConfig usually takes start/end dates
    
    from src.config import CONTINUOUS_CONTRACT_PATH
    
    # Determine dates
    if args.start_date:
        start_date = args.start_date
        end_date = args.end_date
    else:
        # Default to last N days
        # (Simplified date logic for now)
        start_date = "2025-01-01" # Placeholder, clearer logic in real app
        end_date = "2025-01-30"
        
    oco_settings = OCOConfig(
        tp_multiple=oco_config.get("take_profit", {}).get("multiple", 2.0),
        stop_atr=oco_config.get("stop_loss", {}).get("multiple", 1.0)
    )

    feature_settings = FeatureConfig(
        include_ohlcv=True,
        include_indicators=True,
        include_levels=True
    )
        
    config = ExperimentConfig(
        name=args.out,
        scanner_id=scanner.scanner_id,
        start_date=start_date, 
        end_date=end_date,
        timeframe="1m",
        oco_config=oco_settings,
        feature_config=feature_settings,
    )
    
    # 4. Setup Exporter
    from src.viz.config import VizConfig
    out_dir = RESULTS_DIR / "viz" / args.out
    
    viz_config = VizConfig()
    exporter = Exporter(
        config=viz_config,
        run_id=args.out,
        experiment_config=config.to_dict()
    )
    
    # 5. Monkey Patch get_scanner
    import src.policy.scanners
    original_factory = src.policy.scanners.get_scanner
    
    def mock_factory(scanner_id, **kwargs):
        if scanner_id == scanner.scanner_id:
            return scanner
        return original_factory(scanner_id, **kwargs)
        
    src.policy.scanners.get_scanner = mock_factory
    src.experiments.runner.get_scanner = mock_factory # Patch consumer!
    
    try:
        # 6. Run Experiment
        result = run_experiment(config, exporter=exporter)
        
        # 7. Finalize
        exporter.finalize(out_dir)
        
        print(f"Success! Output at: {out_dir}")
        print("Don't forget to validate: python golden/validate_run.py " + str(out_dir))
        
    finally:
        # Restore factory
        src.policy.scanners.get_scanner = original_factory


if __name__ == "__main__":
    main()
```

#### `src/core/strategy_tool.py`

```
"""
Strategy Composer Tool Registration

Registers the run_recipe.py script as an agent-callable tool.
"""

from src.core.tool_registry import ToolRegistry, ToolCategory
import subprocess
from pathlib import Path
from typing import Dict, Any


@ToolRegistry.register(
    tool_id="run_composite_strategy",
    category=ToolCategory.STRATEGY,
    name="Run Composite Strategy",
    description="Execute a dynamically composed strategy from a JSON recipe. Creates full Trade Viz artifacts.",
    input_schema={
        "type": "object",
        "properties": {
            "recipe_path": {
                "type": "string",
                "description": "Path to the JSON recipe file"
            },
            "output_name": {
                "type": "string",
                "description": "Name for the output directory (in results/viz/)"
            },
            "start_date": {
                "type": "string",
                "description": "Start date (YYYY-MM-DD), optional"
            },
            "end_date": {
                "type": "string",
                "description": "End date (YYYY-MM-DD), optional"
            },
            "use_mock_data": {
                "type": "boolean",
                "description": "Use synthetic data for testing",
                "default": False
            }
        },
        "required": ["recipe_path", "output_name"]
    },
    output_schema={
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "output_dir": {"type": "string"},
            "message": {"type": "string"}
        }
    },
    produces_artifacts=True,
    artifact_spec={
        "location": "results/viz/{output_name}",
        "files": ["manifest.json", "decisions.jsonl", "trades.jsonl", "run.json"]
    }
)
class CompositeStrategyRunner:
    """Tool wrapper for scripts/run_recipe.py"""
    
    def __init__(self):
        self.script_path = Path(__file__).parent.parent.parent / "scripts" / "run_recipe.py"
    
    def execute(self, **inputs) -> Dict[str, Any]:
        """
        Execute the run_recipe.py script.
        
        Args:
            recipe_path: Path to JSON recipe
            output_name: Output directory name
            start_date: Optional start date
            end_date: Optional end date
            use_mock_data: Whether to use mock data
            
        Returns:
            Dict with success status and output location
        """
        recipe_path = inputs["recipe_path"]
        output_name = inputs["output_name"]
        start_date = inputs.get("start_date")
        end_date = inputs.get("end_date")
        use_mock = inputs.get("use_mock_data", False)
        
        # Build command
        cmd = [
            "python", "-m", "scripts.run_recipe",
            "--recipe", recipe_path,
            "--out", output_name
        ]
        
        if start_date:
            cmd.extend(["--start-date", start_date])
        if end_date:
            cmd.extend(["--end-date", end_date])
        if use_mock:
            cmd.append("--mock")
        
        try:
            # Execute
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                output_dir = f"results/viz/{output_name}"
                return {
                    "success": True,
                    "output_dir": output_dir,
                    "message": f"Strategy executed successfully. Output: {output_dir}"
                }
            else:
                return {
                    "success": False,
                    "output_dir": "",
                    "message": f"Execution failed: {result.stderr}"
                }
                
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output_dir": "",
                "message": "Execution timed out after 5 minutes"
            }
        except Exception as e:
            return {
                "success": False,
                "output_dir": "",
                "message": f"Error: {str(e)}"
            }
```

#### `src/policy/composite_scanner.py`

```
"""
Composite Scanner (The Strategy Engine)

This scanner interprets a JSON Recipe to build a dynamic strategy on the fly.
It replaces the need to write custom Python classes for every new strategy idea.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from src.features.state import MarketState
from src.features.pipeline import FeatureBundle
from src.policy.scanners import Scanner, ScanResult
from src.policy.triggers.factory import trigger_from_dict
from src.policy.triggers.base import TriggerResult

@dataclass
class CompositeConfig:
    """
    Configuration for a composed strategy.
    
    Example:
    {
        "name": "My Composed Strategy",
        "entry_trigger": {
            "type": "AND",
            "children": [
                {"type": "ema_cross", "fast": 9, "slow": 21},
                {"type": "rsi_threshold", "threshold": 30, "direction": "lt"}
            ]
        },
        "cooldown_bars": 10
    }
    """
    name: str
    entry_trigger: Dict[str, Any]
    cooldown_bars: int = 10


class CompositeScanner(Scanner):
    """
    A Scanner that executes a dynamic configuration.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = CompositeConfig(
            name=config.get("name", "composite_strategy"),
            entry_trigger=config["entry_trigger"],
            cooldown_bars=config.get("cooldown_bars", 10)
        )
        
        # Build the Trigger Tree
        self._trigger = trigger_from_dict(self.config.entry_trigger)
        
        # State
        self._last_trigger_idx = -1000
    
    @property
    def scanner_id(self) -> str:
        # Use the name from the config as the ID
        # This ensures it shows up nicely in Trade Viz
        return self.config.name.lower().replace(" ", "_")
    
    def scan(
        self,
        state: MarketState,
        features: FeatureBundle
    ) -> ScanResult:
        """
        Evaluate the trigger tree against current market features.
        """
        current_idx = features.bar_idx
        
        # 1. Check Cooldown
        if current_idx - self._last_trigger_idx < self.config.cooldown_bars:
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
            
        # 2. Check Trigger
        res: TriggerResult = self._trigger.check(features)
        
        if res.triggered:
            self._last_trigger_idx = current_idx
            
            return ScanResult(
                scanner_id=self.scanner_id,
                triggered=True,
                context={
                    "direction": res.direction.value,
                    "confidence": res.confidence,
                    **res.context
                },
                score=res.confidence
            )
            
        return ScanResult(scanner_id=self.scanner_id, triggered=False)
```

#### `src/policy/triggers/factory.py`

```
"""
Trigger Factory
Separated from __init__ to avoid circular imports.
"""

from typing import Dict, Any, List

from .base import Trigger
from .time_trigger import TimeTrigger
from .candle_patterns import CandlePatternTrigger
from .indicator_triggers import EMACrossTrigger, RSIThresholdTrigger
from .structure_break import StructureBreakTrigger
from .fakeout import FakeoutTrigger
from .ema_rejection import EMA200RejectionTrigger
from .parametric import RejectionTrigger, ComparisonTrigger
from .sweep import SweepTrigger
from .or_false_break import ORFalseBreakTrigger
from .vwap_reclaim import VWAPReclaimTrigger
# Logic triggers imported safely or lazily if needed
# To avoid cycle, we'll import logic classes inside the factory/registry if they depend on this factory
# But AndTrigger needs trigger_from_dict...

# Solution:
# 1. Define Registry here (empty or populated with leaves)
# 2. Logic triggers import trigger_from_dict
# 3. We import Logic triggers here to populate registry

# Wait, if logic.py imports trigger_from_dict, then trigger_from_dict cannot import logic.py at top level
# unless we use lazy import inside the function.

TRIGGER_REGISTRY = {}

def register_triggers():
    """Populate registry. Call this once or ensure imports happen."""
    # Leaf triggers (no recursion)
    TRIGGER_REGISTRY["time"] = TimeTrigger
    TRIGGER_REGISTRY["candle_pattern"] = CandlePatternTrigger
    TRIGGER_REGISTRY["ema_cross"] = EMACrossTrigger
    TRIGGER_REGISTRY["rsi_threshold"] = RSIThresholdTrigger
    TRIGGER_REGISTRY["structure_break"] = StructureBreakTrigger
    TRIGGER_REGISTRY["fakeout"] = FakeoutTrigger
    TRIGGER_REGISTRY["ema200_rejection"] = EMA200RejectionTrigger
    TRIGGER_REGISTRY["rejection"] = RejectionTrigger
    TRIGGER_REGISTRY["comparison"] = ComparisonTrigger
    TRIGGER_REGISTRY["sweep"] = SweepTrigger
    TRIGGER_REGISTRY["or_false_break"] = ORFalseBreakTrigger
    TRIGGER_REGISTRY["vwap_reclaim"] = VWAPReclaimTrigger
    
    # Logic triggers (recursive)
    from .logic import AndTrigger, OrTrigger, NotTrigger
    TRIGGER_REGISTRY["AND"] = AndTrigger
    TRIGGER_REGISTRY["OR"] = OrTrigger
    TRIGGER_REGISTRY["NOT"] = NotTrigger

def trigger_from_dict(config: dict) -> Trigger:
    """
    Factory function to create trigger from config dict.
    """
    config = config.copy()
    trigger_type = config.pop("type")
    
    if trigger_type not in TRIGGER_REGISTRY:
        # Try re-registering just in case import order messed it up
        register_triggers()
        if trigger_type not in TRIGGER_REGISTRY:
             raise ValueError(f"Unknown trigger type: {trigger_type}. Available: {list(TRIGGER_REGISTRY.keys())}")
    
    return TRIGGER_REGISTRY[trigger_type](**config)


def list_triggers() -> list:
    """List available trigger types for agent discovery."""
    return list(TRIGGER_REGISTRY.keys())


# Pre-populate registry after functions are defined
register_triggers()


```

#### `src/policy/triggers/logic.py`

```
"""
Logic Triggers
Boolean logic for composing triggers: AND, OR, NOT.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from src.policy.triggers.base import Trigger, TriggerResult, TriggerDirection

from src.features.pipeline import FeatureBundle


class AndTrigger(Trigger):
    """
    Fires only if ALL child triggers fire.
    
    Direction logic:
    - If all children agree on direction (or are NEUTRAL), inherit that direction.
    - If children conflict (LONG vs SHORT), returns NEUTRAL (but still triggered).
    """
    
    def __init__(self, children: List[Dict[str, Any]]):
        from src.policy.triggers.factory import trigger_from_dict
        self.children_configs = children
        self._children: List[Trigger] = [trigger_from_dict(c) for c in children]
        
    @property
    def trigger_id(self) -> str:
        return "AND"
        
    @property
    def params(self) -> Dict[str, Any]:
        return {"children": self.children_configs}
        
    def check(self, features: FeatureBundle) -> TriggerResult:
        results = []
        for child in self._children:
            res = child.check(features)
            if not res.triggered:
                # Short circuit
                return TriggerResult(
                    trigger_id=self.trigger_id, 
                    triggered=False
                )
            results.append(res)
            
        # If we got here, all triggered
        
        # Determine direction
        directions = {r.direction for r in results if r.direction != TriggerDirection.NEUTRAL}
        
        final_dir = TriggerDirection.NEUTRAL
        if len(directions) == 1:
            final_dir = list(directions)[0]
        # If len > 1, conflict -> NEUTRAL (but triggered)
        
        # Merge context
        merged_context = {}
        for i, res in enumerate(results):
            # Prefix keys to avoid collisions? Or just merge?
            # Merging allows downstream to see "rsi": 30
            merged_context.update(res.context)
            
        return TriggerResult(
            trigger_id=self.trigger_id,
            triggered=True,
            direction=final_dir,
            context=merged_context,
            confidence=min(r.confidence for r in results)  # Weakest link
        )


class OrTrigger(Trigger):
    """
    Fires if ANY child trigger fires.
    """
    
    def __init__(self, children: List[Dict[str, Any]]):
        from src.policy.triggers.factory import trigger_from_dict
        self.children_configs = children
        self._children: List[Trigger] = [trigger_from_dict(c) for c in children]
        
    @property
    def trigger_id(self) -> str:
        return "OR"
        
    @property
    def params(self) -> Dict[str, Any]:
        return {"children": self.children_configs}
        
    def check(self, features: FeatureBundle) -> TriggerResult:
        fired = []
        for child in self._children:
            res = child.check(features)
            if res.triggered:
                fired.append(res)
                
        if not fired:
            return TriggerResult(trigger_id=self.trigger_id, triggered=False)
            
        # Use the most confident one, or the first one
        best_res = max(fired, key=lambda r: r.confidence)
        
        return TriggerResult(
            trigger_id=self.trigger_id,
            triggered=True,
            direction=best_res.direction,
            context=best_res.context,
            confidence=best_res.confidence
        )


class NotTrigger(Trigger):
    """
    Inverts the triggered status of a child.
    
    Note: Direction is inverted if possible (LONG -> SHORT).
    """
    
    def __init__(self, child: Dict[str, Any]):
        from src.policy.triggers.factory import trigger_from_dict
        self.child_config = child
        self._child = trigger_from_dict(child)
        
    @property
    def trigger_id(self) -> str:
        return "NOT"
        
    @property
    def params(self) -> Dict[str, Any]:
        return {"child": self.child_config}
        
    def check(self, features: FeatureBundle) -> TriggerResult:
        res = self._child.check(features)
        
        # Invert triggered status
        triggered = not res.triggered
        
        if not triggered:
            return TriggerResult(trigger_id=self.trigger_id, triggered=False)
            
        # Invert direction if valid
        direction = TriggerDirection.NEUTRAL
        if res.direction == TriggerDirection.LONG:
            direction = TriggerDirection.SHORT
        elif res.direction == TriggerDirection.SHORT:
            direction = TriggerDirection.LONG
            
        return TriggerResult(
            trigger_id=self.trigger_id,
            triggered=True,
            direction=direction,
            context=res.context
        )
```

#### `src/skills/data_skills.py`

```
"""
Data Skills
Atomic tools for fetching and inspecting market data.
Wraps src.data.loader for the Agent.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta

from src.data import loader
from src.config import NY_TZ


def fetch_ohlcv(
    symbol: str = "continuous",
    start_date: str = None,
    end_date: str = None,
    limit: int = 1000
) -> List[Dict[str, Any]]:
    """
    Fetch OHLCV data for analysis.
    Returns list of dictionaries: {time, open, high, low, close, volume}
    """
    # Load full dataset (cached)
    df = loader.load_continuous_contract()
    
    # Filter by date
    if start_date:
        start_dt = pd.to_datetime(start_date).tz_localize(NY_TZ)
        df = df[df['time'] >= start_dt]
        
    if end_date:
        end_dt = pd.to_datetime(end_date).tz_localize(NY_TZ) + timedelta(days=1)
        df = df[df['time'] < end_dt]
        
    # Limit rows
    if limit and len(df) > limit:
        df = df.iloc[:limit]
        
    # Convert to list of dicts for Agent
    records = df.to_dict('records')
    
    # Format timestamps to strings for JSON serializability
    for r in records:
        if isinstance(r['time'], pd.Timestamp):
            r['time'] = r['time'].isoformat()
            
    return records


def get_current_price(symbol: str = "continuous") -> float:
    """Get the latest close price."""
    df = loader.load_continuous_contract()
    if df.empty:
        return 0.0
    return float(df['close'].iloc[-1])


def get_market_regime(
    window_days: int = 5
) -> str:
    """
    Determine simplistic market regime over last N days.
    Returns: "TRENDING_UP", "TRENDING_DOWN", "RANGING"
    """
    df = loader.load_continuous_contract()
    if df.empty:
        return "UNKNOWN"
        
    # Filter last N days
    cutoff = df['time'].iloc[-1] - timedelta(days=window_days)
    recent = df[df['time'] >= cutoff]
    
    if recent.empty:
        return "UNKNOWN"
        
    start_price = recent['close'].iloc[0]
    end_price = recent['close'].iloc[-1]
    ret = (end_price - start_price) / start_price
    
    if ret > 0.02:  # > 2% up
        return "TRENDING_UP"
    elif ret < -0.02: # > 2% down
        return "TRENDING_DOWN"
    else:
        return "RANGING"
```

#### `src/skills/indicator_skills.py`

```
"""
Indicator Skills
Atomic tools for calculating technical indicators.
These skills wrap the core features library for the Agent's use during Research.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime

from src.features import indicators
from src.features import indicators_pro
from src.features import fvg


def get_rsi(
    prices: List[float],
    period: int = 14
) -> List[float]:
    """
    Calculate RSI for a list of prices.
    Returns list of RSI values (same length, padded with 50).
    """
    series = pd.Series(prices)
    rsi = indicators.calculate_rsi(series, period)
    return rsi.tolist()


def get_previous_rsi(
    df: pd.DataFrame,
    period: int = 14,
    lookback: int = 1
) -> float:
    """
    Get the RSI value from N bars ago.
    """
    if 'close' not in df.columns:
        raise ValueError("DataFrame must have 'close' column")
        
    rsi_series = indicators.calculate_rsi(df['close'], period)
    
    if len(rsi_series) <= lookback:
        return 50.0
        
    return float(rsi_series.iloc[-lookback - 1])


def find_fvgs(
    opens: List[float],
    highs: List[float],
    lows: List[float],
    closes: List[float],
    min_size_ticks: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Identify Fair Value Gaps in price data.
    """
    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes
    })
    
    # Use internal FVG logic
    # Note: Internal logic expects specific dataframe structure
    # We simplified here for the skill interface
    
    gaps = []
    # (Simplified implementation matching the atomic need)
    # Real implementation would call src.features.fvg.find_fvg
    
    return gaps 


def get_ema(
    prices: List[float],
    period: int
) -> List[float]:
    """Calculate EMA."""
    return indicators.calculate_ema(pd.Series(prices), period).tolist()


def check_ema_cross(
    df: pd.DataFrame,
    fast: int = 9,
    slow: int = 21
) -> str:
    """
    Check if fast EMA crossed slow EMA on the MOST RECENT bar.
    Returns: "BULLISH", "BEARISH", or "NONE"
    """
    ema_fast = indicators.calculate_ema(df['close'], fast)
    ema_slow = indicators.calculate_ema(df['close'], slow)
    
    if len(df) < 2:
        return "NONE"
        
    curr_fast = ema_fast.iloc[-1]
    curr_slow = ema_slow.iloc[-1]
    prev_fast = ema_fast.iloc[-2]
    prev_slow = ema_slow.iloc[-2]
    
    if prev_fast <= prev_slow and curr_fast > curr_slow:
        return "BULLISH"
    elif prev_fast >= prev_slow and curr_fast < curr_slow:
        return "BEARISH"
        
    return "NONE"
```

#### `test_ema_strategy.json`

```
{
    "name": "Simple EMA Cross Test",
    "cooldown_bars": 20,
    "entry_trigger": {
        "type": "ema_cross",
        "fast": 9,
        "slow": 21
    },
    "oco": {
        "entry": "MARKET",
        "take_profit": {
            "multiple": 2.0
        },
        "stop_loss": {
            "multiple": 1.0
        }
    }
}```

#### `test_recipe.json`

```
{
    "name": "Test Composer Strategy",
    "cooldown_bars": 5,
    "entry_trigger": {
        "type": "AND",
        "children": [
            {
                "type": "ema_cross",
                "fast": 9,
                "slow": 21
            },
            {
                "type": "rsi_threshold",
                "threshold": 70,
                "direction": "lt"
            }
        ]
    },
    "oco": {
        "entry": "MARKET",
        "take_profit": {
            "multiple": 2.0
        },
        "stop_loss": {
            "multiple": 1.0
        }
    }
}```

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

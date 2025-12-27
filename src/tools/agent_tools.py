"""
Agent Tools for MLang2

Registered tools that agents can use for strategy creation, navigation, and analysis.
These replace the hardcoded AGENT_TOOLS and LAB_TOOLS definitions.
"""

from typing import Dict, Any, List
from src.core.tool_registry import ToolRegistry, ToolCategory


# =============================================================================
# Strategy Execution Tools
# =============================================================================

@ToolRegistry.register(
    tool_id="run_strategy",
    category=ToolCategory.STRATEGY,
    name="Run Strategy Scan",
    description="Run a modular strategy scan on historical data. Creates a new run that appears in the run list for visualization.",
    input_schema={
        "type": "object",
        "properties": {
            "strategy": {
                "type": "string",
                "enum": ["modular", "opening_range"],
                "description": "Strategy type. Use 'modular' for custom trigger/bracket configs."
            },
            "start_date": {
                "type": "string",
                "description": "Start date in YYYY-MM-DD format. Data available: 2025-03-18 to 2025-09-17."
            },
            "weeks": {
                "type": "integer",
                "description": "Number of weeks to scan.",
                "minimum": 1,
                "maximum": 26
            },
            "run_name": {
                "type": "string",
                "description": "Optional custom name for the run."
            },
            "trigger_type": {
                "type": "string",
                "enum": ["ema_cross", "ema_bounce", "rsi_threshold", "ifvg", "orb", "candle_pattern", "time"],
                "description": "Type of entry trigger."
            },
            "trigger_params": {
                "type": "object",
                "description": "Parameters for the trigger (e.g., {fast: 9, slow: 21} for ema_cross)."
            },
            "bracket_type": {
                "type": "string",
                "enum": ["atr", "percent", "fixed"],
                "description": "Type of stop/take-profit bracket."
            },
            "stop_atr": {
                "type": "number",
                "description": "Stop loss in ATR multiples (for atr bracket).",
                "default": 2.0
            },
            "tp_atr": {
                "type": "number",
                "description": "Take profit in ATR multiples (for atr bracket).",
                "default": 3.0
            }
        },
        "required": ["strategy", "start_date", "weeks", "trigger_type", "bracket_type"]
    },
    produces_artifacts=True,
    artifact_spec={
        "outputs": ["manifest.json", "decisions.jsonl", "trades.jsonl"],
        "format": "run_artifact_v1"
    }
)
class RunStrategyTool:
    """Tool for running strategy scans."""
    
    def execute(self, **inputs) -> Dict[str, Any]:
        """Execute strategy scan - handled by server/UI."""
        return {
            "status": "queued",
            "message": "Strategy run initiated",
            "inputs": inputs
        }


@ToolRegistry.register(
    tool_id="run_modular_strategy",
    category=ToolCategory.STRATEGY,
    name="Run Modular Strategy",
    description="Run a modular strategy scan on historical data with custom trigger and bracket configuration.",
    input_schema={
        "type": "object",
        "properties": {
            "trigger_type": {
                "type": "string",
                "enum": ["ema_cross", "ema_bounce", "rsi_threshold", "ifvg", "orb", "candle_pattern", "time"],
                "description": "Type of entry trigger"
            },
            "trigger_params": {
                "type": "object",
                "description": "Parameters for the trigger (e.g., {fast: 9, slow: 21} for ema_cross)"
            },
            "bracket_type": {
                "type": "string",
                "enum": ["atr", "percent", "fixed"],
                "description": "Type of stop/take-profit bracket"
            },
            "stop_atr": {
                "type": "number",
                "description": "Stop loss in ATR multiples",
                "default": 2.0
            },
            "tp_atr": {
                "type": "number",
                "description": "Take profit in ATR multiples",
                "default": 3.0
            },
            "start_date": {
                "type": "string",
                "description": "Start date YYYY-MM-DD (data: 2025-03-18 to 2025-09-17)"
            },
            "weeks": {
                "type": "integer",
                "description": "Number of weeks to scan",
                "minimum": 1,
                "maximum": 26
            },
            "run_name": {
                "type": "string",
                "description": "Optional custom name for the run"
            },
            "silent": {
                "type": "boolean",
                "description": "If true, run silently without forcing visualization UI (default: false)",
                "default": false
            }
        },
        "required": ["trigger_type", "bracket_type", "start_date", "weeks"]
    },
    produces_artifacts=True,
    artifact_spec={
        "outputs": ["manifest.json", "decisions.jsonl", "trades.jsonl"],
        "format": "run_artifact_v1"
    }
)
class RunModularStrategyTool:
    """Tool for running modular strategy scans."""
    
    def execute(self, **inputs) -> Dict[str, Any]:
        """Execute modular strategy scan - handled by server/UI."""
        return {
            "status": "queued",
            "message": "Modular strategy run initiated",
            "inputs": inputs
        }


# =============================================================================
# Navigation Tools
# =============================================================================

@ToolRegistry.register(
    tool_id="set_index",
    category=ToolCategory.UTILITY,
    name="Set Index",
    description="Navigate to a specific decision or trade by index number.",
    input_schema={
        "type": "object",
        "properties": {
            "index": {
                "type": "integer",
                "description": "The index to navigate to."
            }
        },
        "required": ["index"]
    }
)
class SetIndexTool:
    """Tool for navigating to a specific index."""
    
    def execute(self, **inputs) -> Dict[str, Any]:
        """Navigate to index - handled by UI."""
        return {
            "status": "success",
            "index": inputs.get("index", 0)
        }


@ToolRegistry.register(
    tool_id="set_mode",
    category=ToolCategory.UTILITY,
    name="Set View Mode",
    description="Switch between viewing decisions or trades.",
    input_schema={
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["DECISION", "TRADE"],
                "description": "The view mode to switch to."
            }
        },
        "required": ["mode"]
    }
)
class SetModeTool:
    """Tool for switching view modes."""
    
    def execute(self, **inputs) -> Dict[str, Any]:
        """Switch view mode - handled by UI."""
        return {
            "status": "success",
            "mode": inputs.get("mode", "DECISION")
        }


# =============================================================================
# Data Access Tools
# =============================================================================

@ToolRegistry.register(
    tool_id="load_run",
    category=ToolCategory.UTILITY,
    name="Load Run",
    description="Load an existing run for visualization.",
    input_schema={
        "type": "object",
        "properties": {
            "run_id": {
                "type": "string",
                "description": "The run ID to load."
            }
        },
        "required": ["run_id"]
    }
)
class LoadRunTool:
    """Tool for loading existing runs."""
    
    def execute(self, **inputs) -> Dict[str, Any]:
        """Load run - handled by UI."""
        return {
            "status": "success",
            "run_id": inputs.get("run_id")
        }


@ToolRegistry.register(
    tool_id="list_runs",
    category=ToolCategory.UTILITY,
    name="List Runs",
    description="List all available runs that can be loaded.",
    input_schema={
        "type": "object",
        "properties": {}
    }
)
class ListRunsTool:
    """Tool for listing available runs."""
    
    def execute(self, **inputs) -> Dict[str, Any]:
        """List runs - handled by server."""
        return {
            "status": "success",
            "runs": []  # Populated by server
        }


# =============================================================================
# Lab-Specific Tools
# =============================================================================

@ToolRegistry.register(
    tool_id="start_live_mode",
    category=ToolCategory.UTILITY,
    name="Start Live Mode",
    description="Start live trading simulation with real-time YFinance data.",
    input_schema={
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "enum": ["MES=F", "ES=F", "NQ=F", "SPY"],
                "description": "Ticker symbol"
            },
            "strategy": {
                "type": "string",
                "enum": ["ema_cross", "ifvg", "orb"],
                "description": "Strategy to use"
            }
        },
        "required": ["ticker", "strategy"]
    }
)
class StartLiveModeTool:
    """Tool for starting live trading simulation."""
    
    def execute(self, **inputs) -> Dict[str, Any]:
        """Start live mode - handled by server."""
        return {
            "status": "started",
            "ticker": inputs.get("ticker"),
            "strategy": inputs.get("strategy")
        }


@ToolRegistry.register(
    tool_id="query_experiments",
    category=ToolCategory.UTILITY,
    name="Query Experiments",
    description="Query the experiment database for past strategy results.",
    input_schema={
        "type": "object",
        "properties": {
            "sort_by": {
                "type": "string",
                "enum": ["win_rate", "total_pnl", "total_trades"],
                "description": "Metric to sort by"
            },
            "min_trades": {
                "type": "integer",
                "description": "Minimum number of trades required to include experiment in results",
                "default": 1
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return",
                "default": 5
            }
        },
        "required": ["sort_by"]
    }
)
class QueryExperimentsTool:
    """Tool for querying experiment history."""
    
    def execute(self, **inputs) -> Dict[str, Any]:
        """Query experiments - handled by server."""
        return {
            "status": "success",
            "sort_by": inputs.get("sort_by"),
            "results": []  # Populated by server
        }
@ToolRegistry.register(
    tool_id="compare_runs",
    category=ToolCategory.UTILITY,
    name="Compare Runs",
    description="Compare results of two or more runs side-by-side",
    input_schema={
        "type": "object",
        "properties": {
            "run_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of run IDs to compare"
            }
        },
        "required": ["run_ids"]
    }
)
class CompareRunsTool:
    def execute(self, **inputs) -> Dict[str, Any]:
        """Compare runs - handled by server."""
        return {"status": "success", "run_ids": inputs.get("run_ids")}


@ToolRegistry.register(
    tool_id="get_run_config",
    category=ToolCategory.UTILITY,
    name="Get Run Config",
    description="Get the configuration/recipe used for a specific run",
    input_schema={
        "type": "object",
        "properties": {
            "run_id": {
                "type": "string",
                "description": "The run ID to inspect"
            }
        },
        "required": ["run_id"]
    }
)
class GetRunConfigTool:
    def execute(self, **inputs) -> Dict[str, Any]:
        """Get run config - handled by server."""
        return {"status": "success", "run_id": inputs.get("run_id")}


@ToolRegistry.register(
    tool_id="create_variation",
    category=ToolCategory.STRATEGY,
    name="Create Variation",
    description="Create a new run by modifying an existing run's configuration",
    input_schema={
        "type": "object",
        "properties": {
            "base_run_id": {
                "type": "string",
                "description": "The run ID to use as a base"
            },
            "modifications": {
                "type": "object",
                "description": "Changes to apply to the base config (e.g., {'tp_atr': 4.0})"
            },
            "run_name": {
                "type": "string",
                "description": "Optional custom name for the new run"
            }
        },
        "required": ["base_run_id", "modifications"]
    }
)
class CreateVariationTool:
    def execute(self, **inputs) -> Dict[str, Any]:
        """Create variation - handled by server."""
        return {"status": "queued", "base_run_id": inputs.get("base_run_id")}


@ToolRegistry.register(
    tool_id="save_to_tradeviz",
    category=ToolCategory.UTILITY,
    name="Save to Trade Viz",
    description="Move a successful experiment from the Lab to the main Trade Viz view",
    input_schema={
        "type": "object",
        "properties": {
            "run_id": {
                "type": "string",
                "description": "The run ID to save"
            }
        },
        "required": ["run_id"]
    }
)
class SaveToTradeVizTool:
    def execute(self, **inputs) -> Dict[str, Any]:
        """Save to Trade Viz - handled by server."""
        return {"status": "success", "run_id": inputs.get("run_id")}


@ToolRegistry.register(
    tool_id="delete_run",
    category=ToolCategory.UTILITY,
    name="Delete Run",
    description="Delete a run and its associated data files",
    input_schema={
        "type": "object",
        "properties": {
            "run_id": {
                "type": "string",
                "description": "The run ID to delete"
            }
        },
        "required": ["run_id"]
    }
)
class DeleteRunTool:
    def execute(self, **inputs) -> Dict[str, Any]:
        """Delete run - handled by server."""
        return {"status": "success", "run_id": inputs.get("run_id")}
@ToolRegistry.register(
    tool_id="train_model",
    category=ToolCategory.STRATEGY,
    name="Train ML Model",
    description="Train a machine learning model (XGBoost, CNN, etc.) on historical data",
    input_schema={
        "type": "object",
        "properties": {
            "model_type": {
                "type": "string",
                "enum": ["xgboost", "cnn", "lstm"],
                "description": "Type of model to train"
            },
            "target": {
                "type": "string",
                "description": "Training target (e.g., 'next_bar_direction', 'atr_cross')"
            },
            "start_date": {
                "type": "string",
                "description": "Training start date (YYYY-MM-DD)"
            },
            "end_date": {
                "type": "string",
                "description": "Training end date (YYYY-MM-DD)"
            },
            "params": {
                "type": "object",
                "description": "Training hyperparameters"
            }
        },
        "required": ["model_type", "target", "start_date", "end_date"]
    }
)
class TrainModelTool:
    def execute(self, **inputs) -> Dict[str, Any]:
        """Train model - handled by server."""
        return {"status": "queued", "model_type": inputs.get("model_type")}

        model_id: str,
        name: str,
        description: str = "",
        input_schema: Dict[str, Any] = None,
        output_schema: Dict[str, Any] = None
    ):
        """Decorator to register a model."""
        def decorator(model_class):
            cls._registry[model_id] = model_class
            cls._info[model_id] = ModelInfo(
                model_id=model_id,
                name=name,
                description=description,
                input_schema=input_schema or {},
                output_schema=output_schema or {},
            )
            return model_class
        return decorator
    
    @classmethod
    def create(cls, model_id: str, **params) -> PolicyModel:
        """Create model instance by ID."""
        if model_id not in cls._registry:
            raise ValueError(f"Unknown model: {model_id}")
        return cls._registry[model_id](**params)
    
    @classmethod
    def list_all(cls) -> List[ModelInfo]:
        """List all registered models."""
        return list(cls._info.values())
    
    @classmethod
    def get_info(cls, model_id: str) -> ModelInfo:
        """Get info for a specific model."""
        if model_id not in cls._info:
            raise ValueError(f"Unknown model: {model_id}")
        return cls._info[model_id]


# =============================================================================
# Indicator Registry
# =============================================================================

@dataclass
class IndicatorSeries:
    """
    First-class indicator series for visualization.
    Not hardcoded in chart - generic overlay rendering.
    """
    indicator_id: str
    name: str
    type: str  # 'line', 'histogram', 'band', 'marker'
    points: List[Dict[str, Any]]  # [{time, value}, ...] or specific format per type
    style: Dict[str, Any] = None  # Color, line width, etc.
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'indicator_id': self.indicator_id,
            'name': self.name,
            'type': self.type,
            'points': self.points,
            'style': self.style or {},
        }


@dataclass
class IndicatorInfo:
    """Metadata about a registered indicator."""
    indicator_id: str
    name: str
    description: str
    output_type: str  # 'line', 'histogram', 'band', 'marker'
    params_schema: Dict[str, Any]


class IndicatorRegistry:
    """
    Registry for indicator implementations.
    
    Usage:
        @IndicatorRegistry.register("ema", "EMA", output_type="line")
        class EMAIndicator:
            def __init__(self, period=20):
                self.period = period
            
            def compute(self, stepper) -> IndicatorSeries:
                # Calculate EMA
                return IndicatorSeries(...)
    """
    
    _registry: Dict[str, Callable] = {}
    _info: Dict[str, IndicatorInfo] = {}
    
    @classmethod
    def register(
        cls,
        indicator_id: str,
        name: str,
        output_type: str = "line",
        description: str = "",
        params_schema: Dict[str, Any] = None
    ):
        """Decorator to register an indicator."""
        def decorator(indicator_class):
            cls._registry[indicator_id] = indicator_class
            cls._info[indicator_id] = IndicatorInfo(
                indicator_id=indicator_id,
                name=name,
                description=description,
                output_type=output_type,
                params_schema=params_schema or {},
            )
            return indicator_class
        return decorator
    
    @classmethod
    def create(cls, indicator_id: str, **params):
        """Create indicator instance by ID."""
        if indicator_id not in cls._registry:
            raise ValueError(f"Unknown indicator: {indicator_id}")
        return cls._registry[indicator_id](**params)
    
    @classmethod
    def list_all(cls) -> List[IndicatorInfo]:
        """List all registered indicators."""
        return list(cls._info.values())
    
    @classmethod
    def get_info(cls, indicator_id: str) -> IndicatorInfo:
        """Get info for a specific indicator."""
        if indicator_id not in cls._info:
            raise ValueError(f"Unknown indicator: {indicator_id}")
        return cls._info[indicator_id]
    
    @classmethod
    def compute_all(cls, stepper: Any, indicator_ids: List[str]) -> List[IndicatorSeries]:
        """Compute multiple indicators at once."""
        results = []
        for indicator_id in indicator_ids:
            indicator = cls.create(indicator_id)
            series = indicator.compute(stepper)
            results.append(series)
        return results

```

### src/core/strategy_tool.py

```python
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
            },
            "light_mode": {
                "type": "boolean",
                "description": "Run in light mode (no heavy visualization files)",
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
            light_mode: Whether to run in light mode
            
        Returns:
            Dict with success status and output location
        """
        recipe_path = inputs["recipe_path"]
        output_name = inputs["output_name"]
        start_date = inputs.get("start_date")
        end_date = inputs.get("end_date")
        use_mock = inputs.get("use_mock_data", False)
        light_mode = inputs.get("light_mode", False)
        
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
        if light_mode:
            cmd.append("--light")
        
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

### src/core/tool_registry.py

```python
"""
Unified Tool Registry

A single, consistent registry system for all executable components in MLang2:
scanners, models, indicators, skills, and agent tools.

This replaces the fragmented system of:
- ScannerRegistry, ModelRegistry, IndicatorRegistry (src/core/registries.py)
- SkillRegistry (src/skills/registry.py)
- Hardcoded AGENT_TOOLS (src/server/main.py)

Usage:
    from src/core.tool_registry import ToolRegistry, ToolCategory
    
    @ToolRegistry.register(
        tool_id="ema_cross_scanner",
        category=ToolCategory.SCANNER,
        name="EMA Cross Scanner",
        description="Detects EMA crossovers",
        input_schema={...},
        output_schema={...}
    )
    class EMACrossScanner:
        def __init__(self, fast=12, slow=26):
            ...
        
        def execute(self, **inputs):
            # Tool execution
            ...
"""

from typing import Dict, Any, List, Callable, Optional, Protocol
from dataclasses import dataclass, field
from enum import Enum
import inspect
import json


class ToolCategory(Enum):
    """Categories for tools - replaces separate registry systems."""
    SCANNER = "scanner"        # Pattern/signal detection (was ScannerRegistry)
    MODEL = "model"            # ML model inference (was ModelRegistry)
    INDICATOR = "indicator"    # Technical indicators (was IndicatorRegistry)
    SKILL = "skill"            # Agent capabilities (was SkillRegistry)
    STRATEGY = "strategy"      # Strategy executors
    EXPORTER = "exporter"      # Viz/data exporters
    UTILITY = "utility"        # Helper tools


class ToolProtocol(Protocol):
    """Protocol that all tools must implement."""
    
    def execute(self, **inputs) -> Dict[str, Any]:
        """Execute the tool with given inputs and return outputs."""
        ...


@dataclass
class ToolInfo:
    """
    Metadata about a registered tool.
    
    This is the unified info structure that replaces:
    - ScannerInfo
    - ModelInfo
    - IndicatorInfo
    - SkillRegistry's dict structure
    """
    tool_id: str
    category: ToolCategory
    name: str
    description: str
    
    # JSON schemas for validation and agent tool generation
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    
    # Optional metadata
    version: str = "1.0"
    author: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    # For deterministic artifact outputs
    produces_artifacts: bool = False
    artifact_spec: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'tool_id': self.tool_id,
            'category': self.category.value,
            'name': self.name,
            'description': self.description,
            'input_schema': self.input_schema,
            'output_schema': self.output_schema,
            'version': self.version,
            'author': self.author,
            'tags': self.tags,
            'produces_artifacts': self.produces_artifacts,
            'artifact_spec': self.artifact_spec,
        }
    
    def to_gemini_function_declaration(self) -> Dict[str, Any]:
        """
        Convert to Gemini function calling format.
        Replaces hardcoded AGENT_TOOLS definitions.
        """
        return {
            'name': self.tool_id,
            'description': self.description,
            'parameters': self.input_schema if self.input_schema else {
                'type': 'object',
                'properties': {}
            }
        }


class ToolRegistry:
    """
    Unified registry for all tools in the system.
    
    Replaces:
    - ScannerRegistry
    - ModelRegistry  
    - IndicatorRegistry
    - SkillRegistry
    - Hardcoded AGENT_TOOLS
    
    Benefits:
    - Single contract for all executable components
    - Auto-generate agent tool schemas
    - Consistent validation
    - Deterministic artifact outputs
    - Category labels instead of separate mechanisms
    """
    
    _registry: Dict[str, Callable] = {}
    _info: Dict[str, ToolInfo] = {}
    
    @classmethod
    def register(
        cls,
        tool_id: str,
        category: ToolCategory,
        name: str,
        description: str = "",
        input_schema: Dict[str, Any] = None,
        output_schema: Dict[str, Any] = None,
        version: str = "1.0",
        author: str = None,
        tags: List[str] = None,
        produces_artifacts: bool = False,
        artifact_spec: Dict[str, Any] = None,
    ):
        """
        Decorator to register a tool.
        
        Args:
            tool_id: Unique identifier (e.g., "ema_cross_scanner")
            category: ToolCategory enum value
            name: Human-readable name
            description: What the tool does
            input_schema: JSON schema for inputs
            output_schema: JSON schema for outputs
            version: Semantic version
            author: Optional author
            tags: Optional tags for filtering
            produces_artifacts: Whether tool creates files
            artifact_spec: Specification of produced artifacts
        
        Returns:
            Decorator function
        """
        def decorator(tool_class):
            cls._registry[tool_id] = tool_class
            cls._info[tool_id] = ToolInfo(
                tool_id=tool_id,
                category=category,
                name=name,
                description=description,
                input_schema=input_schema or {},
                output_schema=output_schema or {},
                version=version,
                author=author,
                tags=tags or [],
                produces_artifacts=produces_artifacts,
                artifact_spec=artifact_spec,
            )
            return tool_class
        return decorator
    
    @classmethod
    def create(cls, tool_id: str, **params) -> ToolProtocol:
        """
        Create a tool instance by ID.
        
        Args:
            tool_id: Tool identifier
            **params: Parameters to pass to tool constructor
        
        Returns:
            Tool instance
        """
        if tool_id not in cls._registry:
            raise ValueError(f"Unknown tool: {tool_id}")
        return cls._registry[tool_id](**params)
    
    @classmethod
    def get_tool(cls, tool_id: str) -> ToolProtocol:
        """
        Get a tool instance by ID (no params, for dynamic execution).
        
        Args:
            tool_id: Tool identifier
        
        Returns:
            Tool instance or None if not found
        """
        if tool_id not in cls._registry:
            return None
        return cls._registry[tool_id]()
    
    @classmethod
    def get_info(cls, tool_id: str) -> ToolInfo:
        """Get metadata for a specific tool."""
        if tool_id not in cls._info:
            raise ValueError(f"Unknown tool: {tool_id}")
        return cls._info[tool_id]
    
    @classmethod
    def list_all(cls, category: ToolCategory = None) -> List[ToolInfo]:
        """
        List all registered tools, optionally filtered by category.
        
        Args:
            category: Optional category filter
        
        Returns:
            List of ToolInfo objects
        """
        if category is None:
            return list(cls._info.values())
        return [info for info in cls._info.values() if info.category == category]
    
    @classmethod
    def list_by_tag(cls, tag: str) -> List[ToolInfo]:
        """List tools with a specific tag."""
        return [info for info in cls._info.values() if tag in info.tags]
    
    @classmethod
    def get_gemini_function_declarations(
        cls, 
        categories: List[ToolCategory] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate Gemini function declarations from registered tools.
        
        This replaces hardcoded AGENT_TOOLS and LAB_TOOLS.
        
        Args:
            categories: Optional list of categories to include
        
        Returns:
            List of Gemini function declarations
        """
        tools_to_export = cls._info.values()
        
        if categories:
            tools_to_export = [
                info for info in tools_to_export 
                if info.category in categories
            ]
        
        return [info.to_gemini_function_declaration() for info in tools_to_export]
    
    @classmethod
    def validate_tool_output(cls, tool_id: str, output: Dict[str, Any]) -> List[str]:
        """
        Validate tool output against its schema.
        
        Args:
            tool_id: Tool identifier
            output: Output to validate
        
        Returns:
            List of validation errors (empty if valid)
        """
        info = cls.get_info(tool_id)
        
        # TODO: Implement JSON schema validation
        # For now, just check if output is a dict
        if not isinstance(output, dict):
            return [f"Tool output must be a dictionary, got {type(output)}"]
        
        return []
    
    @classmethod
    def export_catalog(cls, output_path: str = None) -> Dict[str, Any]:
        """
        Export complete tool catalog as JSON.
        
        Used for:
        - API endpoint /tools/catalog
        - Documentation generation
        - Agent discovery
        
        Args:
            output_path: Optional path to write JSON file
        
        Returns:
            Catalog dictionary
        """
        catalog = {
            'version': '1.0',
            'total_tools': len(cls._info),
            'categories': {},
            'tools': []
        }
        
        # Count by category
        for category in ToolCategory:
            tools_in_category = cls.list_all(category)
            catalog['categories'][category.value] = len(tools_in_category)
        
        # Add all tools
        for info in cls._info.values():
            catalog['tools'].append(info.to_dict())
        
        # Write to file if path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(catalog, f, indent=2)
        
        return catalog


# =============================================================================
# Backward Compatibility Adapters
# =============================================================================

class ScannerRegistryAdapter:
    """
    Adapter to make ToolRegistry work like old ScannerRegistry.
    Allows gradual migration.
    """
    
    @classmethod
    def register(cls, scanner_id: str, name: str, description: str = "", 
                 params_schema: Dict[str, Any] = None):
        """Register a scanner using the new ToolRegistry."""
        return ToolRegistry.register(
            tool_id=scanner_id,
            category=ToolCategory.SCANNER,
            name=name,
            description=description,
            input_schema=params_schema,
        )
    
    @classmethod
    def create(cls, scanner_id: str, **params):
        """Create scanner instance."""
        return ToolRegistry.create(scanner_id, **params)
    
    @classmethod
    def list_all(cls):
        """List all scanners."""
        return ToolRegistry.list_all(ToolCategory.SCANNER)
    
    @classmethod
    def get_info(cls, scanner_id: str):
        """Get scanner info."""
        return ToolRegistry.get_info(scanner_id)


class ModelRegistryAdapter:
    """Adapter for ModelRegistry."""
    
    @classmethod
    def register(cls, model_id: str, name: str, description: str = "",
                 input_schema: Dict[str, Any] = None,
                 output_schema: Dict[str, Any] = None):
        return ToolRegistry.register(
            tool_id=model_id,
            category=ToolCategory.MODEL,
            name=name,
            description=description,
            input_schema=input_schema,
            output_schema=output_schema,
        )
    
    @classmethod
    def create(cls, model_id: str, **params):
        return ToolRegistry.create(model_id, **params)
    
    @classmethod
    def list_all(cls):
        return ToolRegistry.list_all(ToolCategory.MODEL)
    
    @classmethod
    def get_info(cls, model_id: str):
        return ToolRegistry.get_info(model_id)


class IndicatorRegistryAdapter:
    """Adapter for IndicatorRegistry."""
    
    @classmethod
    def register(cls, indicator_id: str, name: str, output_type: str = "line",
                 description: str = "", params_schema: Dict[str, Any] = None):
        return ToolRegistry.register(
            tool_id=indicator_id,
            category=ToolCategory.INDICATOR,
            name=name,
            description=description,
            input_schema=params_schema,
            tags=[f"output_type:{output_type}"],
        )
    
    @classmethod
    def create(cls, indicator_id: str, **params):
        return ToolRegistry.create(indicator_id, **params)
    
    @classmethod
    def list_all(cls):
        return ToolRegistry.list_all(ToolCategory.INDICATOR)
    
    @classmethod
    def get_info(cls, indicator_id: str):
        return ToolRegistry.get_info(indicator_id)


class SkillRegistryAdapter:
    """Adapter for SkillRegistry."""
    
    def __init__(self):
        # Compatibility with old SkillRegistry instance creation
        pass
    
    def register(self, name: str, func: Callable, description: str):
        """Register a skill as a tool."""
        # Wrap the function in a simple tool class
        class SkillToolWrapper:
            def __init__(self, skill_func=func):
                self.skill_func = skill_func
            
            def execute(self, **inputs):
                return self.skill_func(**inputs)
        
        # Get function signature for input schema
        sig = inspect.signature(func)
        input_schema = {
            'type': 'object',
            'properties': {
                param: {'type': 'string'}  # Simple default
                for param in sig.parameters.keys()
            }
        }
        
        ToolRegistry.register(
            tool_id=name,
            category=ToolCategory.SKILL,
            name=name,
            description=description,
            input_schema=input_schema,
        )(SkillToolWrapper)
    
    def list_skills(self) -> List[Dict]:
        """List skills in old format."""
        skills = ToolRegistry.list_all(ToolCategory.SKILL)
        return [
            {
                'name': info.tool_id,
                'description': info.description,
                'signature': str(info.input_schema)
            }
            for info in skills
        ]
    
    def get_skill(self, name: str) -> Callable:
        """Get skill function (compatibility)."""
        tool = ToolRegistry.create(name)
        return lambda **kwargs: tool.execute(**kwargs)


# =============================================================================
# Convenience Functions
# =============================================================================

def register_scanner(scanner_id: str, name: str, **kwargs):
    """Convenience function to register a scanner."""
    return ToolRegistry.register(
        tool_id=scanner_id,
        category=ToolCategory.SCANNER,
        name=name,
        **kwargs
    )


def register_model(model_id: str, name: str, **kwargs):
    """Convenience function to register a model."""
    return ToolRegistry.register(
        tool_id=model_id,
        category=ToolCategory.MODEL,
        name=name,
        **kwargs
    )


def register_indicator(indicator_id: str, name: str, **kwargs):
    """Convenience function to register an indicator."""
    return ToolRegistry.register(
        tool_id=indicator_id,
        category=ToolCategory.INDICATOR,
        name=name,
        **kwargs
    )


def register_skill(skill_id: str, name: str, **kwargs):
    """Convenience function to register a skill."""
    return ToolRegistry.register(
        tool_id=skill_id,
        category=ToolCategory.SKILL,
        name=name,
        **kwargs
    )

```

### src/datasets/__init__.py

```python
# Datasets module
"""Record schemas, sharding, and data loading."""

```

### src/datasets/decision_record.py

```python
"""
Decision Record
Record logged at every decision point (including NO_TRADE).
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

from src.policy.actions import Action, SkipReason
from src.sim.oco_engine import OCOConfig


@dataclass
class DecisionRecord:
    """
    Complete record of a decision point.
    
    Logged at every scanner trigger, not just taken trades.
    This is the core training data structure.
    """
    
    # =========================================================================
    # Identifiers
    # =========================================================================
    timestamp: pd.Timestamp
    bar_idx: int
    decision_id: str = ""          # Unique ID for this decision
    
    # =========================================================================
    # Decision Point Context
    # =========================================================================
    scanner_id: str = ""           # Which scanner triggered
    scanner_context: Dict[str, Any] = field(default_factory=dict)
    
    # =========================================================================
    # Decision Made
    # =========================================================================
    action: Action = Action.NO_TRADE
    skip_reason: SkipReason = SkipReason.NOT_SKIPPED
    skip_reason_detail: str = ""
    
    # =========================================================================
    # Order Configuration (if PLACE_ORDER)
    # =========================================================================
    oco_config: Optional[OCOConfig] = None
    
    # =========================================================================
    # Features (CAUSAL - at decision time)
    # =========================================================================
    # Price windows for CNN
    x_price_1m: Optional[np.ndarray] = None     # (120, 5) or configured
    x_price_5m: Optional[np.ndarray] = None     # (24, 5)
    x_price_15m: Optional[np.ndarray] = None    # (8, 5)
    
    # Context vector for MLP
    x_context: Optional[np.ndarray] = None      # (20,) or configured
    
    # Current market state
    current_price: float = 0.0
    atr: float = 0.0
    
    # =========================================================================
    # Counterfactual Labels (FUTURE-AWARE)
    # =========================================================================
    # These answer: "What WOULD have happened if we traded here?"
    cf_outcome: str = ""           # WIN, LOSS, TIMEOUT
    cf_pnl: float = 0.0           # Points
    cf_pnl_dollars: float = 0.0   # With costs
    cf_mae: float = 0.0           # Max Adverse Excursion
    cf_mfe: float = 0.0           # Max Favorable Excursion
    cf_mae_atr: float = 0.0       # Normalized
    cf_mfe_atr: float = 0.0
    cf_bars_held: int = 0
    cf_entry_price: float = 0.0
    cf_exit_price: float = 0.0
    
    # Optional: outcomes for multiple OCO variants
    cf_multi_oco: Optional[Dict[str, Dict]] = None
    
    # =========================================================================
    # Methods
    # =========================================================================
    
    def is_trade(self) -> bool:
        """Was a trade actually placed?"""
        return self.action == Action.PLACE_ORDER
    
    def was_skipped(self) -> bool:
        """Was this opportunity skipped?"""
        return self.action == Action.NO_TRADE
    
    def is_good_skip(self) -> bool:
        """Skipped and would have lost."""
        return self.was_skipped() and self.cf_outcome == 'LOSS'
    
    def is_bad_skip(self) -> bool:
        """Skipped but would have won."""
        return self.was_skipped() and self.cf_outcome == 'WIN'
    
    def get_label_for_training(self) -> int:
        """Get classification label for training."""
        if self.cf_outcome == 'WIN':
            return 1
        elif self.cf_outcome == 'LOSS':
            return 0
        else:  # TIMEOUT
            return -1  # Could exclude or treat as separate class
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'bar_idx': self.bar_idx,
            'decision_id': self.decision_id,
            'scanner_id': self.scanner_id,
            'action': self.action.value,
            'skip_reason': self.skip_reason.value,
            'current_price': self.current_price,
            'atr': self.atr,
            'cf_outcome': self.cf_outcome,
            'cf_pnl': self.cf_pnl,
            'cf_pnl_dollars': self.cf_pnl_dollars,
            'cf_mae': self.cf_mae,
            'cf_mfe': self.cf_mfe,
            'cf_bars_held': self.cf_bars_held,
        }
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'DecisionRecord':
        """Create from dictionary."""
        record = DecisionRecord(
            timestamp=pd.Timestamp(d['timestamp']) if d.get('timestamp') else None,
            bar_idx=d.get('bar_idx', 0),
            decision_id=d.get('decision_id', ''),
            scanner_id=d.get('scanner_id', ''),
            action=Action(d.get('action', 'NO_TRADE')),
            skip_reason=SkipReason(d.get('skip_reason', 'NOT_SKIPPED')),
            current_price=d.get('current_price', 0.0),
            atr=d.get('atr', 0.0),
            cf_outcome=d.get('cf_outcome', ''),
            cf_pnl=d.get('cf_pnl', 0.0),
            cf_pnl_dollars=d.get('cf_pnl_dollars', 0.0),
            cf_mae=d.get('cf_mae', 0.0),
            cf_mfe=d.get('cf_mfe', 0.0),
            cf_bars_held=d.get('cf_bars_held', 0),
        )
        return record

```

### src/datasets/reader.py

```python
"""
Shard Reader
Read sharded datasets for training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Iterator, Optional
import json

import torch
from torch.utils.data import Dataset, DataLoader

from src.datasets.decision_record import DecisionRecord
from src.datasets.schema import DatasetSchema, DEFAULT_SCHEMA
from src.config import SHARDS_DIR


class ShardReader:
    """
    Read sharded DecisionRecords.
    """
    
    def __init__(self, shard_dir: Path = None):
        self.shard_dir = Path(shard_dir or SHARDS_DIR)
        
        # Load manifest
        manifest_path = self.shard_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                self.manifest = json.load(f)
        else:
            self.manifest = {}
        
        # Find shard files
        self.shard_paths = sorted(self.shard_dir.glob("shard_*.parquet"))
        self.arrays_dir = self.shard_dir / "arrays"
    
    def __len__(self) -> int:
        return self.manifest.get('total_records', 0)
    
    def __iter__(self) -> Iterator[DecisionRecord]:
        """Iterate over all records."""
        for shard_path in self.shard_paths:
            df = pd.read_parquet(shard_path)
            
            for _, row in df.iterrows():
                record = DecisionRecord.from_dict(row.to_dict())
                
                # Load arrays
                if 'x_price_1m_path' in row and pd.notna(row['x_price_1m_path']):
                    record.x_price_1m = np.load(row['x_price_1m_path'])
                
                if 'x_price_5m_path' in row and pd.notna(row['x_price_5m_path']):
                    record.x_price_5m = np.load(row['x_price_5m_path'])
                
                if 'x_price_15m_path' in row and pd.notna(row['x_price_15m_path']):
                    record.x_price_15m = np.load(row['x_price_15m_path'])
                
                if 'x_context_path' in row and pd.notna(row['x_context_path']):
                    record.x_context = np.load(row['x_context_path'])
                
                yield record
    
    def to_dataframe(self) -> pd.DataFrame:
        """Load all metadata (without arrays) as DataFrame."""
        dfs = [pd.read_parquet(p) for p in self.shard_paths]
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)


class DecisionDataset(Dataset):
    """
    PyTorch Dataset for training.
    """
    
    def __init__(
        self,
        shard_dir: Path,
        schema: DatasetSchema = None,
        include_timeout: bool = False
    ):
        self.schema = schema or DEFAULT_SCHEMA
        self.include_timeout = include_timeout
        
        # Load all records (for simplicity - could be lazy)
        reader = ShardReader(shard_dir)
        self.records = []
        
        for record in reader:
            # Filter by label
            if record.cf_outcome == 'TIMEOUT' and not include_timeout:
                continue
            if record.cf_outcome not in self.schema.y_classification:
                continue
            
            self.records.append(record)
    
    def __len__(self) -> int:
        return len(self.records)
    
    def __getitem__(self, idx: int):
        record = self.records[idx]
        
        # Price windows - (C, L) format for Conv1d
        x_price_1m = torch.FloatTensor(record.x_price_1m.T) if record.x_price_1m is not None else torch.zeros(5, 120)
        x_price_5m = torch.FloatTensor(record.x_price_5m.T) if record.x_price_5m is not None else torch.zeros(5, 24)
        x_price_15m = torch.FloatTensor(record.x_price_15m.T) if record.x_price_15m is not None else torch.zeros(5, 8)
        
        # Context vector
        x_context = torch.FloatTensor(record.x_context) if record.x_context is not None else torch.zeros(self.schema.x_context_dim)
        
        # Label
        label_idx = self.schema.get_label_idx(record.cf_outcome) if record.cf_outcome in self.schema.y_classification else 0
        y = torch.LongTensor([label_idx])
        
        # Regression targets
        y_reg = torch.FloatTensor([
            record.cf_pnl,
            record.cf_mae,
            record.cf_mfe,
            float(record.cf_bars_held)
        ])
        
        return {
            'x_price_1m': x_price_1m,
            'x_price_5m': x_price_5m,
            'x_price_15m': x_price_15m,
            'x_context': x_context,
            'y': y,
            'y_reg': y_reg,
        }


def create_dataloader(
    shard_dir: Path,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs
) -> DataLoader:
    """Create PyTorch DataLoader from shard directory."""
    dataset = DecisionDataset(shard_dir, **kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

```

### src/datasets/schema.py

```python
"""
Dataset Schema
Explicit separation of x_price (CNN) from x_context (MLP).
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DatasetSchema:
    """
    Defines the structure of training data.
    
    Separates:
    - x_price: Raw OHLCV windows for CNN
    - x_context: Derived features for MLP
    - y: Labels (classification and regression)
    """
    
    # =========================================================================
    # Price Windows (for CNN)
    # =========================================================================
    # Shape: (lookback, channels) where channels = OHLCV = 5
    x_price_2h_1m: Tuple[int, int] = (120, 5)   # 120 1m bars = 2 hours
    x_price_2h_5m: Tuple[int, int] = (24, 5)    # 24 5m bars = 2 hours
    x_price_2h_15m: Tuple[int, int] = (8, 5)    # 8 15m bars = 2 hours
    
    # =========================================================================
    # Context Vector (for MLP)
    # =========================================================================
    x_context_dim: int = 20
    
    x_context_features: List[str] = field(default_factory=lambda: [
        'dist_ema_5m_200_atr',
        'dist_ema_15m_200_atr',
        'dist_vwap_session_atr',
        'dist_vwap_weekly_atr',
        'dist_nearest_1h_level_atr',
        'dist_nearest_4h_level_atr',
        'dist_pdh_atr',
        'dist_pdl_atr',
        'adr_pct_used',
        'rsi_5m_14_norm',
        'rsi_15m_14_norm',
        'relative_volume',
        'hour_sin',
        'hour_cos',
        'dow_sin',
        'dow_cos',
        'is_rth',
        'is_first_hour',
        'is_last_hour',
        'mins_into_session_norm',
    ])
    
    # =========================================================================
    # Labels
    # =========================================================================
    # Classification: Counterfactual outcome
    # NOTE: NO_TRADE is NOT a label class - it's an action
    y_classification: List[str] = field(default_factory=lambda: [
        'WIN',
        'LOSS',
        'TIMEOUT'
    ])
    
    # Regression targets
    y_regression: List[str] = field(default_factory=lambda: [
        'cf_pnl',
        'cf_mae',
        'cf_mfe',
        'cf_bars_held'
    ])
    
    def to_dict(self) -> dict:
        return {
            'x_price_2h_1m': self.x_price_2h_1m,
            'x_price_2h_5m': self.x_price_2h_5m,
            'x_price_2h_15m': self.x_price_2h_15m,
            'x_context_dim': self.x_context_dim,
            'x_context_features': self.x_context_features,
            'y_classification': self.y_classification,
            'y_regression': self.y_regression,
        }
    
    def get_label_idx(self, label: str) -> int:
        """Get index for classification label."""
        return self.y_classification.index(label)
    
    def label_from_idx(self, idx: int) -> str:
        """Get label name from index."""
        return self.y_classification[idx]


# Default schema
DEFAULT_SCHEMA = DatasetSchema()


def validate_record_schema(record, schema: DatasetSchema = None) -> bool:
    """
    Validate that a DecisionRecord matches the schema.
    """
    schema = schema or DEFAULT_SCHEMA
    
    # Check price windows
    if record.x_price_1m is not None:
        expected = schema.x_price_2h_1m
        actual = record.x_price_1m.shape
        if actual != expected:
            return False
    
    # Check context vector
    if record.x_context is not None:
        if len(record.x_context) != schema.x_context_dim:
            return False
    
    # Check label is valid
    if record.cf_outcome and record.cf_outcome not in schema.y_classification:
        return False
    
    return True

```

### src/datasets/trade_record.py

```python
"""
Trade Record
Record of a completed trade (after exit).
"""

import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class TradeRecord:
    """
    Record of a completed trade.
    
    Only created when a trade exits (via SL, TP, or timeout).
    """
    
    # Identifiers
    trade_id: str = ""
    decision_id: str = ""          # Links to original DecisionRecord
    
    # Entry
    entry_time: Optional[pd.Timestamp] = None
    entry_bar: int = 0
    entry_price: float = 0.0
    direction: str = ""
    
    # Exit
    exit_time: Optional[pd.Timestamp] = None
    exit_bar: int = 0
    exit_price: float = 0.0
    exit_reason: str = ""          # 'SL', 'TP', 'TIMEOUT', 'MANUAL'
    
    # Outcome
    outcome: str = ""              # 'WIN', 'LOSS', 'TIMEOUT'
    pnl_points: float = 0.0
    pnl_dollars: float = 0.0
    gross_pnl: float = 0.0
    commission: float = 0.0
    
    # Analytics
    bars_held: int = 0
    mae: float = 0.0               # Max Adverse Excursion
    mfe: float = 0.0               # Max Favorable Excursion
    r_multiple: float = 0.0        # PnL / initial risk
    
    # Context at entry
    scanner_id: str = ""
    entry_atr: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'trade_id': self.trade_id,
            'decision_id': self.decision_id,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'entry_bar': self.entry_bar,
            'entry_price': self.entry_price,
            'direction': self.direction,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'exit_bar': self.exit_bar,
            'exit_price': self.exit_price,
            'exit_reason': self.exit_reason,
            'outcome': self.outcome,
            'pnl_points': self.pnl_points,
            'pnl_dollars': self.pnl_dollars,
            'bars_held': self.bars_held,
            'mae': self.mae,
            'mfe': self.mfe,
            'r_multiple': self.r_multiple,
            'scanner_id': self.scanner_id,
        }

```

### src/datasets/writer.py

```python
"""
Shard Writer
Write decision records to sharded parquet files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
import json
import uuid

from src.datasets.decision_record import DecisionRecord
from src.config import SHARDS_DIR


class ShardWriter:
    """
    Write DecisionRecords to sharded files.
    
    Features:
    - Fixed number of records per shard
    - Parquet format for efficient storage
    - Separate files for numpy arrays (optional)
    """
    
    def __init__(
        self,
        output_dir: Path = None,
        records_per_shard: int = 10000,
        experiment_id: str = None
    ):
        self.output_dir = Path(output_dir or SHARDS_DIR)
        self.records_per_shard = records_per_shard
        self.experiment_id = experiment_id or str(uuid.uuid4())[:8]
        
        # Ensure directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Buffer
        self._buffer: List[DecisionRecord] = []
        self._shard_idx = 0
        self._total_records = 0
        
        # Arrays storage
        self._arrays_dir = self.output_dir / "arrays"
        self._arrays_dir.mkdir(exist_ok=True)
    
    def write(self, record: DecisionRecord):
        """Add a record to the buffer."""
        self._buffer.append(record)
        self._total_records += 1
        
        if len(self._buffer) >= self.records_per_shard:
            self._flush_shard()
    
    def write_batch(self, records: List[DecisionRecord]):
        """Write multiple records."""
        for record in records:
            self.write(record)
    
    def _flush_shard(self):
        """Write buffered records to a shard file."""
        if not self._buffer:
            return
        
        # Convert to DataFrame
        rows = []
        array_refs = []
        
        for i, record in enumerate(self._buffer):
            row = record.to_dict()
            
            # Store numpy arrays separately
            record_id = f"{self.experiment_id}_{self._shard_idx}_{i}"
            
            if record.x_price_1m is not None:
                arr_path = self._save_array(record.x_price_1m, f"{record_id}_x_price_1m")
                row['x_price_1m_path'] = str(arr_path)
            
            if record.x_price_5m is not None:
                arr_path = self._save_array(record.x_price_5m, f"{record_id}_x_price_5m")
                row['x_price_5m_path'] = str(arr_path)
            
            if record.x_price_15m is not None:
                arr_path = self._save_array(record.x_price_15m, f"{record_id}_x_price_15m")
                row['x_price_15m_path'] = str(arr_path)
            
            if record.x_context is not None:
                arr_path = self._save_array(record.x_context, f"{record_id}_x_context")
                row['x_context_path'] = str(arr_path)
            
            rows.append(row)
        
        # Write parquet
        df = pd.DataFrame(rows)
        shard_path = self.output_dir / f"shard_{self._shard_idx:04d}.parquet"
        df.to_parquet(shard_path)
        
        # Clear buffer
        self._buffer = []
        self._shard_idx += 1
    
    def _save_array(self, arr: np.ndarray, name: str) -> Path:
        """Save numpy array to file."""
        path = self._arrays_dir / f"{name}.npy"
        np.save(path, arr)
        return path
    
    def close(self):
        """Flush remaining records and write metadata."""
        self._flush_shard()
        
        # Write manifest
        manifest = {
            'experiment_id': self.experiment_id,
            'total_records': self._total_records,
            'num_shards': self._shard_idx,
            'records_per_shard': self.records_per_shard,
        }
        
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

```

### src/eval/__init__.py

```python
# Eval module
"""Trade-quality metrics and breakdowns."""

```

### src/eval/breakdown.py

```python
"""
Breakdown Analysis
Metrics by setup, time of day, volatility regime.
"""

from typing import List, Dict
from collections import defaultdict

from src.datasets.decision_record import DecisionRecord
from src.eval.metrics import TradeMetrics, compute_from_records


def breakdown_by_scanner(
    records: List[DecisionRecord]
) -> Dict[str, TradeMetrics]:
    """Metrics grouped by scanner/setup type."""
    by_scanner = defaultdict(list)
    
    for r in records:
        by_scanner[r.scanner_id].append(r)
    
    return {k: compute_from_records(v) for k, v in by_scanner.items()}


def breakdown_by_hour(
    records: List[DecisionRecord]
) -> Dict[int, TradeMetrics]:
    """Metrics grouped by hour (NY time)."""
    by_hour = defaultdict(list)
    
    for r in records:
        if r.timestamp:
            hour = r.timestamp.hour
            by_hour[hour].append(r)
    
    return {k: compute_from_records(v) for k, v in sorted(by_hour.items())}


def breakdown_by_day(
    records: List[DecisionRecord]
) -> Dict[int, TradeMetrics]:
    """Metrics grouped by day of week (0=Mon, 4=Fri)."""
    by_day = defaultdict(list)
    
    for r in records:
        if r.timestamp:
            dow = r.timestamp.weekday()
            by_day[dow].append(r)
    
    return {k: compute_from_records(v) for k, v in sorted(by_day.items())}


def breakdown_by_action(
    records: List[DecisionRecord]
) -> Dict[str, dict]:
    """
    Analyze by action taken.
    
    Returns stats for:
    - Trades taken
    - Trades skipped (broken down by skip reason)
    - Skipped good (would have lost)
    - Skipped bad (would have won)
    """
    taken = [r for r in records if r.is_trade()]
    skipped = [r for r in records if r.was_skipped()]
    
    skipped_good = [r for r in skipped if r.is_good_skip()]
    skipped_bad = [r for r in skipped if r.is_bad_skip()]
    
    return {
        'taken': {
            'count': len(taken),
            'metrics': compute_from_records(taken),
        },
        'skipped': {
            'count': len(skipped),
            'good_skips': len(skipped_good),
            'bad_skips': len(skipped_bad),
            'good_skip_rate': len(skipped_good) / len(skipped) if skipped else 0,
        },
        'by_skip_reason': _count_skip_reasons(skipped),
    }


def _count_skip_reasons(records: List[DecisionRecord]) -> Dict[str, int]:
    """Count records by skip reason."""
    counts = defaultdict(int)
    for r in records:
        counts[r.skip_reason.value] += 1
    return dict(counts)


def print_breakdown_summary(
    records: List[DecisionRecord],
    title: str = "Breakdown Summary"
):
    """Print formatted breakdown summary."""
    print(f"\n{'='*50}")
    print(title)
    print('='*50)
    
    # Overall
    overall = compute_from_records(records)
    print(f"\nOverall: {overall.total_trades} records, "
          f"{overall.win_rate:.1%} WR, "
          f"${overall.total_pnl:.2f} PnL")
    
    # By hour
    print("\nBy Hour (NY):")
    by_hour = breakdown_by_hour(records)
    for hour, m in by_hour.items():
        if m.total_trades > 0:
            print(f"  {hour:02d}:00 - {m.total_trades:4d} trades, "
                  f"{m.win_rate:.1%} WR, ${m.total_pnl:7.2f}")
    
    # By action
    print("\nBy Action:")
    by_action = breakdown_by_action(records)
    print(f"  Taken: {by_action['taken']['count']}")
    print(f"  Skipped: {by_action['skipped']['count']} "
          f"({by_action['skipped']['good_skips']} good, "
          f"{by_action['skipped']['bad_skips']} bad)")

```

### src/eval/mae_mfe.py

```python
"""
MAE/MFE Analysis
Max Adverse/Favorable Excursion distributions.
"""

from dataclasses import dataclass
from typing import List
import numpy as np

from src.datasets.decision_record import DecisionRecord


@dataclass
class ExcursionMetrics:
    """MAE/MFE distribution metrics."""
    # MAE (Max Adverse Excursion - how much against you)
    mae_mean: float
    mae_std: float
    mae_median: float
    mae_max: float
    
    # MFE (Max Favorable Excursion - how much for you)
    mfe_mean: float
    mfe_std: float
    mfe_median: float
    mfe_max: float
    
    # Distributions
    mae_distribution: np.ndarray
    mfe_distribution: np.ndarray


def compute_excursions(records: List[DecisionRecord]) -> ExcursionMetrics:
    """Compute MAE/MFE metrics from decision records."""
    if not records:
        return ExcursionMetrics(
            mae_mean=0, mae_std=0, mae_median=0, mae_max=0,
            mfe_mean=0, mfe_std=0, mfe_median=0, mfe_max=0,
            mae_distribution=np.array([]),
            mfe_distribution=np.array([])
        )
    
    mae_values = np.array([r.cf_mae for r in records])
    mfe_values = np.array([r.cf_mfe for r in records])
    
    return ExcursionMetrics(
        mae_mean=np.mean(mae_values),
        mae_std=np.std(mae_values),
        mae_median=np.median(mae_values),
        mae_max=np.max(mae_values),
        mfe_mean=np.mean(mfe_values),
        mfe_std=np.std(mfe_values),
        mfe_median=np.median(mfe_values),
        mfe_max=np.max(mfe_values),
        mae_distribution=mae_values,
        mfe_distribution=mfe_values,
    )


def compute_excursions_by_outcome(
    records: List[DecisionRecord]
) -> dict:
    """Compute MAE/MFE separately for wins and losses."""
    wins = [r for r in records if r.cf_outcome == 'WIN']
    losses = [r for r in records if r.cf_outcome == 'LOSS']
    
    return {
        'wins': compute_excursions(wins),
        'losses': compute_excursions(losses),
        'all': compute_excursions(records),
    }

```

### src/eval/metrics.py

```python
"""
Trade Metrics
Expectancy, win rate, payoff ratio, drawdown.
"""

from dataclasses import dataclass
from typing import List
import numpy as np

from src.datasets.trade_record import TradeRecord
from src.datasets.decision_record import DecisionRecord


@dataclass
class TradeMetrics:
    """Comprehensive trade metrics."""
    total_trades: int
    wins: int
    losses: int
    timeouts: int
    win_rate: float
    
    # PnL
    total_pnl: float
    avg_pnl: float
    avg_win: float
    avg_loss: float
    
    # Risk-adjusted
    payoff_ratio: float      # avg_win / avg_loss
    expectancy: float        # (WR * avg_win) - ((1-WR) * avg_loss)
    profit_factor: float     # gross_win / gross_loss
    
    # Drawdown
    max_drawdown: float
    max_drawdown_pct: float
    
    # Additional
    avg_bars_held: float
    avg_r_multiple: float


def compute_trade_metrics(trades: List[TradeRecord]) -> TradeMetrics:
    """Compute metrics from trade records."""
    if not trades:
        return TradeMetrics(
            total_trades=0, wins=0, losses=0, timeouts=0, win_rate=0,
            total_pnl=0, avg_pnl=0, avg_win=0, avg_loss=0,
            payoff_ratio=0, expectancy=0, profit_factor=0,
            max_drawdown=0, max_drawdown_pct=0,
            avg_bars_held=0, avg_r_multiple=0
        )
    
    # Basic counts
    wins = [t for t in trades if t.outcome == 'WIN']
    losses = [t for t in trades if t.outcome == 'LOSS']
    timeouts = [t for t in trades if t.outcome == 'TIMEOUT']
    
    win_count = len(wins)
    loss_count = len(losses)
    total = len(trades)
    
    win_rate = win_count / total if total > 0 else 0
    
    # PnL
    total_pnl = sum(t.pnl_dollars for t in trades)
    avg_pnl = total_pnl / total if total > 0 else 0
    
    avg_win = np.mean([t.pnl_dollars for t in wins]) if wins else 0
    avg_loss = np.mean([abs(t.pnl_dollars) for t in losses]) if losses else 0
    
    # Risk-adjusted
    payoff_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    
    gross_win = sum(t.pnl_dollars for t in wins) if wins else 0
    gross_loss = abs(sum(t.pnl_dollars for t in losses)) if losses else 0
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float('inf')
    
    # Drawdown
    equity_curve = np.cumsum([t.pnl_dollars for t in trades])
    peak = np.maximum.accumulate(equity_curve)
    drawdowns = peak - equity_curve
    max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0
    max_dd_pct = max_dd / np.max(peak) if np.max(peak) > 0 else 0
    
    # Additional
    avg_bars = np.mean([t.bars_held for t in trades])
    avg_r = np.mean([t.r_multiple for t in trades if t.r_multiple != 0])
    
    return TradeMetrics(
        total_trades=total,
        wins=win_count,
        losses=loss_count,
        timeouts=len(timeouts),
        win_rate=win_rate,
        total_pnl=total_pnl,
        avg_pnl=avg_pnl,
        avg_win=avg_win,
        avg_loss=avg_loss,
        payoff_ratio=payoff_ratio,
        expectancy=expectancy,
        profit_factor=profit_factor,
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd_pct,
        avg_bars_held=avg_bars,
        avg_r_multiple=avg_r if not np.isnan(avg_r) else 0,
    )


def compute_from_records(records: List[DecisionRecord]) -> TradeMetrics:
    """Compute metrics from decision records using counterfactual outcomes."""
    if not records:
        return compute_trade_metrics([])
    
    # Convert counterfactual outcomes to simple format
    class SimpleRecord:
        def __init__(self, r: DecisionRecord):
            self.outcome = r.cf_outcome
            self.pnl_dollars = r.cf_pnl_dollars
            self.bars_held = r.cf_bars_held
            self.r_multiple = 0  # Not tracked in decision records
    
    simple = [SimpleRecord(r) for r in records if r.cf_outcome in ['WIN', 'LOSS', 'TIMEOUT']]
    
    # Reuse computation
    return compute_trade_metrics(simple)

```

### src/experiments/__init__.py

```python
# Experiments module
"""Experiment framework - configs, sweeps, reports."""

```

### src/experiments/config.py

```python
"""
Experiment Configuration
Central config dataclass for experiments.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
from enum import Enum
import json

from src.features.pipeline import FeatureConfig
from src.labels.labeler import LabelConfig
from src.sim.oco_engine import OCOConfig
from src.sim.bar_fill_model import BarFillConfig
from src.sim.costs import CostModel, DEFAULT_COSTS
from src.models.train import TrainConfig
from src.datasets.schema import DatasetSchema


from src.core.enums import RunMode


@dataclass
class ReplayConfig:
    """
    Configuration for replay mode.
    
    Controls how to step through historical data in replay mode.
    """
    # Time range
    start_bar: int = 0
    end_bar: Optional[int] = None
    
    # Playback controls
    speed_multiplier: float = 1.0  # 1.0 = real-time, 2.0 = 2x speed, etc.
    auto_play: bool = True
    pause_on_decision: bool = False
    
    # What to show
    show_future_bars: int = 20  # How many bars ahead to display
    show_oco_zones: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'start_bar': self.start_bar,
            'end_bar': self.end_bar,
            'speed_multiplier': self.speed_multiplier,
            'auto_play': self.auto_play,
            'pause_on_decision': self.pause_on_decision,
            'show_future_bars': self.show_future_bars,
            'show_oco_zones': self.show_oco_zones,
        }


@dataclass
class ExperimentConfig:
    """
    Complete experiment configuration.
    
    Single source of truth for all parameters.
    """
    # Identification
    name: str = "experiment"
    description: str = ""
    
    # Run mode
    run_mode: RunMode = RunMode.TRAIN
    
    # Replay configuration (only used when run_mode == REPLAY)
    replay_config: ReplayConfig = field(default_factory=ReplayConfig)
    
    # Data range
    start_date: str = ""
    end_date: str = ""
    timeframe: str = "1m"
    
    # Scanner
    scanner_id: str = "always"
    scanner_params: Dict[str, Any] = field(default_factory=dict)
    
    # Features
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    
    # Labels
    label_config: LabelConfig = field(default_factory=LabelConfig)
    oco_config: OCOConfig = field(default_factory=OCOConfig)
    
    # Simulation
    fill_config: BarFillConfig = field(default_factory=BarFillConfig)
    cost_model: CostModel = field(default_factory=lambda: DEFAULT_COSTS)
    
    # Training
    train_config: TrainConfig = field(default_factory=TrainConfig)
    
    # Schema
    schema: DatasetSchema = field(default_factory=DatasetSchema)
    
    # Reproducibility
    seed: int = 42
    
    # Performance options
    compute_cf: bool = True  # Whether to compute counterfactual outcomes (slower but more accurate)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'run_mode': self.run_mode.value,
            'replay_config': self.replay_config.to_dict(),
            'start_date': self.start_date,
            'end_date': self.end_date,
            'timeframe': self.timeframe,
            'scanner_id': self.scanner_id,
            'scanner_params': self.scanner_params,
            'feature_config': self.feature_config.to_dict(),
            'oco_config': self.oco_config.to_dict(),
            'fill_config': self.fill_config.to_dict(),
            'train_config': self.train_config.to_dict(),
            'schema': self.schema.to_dict(),
            'seed': self.seed,
        }
    
    def to_cli_args(self) -> List[str]:
        """Generate CLI arguments."""
        args = [
            '--name', self.name,
            '--start-date', self.start_date,
            '--end-date', self.end_date,
            '--timeframe', self.timeframe,
            '--scanner', self.scanner_id,
            '--seed', str(self.seed),
        ]
        args.extend(self.oco_config.to_cli_args())
        return args
    
    def save(self, path: Path):
        """Save config to JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    @classmethod
    def load(cls, path: Path) -> 'ExperimentConfig':
        """Load config from JSON."""
        with open(path) as f:
            data = json.load(f)
        
        config = cls(
            name=data.get('name', 'experiment'),
            description=data.get('description', ''),
            start_date=data.get('start_date', ''),
            end_date=data.get('end_date', ''),
            timeframe=data.get('timeframe', '1m'),
            scanner_id=data.get('scanner_id', 'always'),
            seed=data.get('seed', 42),
        )
        
        # Load nested configs if present
        if 'oco_config' in data:
            oco = data['oco_config']
            config.oco_config = OCOConfig(
                direction=oco.get('direction', 'LONG'),
                tp_multiple=oco.get('tp_multiple', 1.4),
                stop_atr=oco.get('stop_atr', 1.0),
            )
        
        return config

```

### src/experiments/fast_forward.py

```python
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

class EventScheduler:
    """
    Identifies 'interesting' bars where a decision might be needed.
    Used to skip empty periods in bar-by-bar simulation.
    """
    
    @staticmethod
    def get_events(df: pd.DataFrame, recipe: Dict[str, Any]) -> Optional[List[int]]:
        """
        Get sorted list of indices where signals MIGHT occur.
        Returns None if optimization is not possible (must check all bars).
        
        Args:
            df: DataFrame with OHLCV data
            recipe: Strategy recipe dictionary
        """
        try:
            trigger = recipe.get("entry_trigger", {})
            t_type = trigger.get("type", "").lower()
            
            # 1. EMA Cross
            if "ema_cross" in t_type:
                return EventScheduler._scan_ema_cross(df, trigger)
            
            # 2. RSI Extreme
            if "rsi" in t_type:
                return EventScheduler._scan_rsi(df, trigger)
                
            # 3. Composite (AND/OR) - specific case for common combos?
            # For now, default to None (safe mode)
            
            print(f"[FastForward] Unknown trigger type '{t_type}', running full simulation.")
            return None
            
        except Exception as e:
            print(f"[FastForward] Error predicting events: {e}. Running full simulation.")
            return None

    @staticmethod
    def _scan_ema_cross(df: pd.DataFrame, config: Dict[str, Any]) -> List[int]:
        """Vectorized scan for EMA Cross."""
        fast_len = config.get("fast", 9)
        slow_len = config.get("slow", 21)
        
        close = df['close']
        fast_ema = close.ewm(span=fast_len, adjust=False).mean()
        slow_ema = close.ewm(span=slow_len, adjust=False).mean()
        
        # Find where relation changes
        # fast > slow
        above = fast_ema > slow_ema
        
        # Cross occurred where 'above' value changes
        # shift(1) compares current to prev
        # fillna(False) to handle first bar
        crosses = (above != above.shift(1)).fillna(False)
        
        # Get indices
        # We need to ensure we return the index of the BAR that completes the cross
        # indices here are from df.index.
        # But we need INTEGER indices for the stepper.
        # df.index might be DatetimeIndex.
        # We need iloc positions.
        
        indices = np.where(crosses)[0].tolist()
        
        print(f"[FastForward] Found {len(indices)} potential EMA Check points")
        return indices

    @staticmethod
    def _scan_rsi(df: pd.DataFrame, config: Dict[str, Any]) -> List[int]:
        """Vectorized scan for RSI Extreme."""
        length = config.get("length", 14)
        oversold = config.get("oversold", 30)
        overbought = config.get("overbought", 70)
        
        close = df['close']
        delta = close.diff()
        
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        
        avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan) # Handle div by zero
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50)
        
        # Check condition
        mask = (rsi <= oversold) | (rsi >= overbought)
        
        indices = np.where(mask)[0].tolist()
        print(f"[FastForward] Found {len(indices)} potential RSI Check points")
        return indices

```

### src/experiments/fingerprint.py

```python
"""
Experiment Fingerprint
SHA256 hash for reproducibility tracking.
"""

import hashlib
import json
from typing import Any

from src.experiments.config import ExperimentConfig


def compute_fingerprint(config: ExperimentConfig) -> str:
    """
    Compute SHA256 fingerprint of experiment configuration.
    
    Ensures reproducibility tracking - same config = same fingerprint.
    
    Returns:
        First 16 characters of SHA256 hash
    """
    # Serialize config to deterministic JSON
    config_dict = config.to_dict()
    
    # Sort keys for determinism
    json_str = json.dumps(config_dict, sort_keys=True, default=str)
    
    # Compute hash
    hash_obj = hashlib.sha256(json_str.encode())
    
    return hash_obj.hexdigest()[:16]


def verify_fingerprint(
    config: ExperimentConfig,
    expected: str
) -> bool:
    """
    Verify that config matches expected fingerprint.
    """
    actual = compute_fingerprint(config)
    return actual == expected

```

### src/experiments/report.py

```python
"""
Report Generation
Generate markdown reports from experiment results.
"""

from pathlib import Path
from typing import List
import pandas as pd

from src.experiments.runner import ExperimentResult
from src.config import RESULTS_DIR


def generate_report(
    results: List[ExperimentResult],
    output_path: Path = None
) -> Path:
    """
    Generate markdown report from experiment results.
    """
    output_path = output_path or RESULTS_DIR / "report.md"
    
    lines = [
        "# Experiment Report",
        "",
        f"Generated: {pd.Timestamp.now()}",
        "",
        f"Total experiments: {len(results)}",
        "",
        "## Summary",
        "",
    ]
    
    # Create summary table
    lines.extend([
        "| Name | Records | WIN | LOSS | Best Val Loss | Best Epoch |",
        "|------|---------|-----|------|---------------|------------|",
    ])
    
    for r in results:
        val_loss = f"{r.train_result.best_val_loss:.4f}" if r.train_result else "N/A"
        epoch = str(r.train_result.best_epoch) if r.train_result else "N/A"
        
        lines.append(
            f"| {r.config.name} | {r.total_records} | {r.win_records} | "
            f"{r.loss_records} | {val_loss} | {epoch} |"
        )
    
    lines.extend(["", "## Configuration Details", ""])
    
    for r in results:
        lines.extend([
            f"### {r.config.name}",
            "",
            f"- Fingerprint: `{r.fingerprint}`",
            f"- Scanner: {r.config.scanner_id}",
            f"- Direction: {r.config.oco_config.direction}",
            f"- TP Multiple: {r.config.oco_config.tp_multiple}",
            f"- Stop ATR: {r.config.oco_config.stop_atr}",
            f"- Records: {r.total_records} ({r.win_records}W / {r.loss_records}L)",
            "",
        ])
    
    # Write file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Report saved to {output_path}")
    return output_path

```

### src/experiments/runner.py

```python
"""
Experiment Runner
Run a single experiment end-to-end.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING
import uuid

if TYPE_CHECKING:
    from src.viz.export import Exporter

from src.experiments.config import ExperimentConfig
from src.experiments.fingerprint import compute_fingerprint
from src.experiments.splits import generate_walk_forward_splits, WalkForwardConfig

from src.data.loader import load_continuous_contract
from src.data.resample import resample_all_timeframes

from src.sim.stepper import MarketStepper
from src.sim.oco_engine import create_oco_bracket
from src.sim.sizing import calculate_contracts, calculate_pnl_dollars
from src.sim.causal_runner import CausalExecutor, StepResult
from src.sim.account_manager import AccountManager
from src.features.pipeline import compute_features, precompute_indicators
from src.policy.scanners import get_scanner
from src.policy.filters import DEFAULT_FILTERS
from src.policy.cooldown import CooldownManager
from src.policy.actions import Action, SkipReason
from src.viz.window_utils import enforce_2hour_window

from src.labels.labeler import Labeler
from src.datasets.decision_record import DecisionRecord
from src.datasets.trade_record import TradeRecord
from src.datasets.writer import ShardWriter
from src.datasets.reader import create_dataloader

from src.models.fusion import FusionModel
from src.models.train import train_model, TrainResult

from src.config import PROCESSED_DIR, SHARDS_DIR, RESULTS_DIR, DEFAULT_MAX_RISK_DOLLARS


@dataclass
class ExperimentResult:
    """Result of running an experiment."""
    config: ExperimentConfig
    fingerprint: str
    
    # Dataset stats
    total_records: int
    win_records: int
    loss_records: int
    timeout_records: int
    
    # Training results
    train_result: Optional[TrainResult] = None

    # Financial metrics
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    
    # Created at
    created_at: pd.Timestamp = None
    
    def to_dict(self):
        return {
            'fingerprint': self.fingerprint,
            'total_records': self.total_records,
            'win_records': self.win_records,
            'loss_records': self.loss_records,
            'timeout_records': self.timeout_records,
            'best_val_loss': self.train_result.best_val_loss if self.train_result else None,
            'best_epoch': self.train_result.best_epoch if self.train_result else None,
        }


from src.experiments.fast_forward import EventScheduler


def run_experiment(
    config: ExperimentConfig,
    exporter: Optional['Exporter'] = None
) -> ExperimentResult:
    """
    Run a complete experiment:
    
    1. Load data
    2. Generate decision records at scanner points
    3. Label all records with counterfactual outcomes
    4. Write to shards
    5. Train model
    6. Return results
    
    Args:
        config: Experiment configuration
        exporter: Optional Exporter for viz output
    """
    print(f"Running experiment: {config.name}")
    
    # Compute fingerprint
    fingerprint = compute_fingerprint(config)
    print(f"Fingerprint: {fingerprint}")
    
    # 1. Load and prepare data (with date filtering for performance)
    print("Loading data...")
    df = load_continuous_contract(
        start_date=config.start_date,
        end_date=config.end_date,
        buffer_hours=2  # 2hr buffer before/after for context
    )
    
    # ========================================
    # CRITICAL: Add padding to date range for full window coverage
    # - START: 2 hours before first potential trade entry
    # - END: max_bars (trade duration) + 2 hours post-exit window
    #   A trade at end_date can hold for max_bars minutes, 
    #   then we need 2 more hours for the post-exit window
    # Without this, enforce_2hour_window() will emit warnings
    # ========================================
    from datetime import timedelta
    
    # Get max_bars from OCO config (default 50 minutes if not set)
    max_bars = config.oco_config.max_bars if config.oco_config else 50
    
    # Parse dates if they're strings and add padding
    start_dt = pd.to_datetime(config.start_date) if config.start_date else None
    end_dt = pd.to_datetime(config.end_date) if config.end_date else None
    
    padded_start = start_dt - timedelta(hours=2) if start_dt is not None else None
    # End needs: max_bars (trade can still be open) + 2 hours (post-exit window)
    padded_end = end_dt + timedelta(minutes=max_bars, hours=2) if end_dt is not None else None
    
    # Localize to match df['time'] timezone if needed
    if padded_start is not None and padded_start.tzinfo is None and df['time'].dt.tz is not None:
        padded_start = padded_start.tz_localize(df['time'].dt.tz)
    if padded_end is not None and padded_end.tzinfo is None and df['time'].dt.tz is not None:
        padded_end = padded_end.tz_localize(df['time'].dt.tz)
    
    # Filter by padded date range
    if padded_start is not None:
        df = df[df['time'] >= padded_start]
    if padded_end is not None:
        df = df[df['time'] <= padded_end]
    
    df = df.reset_index(drop=True)
    print(f"Data range: {df['time'].min()} to {df['time'].max()}")
    print(f"Total bars: {len(df)} (padded by 2h on each side)")
    
    # Resample to higher timeframes
    htf_data = resample_all_timeframes(df)
    df_5m = htf_data.get('5m')
    df_15m = htf_data.get('15m')
    
    # Precompute indicators
    print("Precomputing indicators...")
    indicators_map = precompute_indicators(df, df_5m, df_15m)
    
    # 2. Generate decision records
    print("Generating decision records...")
    
    # ========================================
    # CRITICAL: Constrain stepper to ORIGINAL date range
    # The padded data provides 2h context BEFORE first decision
    # and max_bars + 2h AFTER last decision
    # But decisions should ONLY occur within original start_date to end_date
    # ========================================
    
    # Find indices where original dates fall in the padded data
    if start_dt is not None:
        # Make start_dt timezone-aware if needed
        original_start = start_dt
        if original_start.tzinfo is None and df['time'].dt.tz is not None:
            original_start = original_start.tz_localize(df['time'].dt.tz)
        start_mask = df['time'] >= original_start
        decision_start_idx = start_mask.idxmax() if start_mask.any() else 200
    else:
        decision_start_idx = 200
        
    if end_dt is not None:
        # Make end_dt timezone-aware if needed
        original_end = end_dt
        if original_end.tzinfo is None and df['time'].dt.tz is not None:
            original_end = original_end.tz_localize(df['time'].dt.tz)
        end_mask = df['time'] <= original_end
        # Get the LAST index where time <= end_dt
        if end_mask.any():
            decision_end_idx = df[end_mask].index[-1]
        else:
            decision_end_idx = len(df) - 200
    else:
        decision_end_idx = len(df) - 200
    
    print(f"Decisions restricted to indices {decision_start_idx} to {decision_end_idx} (original date range)")
    stepper = MarketStepper(df, start_idx=decision_start_idx, end_idx=decision_end_idx)
    scanner = get_scanner(config.scanner_id, **config.scanner_params)
    
    # ========================================
    # FAST FORWARD PRE-SCAN
    # ========================================
    event_indices = None
    recipe = None
    if isinstance(config.scanner_params, dict) and 'entry_trigger' in config.scanner_params:
         recipe = config.scanner_params
    
    if recipe:
        print("[FastForward] Pre-scanning for events...")
        event_indices = EventScheduler.get_events(df, recipe)
        
        if event_indices is not None:
            # Filter valid range
            event_indices = sorted([i for i in event_indices if decision_start_idx <= i < decision_end_idx])
            print(f"[FastForward] Optimized: {len(event_indices)} potential events found. Skipping empty periods.")
    
    event_ptr = 0
    # ========================================

    # CRITICAL: Ensure labeler uses the SAME oco_config as viz export
    # Otherwise bars_held will mismatch the displayed TP/SL levels
    label_config = config.label_config
    label_config.oco_config = config.oco_config  # Override to ensure consistency
    labeler = Labeler(label_config)
    
    cooldown = CooldownManager()
    
    # Initialize Causal Executor
    # Note: Experiment runner uses its own strict stepper, so we pass it in.
    # We also use a dummy AccountManager as we are just checking for signals/records here.
    account_manager = AccountManager()
    executor = CausalExecutor(
        df=df,
        stepper=stepper,
        account_manager=account_manager,
        scanner=scanner,
        feature_config=config.feature_config,
        df_5m=df_5m,
        df_15m=df_15m,
        precomputed_indicators=indicators_map
    )
    
    records: List[DecisionRecord] = []
    
    while True:
        # Fast Forward Logic
        if event_indices and not executor.active_brackets:
            # If no open trades, we can safely jump to next event
            current = stepper.current_idx
            
            # Find next event >= current
            while event_ptr < len(event_indices) and event_indices[event_ptr] < current:
                event_ptr += 1
            
            if event_ptr < len(event_indices):
                next_event_idx = event_indices[event_ptr]
                if next_event_idx > current:
                    stepper.skip_to(next_event_idx)
        
        # Step the unified executor
        result = executor.step()
        if not result:
            break
            
        # If scanner triggered, we have a potential record
        # In CausalExecutor, triggers are in result.scanner_triggers and result.new_orders
        # We need to map this back to DecisionRecord format.
        
        # We only care if meaningful decision occurred (scanner checked)
        # CausalExecutor runs scanner every step if provided.
        # But we only want to RECORD if it triggered or if we want negative samples?
        # The original code recorded ONLY IF skip_reason != SKIP (or if it was filter blocked).
        # Actually original code recorded ALL scan attempts that passed basic checks?
        # Original: "if not scan_result.triggered: continue"
        
        # So we check if triggered.
        if not result.scanner_triggers:
            continue
            
        # Extract the first trigger (assuming one per bar for now)
        trigger = result.scanner_triggers[0]
        # And the bracket (order) if any
        bracket_ref = result.new_orders[0] if result.new_orders else None
        
        # Get direction from scanner context (dynamic) or fall back to config (static)
        # Note: trigger could be a ScanResult object or dict depending on CausalExecutor
        if hasattr(trigger, 'context'):
            scanner_context = trigger.context
        else:
            scanner_context = trigger.get('context', {}) if isinstance(trigger, dict) else {}
        scanner_direction = scanner_context.get('direction') if scanner_context else None
        effective_direction = scanner_direction or config.oco_config.direction
        
        # Features are available in result
        features = result.features
        
        # Re-verify filters/cooldown using the centralized logic or here?
        # CausalExecutor creates the order if triggered. It doesn't check "cooldown" from policy/cooldown.py
        # because that's a higher-level policy. 
        # Wait, if CausalExecutor creates the order, it implies it passed checks?
        # The current CausalExecutor implementation is bare-bones: Trigger -> Order.
        # It misses the Filter/Cooldown/Skip logic from the old runner.
        
        # To maintain exact parity, we should move Filter/Cooldown INTO CausalExecutor?
        # Or check it here and "Cancel" the order?
        # FOR NOW: We will re-implement the check here to decide SKIP vs PLACE, matching old runner.
        # Ideally, CausalExecutor should take a 'Policy' object that handles this.
        
        # Check filters
        filter_result = DEFAULT_FILTERS.check(features)
        if not filter_result.passed:
            skip_reason = SkipReason.FILTER_BLOCK
        # Check cooldown
        elif cooldown.is_on_cooldown(result.bar_idx, result.timestamp)[0]:
            skip_reason = SkipReason.COOLDOWN
        else:
            skip_reason = SkipReason.NOT_SKIPPED
            
        # Determine Action
        action = Action.NO_TRADE if skip_reason != SkipReason.NOT_SKIPPED else Action.PLACE_ORDER
        
        # If we skipped, we technically "cancelled" the order the executor made.
        # But for 'Generating Data', we just record the decision.
        
        # Create record
        # CRITICAL: Include direction in scanner_context for UI position boxes
        context_with_direction = {**(scanner_context or {}), 'direction': effective_direction}
        
        record = DecisionRecord(
            timestamp=result.timestamp,
            bar_idx=result.bar_idx,
            decision_id=str(uuid.uuid4())[:8],
            scanner_id=config.scanner_id,
            scanner_context=context_with_direction,
            action=action,
            skip_reason=skip_reason,
            x_price_1m=features.x_price_1m,
            x_price_5m=features.x_price_5m,
            x_price_15m=features.x_price_15m,
            x_context=features.x_context,
            current_price=features.current_price,
            atr=features.atr,
        )
        
        # 3. Label with counterfactual outcome (TRAINING/DATA GEN ONLY)
        # This is the "Lookahead" step that we keep ONLY for data generation.
        # It uses the Labeler to jump ahead and see what happened.
        # CRITICAL: Must use effective_direction from scanner, not config default
        labeler.config.oco_config.direction = effective_direction
        cf_label = labeler.label_decision_point(df, result.bar_idx, features.atr)
        record.cf_outcome = cf_label.outcome
        record.cf_pnl = cf_label.pnl
        record.cf_pnl_dollars = cf_label.pnl_dollars
        record.cf_mae = cf_label.mae
        record.cf_mfe = cf_label.mfe
        record.cf_mae_atr = cf_label.mae_atr
        record.cf_mfe_atr = cf_label.mfe_atr
        record.cf_bars_held = cf_label.bars_held
        
        records.append(record)
        
        if exporter:
            curr_idx = result.bar_idx
            
            exit_time = None
            if record.action == Action.PLACE_ORDER:
                exit_time = features.timestamp + pd.Timedelta(minutes=record.cf_bars_held)

            raw_ohlcv, window_warning = enforce_2hour_window(
                df_1m=df,
                entry_time=features.timestamp,
                exit_time=exit_time,
                bars_held=record.cf_bars_held
            )

            if window_warning:
                exporter._window_warnings.append(window_warning)
            
            # Extract future bars separately (for compatibility)
            future_bars = []
            end_future_idx = min(len(df), curr_idx + 21)
            if end_future_idx > curr_idx + 1:
                future_slice = df.iloc[curr_idx+1 : end_future_idx]
                future_bars = future_slice[['open', 'high', 'low', 'close', 'volume']].values.tolist()
            
            # Extract indicator values for overlay
            ind = features.indicators
            indicators_dict = {}
            if ind:
                indicators_dict = {
                    'ema': ind.ema_5m_20,
                    'atr': ind.atr_5m_14,
                    'rsi': ind.rsi_5m_14,
                }

            exporter.on_decision(
                record, features, 
                future_1m=future_bars,
                raw_ohlcv=raw_ohlcv,
                indicators=indicators_dict
            )
            
            # Export Bracket if trade
            if record.action == Action.PLACE_ORDER:
                # Create bracket to visualize TP/SL (use scanner direction)
                bracket = create_oco_bracket(
                    config=config.oco_config,
                    base_price=features.current_price,
                    atr=features.atr,
                    direction_override=effective_direction
                )
                sizing_result = calculate_contracts(
                    entry_price=bracket.entry_price,
                    stop_price=bracket.stop_price,
                    max_risk_dollars=DEFAULT_MAX_RISK_DOLLARS
                )
                exporter.on_bracket_created(
                    record.decision_id,
                    bracket,
                    contracts=sizing_result.contracts
                )
        
        # Update cooldown if trade placed
        if record.action == Action.PLACE_ORDER:
            cooldown.record_trade(result.bar_idx, cf_label.outcome, features.timestamp)
            
            # Export Trade Record for Viz (Constructed from CF outcome)
            if exporter:
                # Approximate exit bar
                exit_bar = result.bar_idx + record.cf_bars_held
                exit_time = features.timestamp + pd.Timedelta(minutes=record.cf_bars_held)
                
                bracket = create_oco_bracket(
                    config=config.oco_config,
                    base_price=features.current_price,
                    atr=features.atr,
                    direction_override=effective_direction
                )
                sizing_result = calculate_contracts(
                    entry_price=bracket.entry_price,
                    stop_price=bracket.stop_price,
                    max_risk_dollars=DEFAULT_MAX_RISK_DOLLARS
                )
                exit_price = features.current_price + (
                    record.cf_pnl / (1 if effective_direction == "LONG" else -1)
                )
                pnl_points, pnl_dollars = calculate_pnl_dollars(
                    entry_price=features.current_price,
                    exit_price=exit_price,
                    direction=effective_direction,  # FIXED: was config.oco_config.direction
                    contracts=sizing_result.contracts
                )

                trade = TradeRecord(
                    trade_id=str(uuid.uuid4())[:8],
                    decision_id=record.decision_id,
                    entry_time=features.timestamp,
                    entry_bar=result.bar_idx,
                    entry_price=features.current_price,
                    direction=effective_direction,
                    exit_time=exit_time,
                    exit_bar=exit_bar,
                    exit_price=exit_price,
                    exit_reason=record.cf_outcome,
                    outcome=record.cf_outcome,
                    pnl_points=pnl_points,
                    pnl_dollars=pnl_dollars,
                    r_multiple=pnl_dollars / (features.atr * config.oco_config.stop_atr * 50) if features.atr > 0 else 0,
                    bars_held=record.cf_bars_held,
                    mae=record.cf_mae,
                    mfe=record.cf_mfe,
                    scanner_id=record.scanner_id,
                    entry_atr=features.atr
                )
                exporter.on_trade_closed(trade)
                
    print(f"Generated {len(records)} decision records")
    
    # Count outcomes
    win_count = sum(1 for r in records if r.cf_outcome == 'WIN')
    loss_count = sum(1 for r in records if r.cf_outcome == 'LOSS')
    timeout_count = sum(1 for r in records if r.cf_outcome == 'TIMEOUT')
    
    print(f"Outcomes: {win_count} WIN, {loss_count} LOSS, {timeout_count} TIMEOUT")
    
    # 4. Write to shards
    shard_dir = SHARDS_DIR / fingerprint
    print(f"Writing shards to {shard_dir}")
    
    with ShardWriter(shard_dir, experiment_id=fingerprint) as writer:
        for record in records:
            writer.write(record)
    
    # 5. Train model (if enough data)
    train_result = None
    if win_count + loss_count >= 100:
        print("Training model...")
        
        # Create dataloaders (simple 80/20 split for now)
        loader = create_dataloader(shard_dir, batch_size=config.train_config.batch_size)
        
        # Split into train/val
        dataset = loader.dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        from torch.utils.data import random_split, DataLoader
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_ds, batch_size=config.train_config.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config.train_config.batch_size)
        
        # Create model
        model = FusionModel(
            context_dim=config.schema.x_context_dim,
            num_classes=2,  # WIN/LOSS
            dropout=config.train_config.dropout,
        )
        
        # Train
        train_result = train_model(model, train_loader, val_loader, config.train_config)
    
    # Calculate financial metrics
    total_pnl = sum(r.cf_pnl_dollars for r in records if r.cf_outcome in ['WIN', 'LOSS'])
    trade_count = win_count + loss_count
    avg_pnl = total_pnl / trade_count if trade_count > 0 else 0.0

    # 6. Return results
    return ExperimentResult(
        config=config,
        fingerprint=fingerprint,
        total_records=len(records),
        win_records=win_count,
        loss_records=loss_count,
        timeout_records=timeout_count,
        train_result=train_result,
        total_pnl=total_pnl,
        avg_pnl=avg_pnl,
        created_at=pd.Timestamp.now(),
    )

```

### src/experiments/splits.py

```python
"""
Walk-Forward Splits
Time-series cross-validation with embargo.
"""

import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class WalkForwardConfig:
    """Walk-forward split configuration."""
    train_weeks: int = 3
    test_weeks: int = 1
    embargo_bars: int = 100  # Gap to prevent feature leakage
    min_train_records: int = 1000


@dataclass
class Split:
    """Single train/test split."""
    split_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    embargo_start: pd.Timestamp
    embargo_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    
    def __repr__(self):
        return (f"Split(train={self.train_start.date()}→{self.train_end.date()}, "
                f"test={self.test_start.date()}→{self.test_end.date()})")


def generate_walk_forward_splits(
    df: pd.DataFrame,
    config: WalkForwardConfig,
    time_col: str = 'time'
) -> List[Split]:
    """
    Generate walk-forward splits with embargo gaps.
    
    Layout:
    |---train---|--embargo--|---test---|---train---|--embargo--|---test---|
    
    Args:
        df: DataFrame with time column
        config: Split configuration
        time_col: Name of time column
        
    Returns:
        List of Split objects
    """
    times = pd.to_datetime(df[time_col]).sort_values()
    start_time = times.min()
    end_time = times.max()
    
    train_duration = pd.Timedelta(weeks=config.train_weeks)
    test_duration = pd.Timedelta(weeks=config.test_weeks)
    embargo_duration = pd.Timedelta(minutes=config.embargo_bars)  # Assuming 1m bars
    
    splits = []
    current_start = start_time
    split_idx = 0
    
    while True:
        train_end = current_start + train_duration
        embargo_start = train_end
        embargo_end = embargo_start + embargo_duration
        test_start = embargo_end
        test_end = test_start + test_duration
        
        # Check if we have enough data for this split
        if test_end > end_time:
            break
        
        split = Split(
            split_idx=split_idx,
            train_start=current_start,
            train_end=train_end,
            embargo_start=embargo_start,
            embargo_end=embargo_end,
            test_start=test_start,
            test_end=test_end,
        )
        splits.append(split)
        
        # Move to next window
        current_start = test_start  # Or test_end for non-overlapping
        split_idx += 1
    
    return splits


def apply_split(
    df: pd.DataFrame,
    split: Split,
    time_col: str = 'time'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply a split to get train and test DataFrames.
    """
    times = pd.to_datetime(df[time_col])
    
    train_mask = (times >= split.train_start) & (times < split.train_end)
    test_mask = (times >= split.test_start) & (times < split.test_end)
    
    return df[train_mask].copy(), df[test_mask].copy()


def apply_embargo_to_records(
    train_records: list,
    test_start: pd.Timestamp,
    embargo_bars: int
) -> list:
    """
    Remove training records within embargo window of test start.
    
    Prevents information leakage from rolling features.
    """
    # Calculate cutoff time
    embargo_duration = pd.Timedelta(minutes=embargo_bars)
    cutoff = test_start - embargo_duration
    
    # Filter records
    filtered = [r for r in train_records if r.timestamp < cutoff]
    
    removed = len(train_records) - len(filtered)
    if removed > 0:
        print(f"Embargo: removed {removed} records within {embargo_bars} bars of test start")
    
    return filtered

```

### src/experiments/strategy_config.py

```python
"""
Strategy Configuration
Serializable configuration for strategy runs.

This allows strategies to be parameterized and run from the agent or UI
without needing code changes.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
import json
from pathlib import Path


@dataclass
class StrategyConfig:
    """
    Complete strategy configuration for a run.
    
    This is the "public API" for configuring and running strategies.
    All parameters should be serializable and agent-controllable.
    """
    
    # Strategy identification
    strategy_id: str = "always"  # Scanner/strategy name
    strategy_params: Dict[str, Any] = field(default_factory=dict)
    
    # Data range
    start_date: str = ""
    end_date: str = ""
    timeframe: str = "1m"
    
    # OCO Configuration
    oco_direction: str = "LONG"  # or "SHORT"
    oco_tp_multiple: float = 1.4
    oco_stop_atr: float = 1.0
    oco_max_bars: int = 200
    oco_entry_type: str = "LIMIT"  # or "MARKET"
    
    # Feature toggles
    use_1m_features: bool = True
    use_5m_features: bool = True
    use_15m_features: bool = True
    use_1h_features: bool = False
    use_4h_features: bool = False
    
    # Filter parameters
    enable_filters: bool = True
    filter_min_volume: Optional[float] = None
    filter_session_only: Optional[str] = None  # "rth", "overnight", None
    
    # Cooldown
    cooldown_bars: int = 10
    
    # Training
    train_model: bool = False
    model_epochs: int = 10
    model_batch_size: int = 64
    
    # Output
    output_name: Optional[str] = None
    enable_viz_export: bool = True
    
    # Reproducibility
    seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyConfig':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_json(cls, json_str: str) -> 'StrategyConfig':
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def save(self, path: Path):
        """Save to JSON file."""
        with open(path, 'w') as f:
            f.write(self.to_json())
    
    @classmethod
    def load(cls, path: Path) -> 'StrategyConfig':
        """Load from JSON file."""
        with open(path) as f:
            return cls.from_json(f.read())
    
    def to_cli_args(self) -> list:
        """
        Convert to CLI arguments for backwards compatibility.
        
        This allows existing scripts to be called with this config.
        """
        args = [
            '--strategy', self.strategy_id,
            '--start-date', self.start_date,
            '--end-date', self.end_date,
            '--timeframe', self.timeframe,
            '--oco-tp', str(self.oco_tp_multiple),
            '--oco-stop', str(self.oco_stop_atr),
            '--seed', str(self.seed),
        ]
        
        if self.output_name:
            args.extend(['--out-name', self.output_name])
        
        if not self.enable_filters:
            args.append('--no-filters')
        
        if self.train_model:
            args.extend(['--train', '--epochs', str(self.model_epochs)])
        
        return args


# Preset configurations for common strategies
PRESET_CONFIGS = {
    "opening_range_default": StrategyConfig(
        strategy_id="opening_range",
        oco_direction="LONG",
        oco_tp_multiple=1.4,
        oco_stop_atr=1.0,
        use_1m_features=True,
        use_5m_features=True,
        use_15m_features=True,
    ),
    
    "opening_range_conservative": StrategyConfig(
        strategy_id="opening_range",
        oco_direction="LONG",
        oco_tp_multiple=1.0,
        oco_stop_atr=0.8,
        use_1m_features=True,
        use_5m_features=True,
        use_15m_features=True,
        filter_min_volume=1000.0,
    ),
    
    "opening_range_aggressive": StrategyConfig(
        strategy_id="opening_range",
        oco_direction="LONG",
        oco_tp_multiple=2.0,
        oco_stop_atr=1.2,
        use_1m_features=True,
        use_5m_features=True,
        use_15m_features=True,
    ),
    
    "always_default": StrategyConfig(
        strategy_id="always",
        oco_direction="LONG",
        oco_tp_multiple=1.4,
        oco_stop_atr=1.0,
        use_1m_features=True,
        use_5m_features=True,
        use_15m_features=True,
        use_1h_features=True,
        use_4h_features=True,
    ),
}


def get_preset_config(name: str) -> Optional[StrategyConfig]:
    """Get a preset configuration by name."""
    return PRESET_CONFIGS.get(name)

```

### src/experiments/sweep.py

```python
"""
Parameter Sweep
Run experiments across parameter grids.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any
import itertools
import copy
from pathlib import Path
import json

from src.experiments.config import ExperimentConfig
from src.experiments.runner import run_experiment, ExperimentResult
from src.config import RESULTS_DIR


@dataclass
class SweepConfig:
    """Configuration for parameter sweep."""
    base_config: ExperimentConfig
    sweep_params: Dict[str, List[Any]] = field(default_factory=dict)
    
    # e.g. {'oco_config.tp_multiple': [1.2, 1.4, 1.6],
    #       'oco_config.stop_atr': [0.8, 1.0, 1.2]}


def generate_sweep_configs(sweep: SweepConfig) -> List[ExperimentConfig]:
    """
    Generate all config combinations from sweep parameters.
    """
    if not sweep.sweep_params:
        return [sweep.base_config]
    
    # Generate all combinations
    param_names = list(sweep.sweep_params.keys())
    param_values = list(sweep.sweep_params.values())
    
    configs = []
    for values in itertools.product(*param_values):
        # Deep copy base config
        config = copy.deepcopy(sweep.base_config)
        
        # Apply parameter values
        for name, value in zip(param_names, values):
            _set_nested_attr(config, name, value)
        
        # Update name
        param_str = '_'.join(f"{n.split('.')[-1]}={v}" for n, v in zip(param_names, values))
        config.name = f"{sweep.base_config.name}_{param_str}"
        
        configs.append(config)
    
    return configs


def _set_nested_attr(obj, path: str, value):
    """Set nested attribute using dot notation."""
    parts = path.split('.')
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def run_sweep(sweep: SweepConfig) -> List[ExperimentResult]:
    """
    Run all experiments in a sweep.
    """
    configs = generate_sweep_configs(sweep)
    print(f"Running sweep with {len(configs)} configurations")
    
    results = []
    for i, config in enumerate(configs):
        print(f"\n--- Configuration {i+1}/{len(configs)}: {config.name} ---")
        
        try:
            result = run_experiment(config)
            results.append(result)
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    # Save sweep results
    sweep_id = sweep.base_config.name
    save_sweep_results(results, sweep_id)
    
    return results


def save_sweep_results(results: List[ExperimentResult], sweep_id: str):
    """Save sweep results to JSON."""
    output_path = RESULTS_DIR / f"sweep_{sweep_id}.json"
    
    data = {
        'sweep_id': sweep_id,
        'num_experiments': len(results),
        'results': [r.to_dict() for r in results],
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    print(f"Saved sweep results to {output_path}")

```

### src/features/__init__.py

```python
# Features module
"""Causal feature computation - no future leaking."""

```

### src/features/chart_indicators.ts

```typescript
/**
 * Chart Indicators - Reusable calculation module
 * 
 * This module provides indicator calculations that can be used by:
 * - Chart rendering (via useIndicators hook)
 * - Strategy triggers
 * - Backend analysis
 * 
 * All calculations are pure functions for easy testing and reuse.
 */

export interface OHLCV {
    time: number | string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume?: number;
}

export interface IndicatorPoint {
    time: number | string;
    value: number;
}

export interface BandPoint {
    time: number | string;
    upper: number;
    lower: number;
    middle?: number;
}

// =============================================================================
// EMA (Exponential Moving Average)
// =============================================================================

/**
 * Calculate EMA for a series of candles
 */
export function calculateEMA(candles: OHLCV[], period: number): IndicatorPoint[] {
    if (candles.length < period) return [];

    const multiplier = 2 / (period + 1);
    const result: IndicatorPoint[] = [];

    // Initial SMA for first EMA value
    let sum = 0;
    for (let i = 0; i < period; i++) {
        sum += candles[i].close;
    }
    let ema = sum / period;

    result.push({ time: candles[period - 1].time, value: ema });

    // Calculate EMA for remaining candles
    for (let i = period; i < candles.length; i++) {
        ema = (candles[i].close - ema) * multiplier + ema;
        result.push({ time: candles[i].time, value: ema });
    }

    return result;
}

// =============================================================================
// VWAP (Volume Weighted Average Price)
// =============================================================================

/**
 * Calculate VWAP - resets at each session start (9:30 ET)
 */
export function calculateVWAP(candles: OHLCV[]): IndicatorPoint[] {
    const result: IndicatorPoint[] = [];

    let cumulativeTPV = 0;  // Cumulative (TP * Volume)
    let cumulativeVolume = 0;
    let lastSessionDate = '';

    for (const candle of candles) {
        // Get session date for reset detection
        const candleDate = typeof candle.time === 'number'
            ? new Date(candle.time * 1000).toDateString()
            : new Date(candle.time).toDateString();

        // Reset at new session
        if (candleDate !== lastSessionDate) {
            cumulativeTPV = 0;
            cumulativeVolume = 0;
            lastSessionDate = candleDate;
        }

        const typicalPrice = (candle.high + candle.low + candle.close) / 3;
        const volume = candle.volume || 1;

        cumulativeTPV += typicalPrice * volume;
        cumulativeVolume += volume;

        const vwap = cumulativeVolume > 0 ? cumulativeTPV / cumulativeVolume : typicalPrice;
        result.push({ time: candle.time, value: vwap });
    }

    return result;
}

// =============================================================================
// ATR (Average True Range) + Bands
// =============================================================================

/**
 * Calculate ATR
 */
export function calculateATR(candles: OHLCV[], period: number = 14): IndicatorPoint[] {
    if (candles.length < 2) return [];

    const trueRanges: number[] = [];

    // Calculate True Range for each candle
    for (let i = 1; i < candles.length; i++) {
        const high = candles[i].high;
        const low = candles[i].low;
        const prevClose = candles[i - 1].close;

        const tr = Math.max(
            high - low,
            Math.abs(high - prevClose),
            Math.abs(low - prevClose)
        );
        trueRanges.push(tr);
    }

    if (trueRanges.length < period) return [];

    const result: IndicatorPoint[] = [];

    // Initial ATR (SMA of first n TRs)
    let sum = 0;
    for (let i = 0; i < period; i++) {
        sum += trueRanges[i];
    }
    let atr = sum / period;

    result.push({ time: candles[period].time, value: atr });

    // Smoothed ATR for remaining
    for (let i = period; i < trueRanges.length; i++) {
        atr = (atr * (period - 1) + trueRanges[i]) / period;
        result.push({ time: candles[i + 1].time, value: atr });
    }

    return result;
}

/**
 * Calculate ATR Bands (price ± ATR multiple)
 */
export function calculateATRBands(candles: OHLCV[], period: number = 14, multiple: number = 2): BandPoint[] {
    const atr = calculateATR(candles, period);
    const result: BandPoint[] = [];

    // Map ATR to bands using close price as center
    for (const point of atr) {
        const candle = candles.find(c => c.time === point.time);
        if (candle) {
            result.push({
                time: point.time,
                upper: candle.close + point.value * multiple,
                lower: candle.close - point.value * multiple,
                middle: candle.close
            });
        }
    }

    return result;
}

// =============================================================================
// Bollinger Bands
// =============================================================================

/**
 * Calculate Bollinger Bands (SMA ± std dev)
 */
export function calculateBollingerBands(candles: OHLCV[], period: number = 20, stdDev: number = 2): BandPoint[] {
    if (candles.length < period) return [];

    const result: BandPoint[] = [];

    for (let i = period - 1; i < candles.length; i++) {
        // Calculate SMA
        let sum = 0;
        for (let j = i - period + 1; j <= i; j++) {
            sum += candles[j].close;
        }
        const sma = sum / period;

        // Calculate Standard Deviation
        let sqSum = 0;
        for (let j = i - period + 1; j <= i; j++) {
            sqSum += Math.pow(candles[j].close - sma, 2);
        }
        const std = Math.sqrt(sqSum / period);

        result.push({
            time: candles[i].time,
            upper: sma + stdDev * std,
            lower: sma - stdDev * std,
            middle: sma
        });
    }

    return result;
}

// =============================================================================
// Donchian Channels
// =============================================================================

/**
 * Calculate Donchian Channels (highest high, lowest low over period)
 */
export function calculateDonchianChannels(candles: OHLCV[], period: number = 20): BandPoint[] {
    if (candles.length < period) return [];

    const result: BandPoint[] = [];

    for (let i = period - 1; i < candles.length; i++) {
        let highestHigh = -Infinity;
        let lowestLow = Infinity;

        for (let j = i - period + 1; j <= i; j++) {
            highestHigh = Math.max(highestHigh, candles[j].high);
            lowestLow = Math.min(lowestLow, candles[j].low);
        }

        result.push({
            time: candles[i].time,
            upper: highestHigh,
            lower: lowestLow,
            middle: (highestHigh + lowestLow) / 2
        });
    }

    return result;
}

// =============================================================================
// SMA (Simple Moving Average)
// =============================================================================

/**
 * Calculate SMA for a series of candles
 */
export function calculateSMA(candles: OHLCV[], period: number): IndicatorPoint[] {
    if (candles.length < period) return [];

    const result: IndicatorPoint[] = [];

    for (let i = period - 1; i < candles.length; i++) {
        let sum = 0;
        for (let j = i - period + 1; j <= i; j++) {
            sum += candles[j].close;
        }
        result.push({ time: candles[i].time, value: sum / period });
    }

    return result;
}

// =============================================================================
// ADR (Average Daily Range)
// =============================================================================

export interface AdrZones {
    time: number | string;
    resTop: number;      // Resistance zone top
    resBottom: number;   // Resistance zone bottom  
    supTop: number;      // Support zone top (same as resBottom for middle)
    supBottom: number;   // Support zone bottom
    sessionOpen: number; // Session open price (midpoint)
}

/**
 * Calculate ADR zones - outputs ADR levels for EVERY candle so lines render horizontally.
 * 
 * ADR zones are based on average daily range projected from session open.
 * - Resistance zone: sessionOpen + halfRange (14-period) to sessionOpen + halfRange * 0.5 (7-period)
 * - Support zone: sessionOpen - halfRange * 0.5 to sessionOpen - halfRange
 * 
 * @param candles - OHLCV data (should be at your display timeframe)
 * @param period - Lookback period for ADR calculation (default 14)
 */
export function calculateADR(candles: OHLCV[], period: number = 14): AdrZones[] {
    if (candles.length < period + 1) return [];

    // Step 1: Group candles by day and calculate daily high/low/open
    const dailyData: Map<string, { high: number; low: number; open: number; candles: OHLCV[] }> = new Map();

    for (const candle of candles) {
        const date = typeof candle.time === 'number'
            ? new Date(candle.time * 1000).toDateString()
            : new Date(candle.time).toDateString();

        const existing = dailyData.get(date);
        if (existing) {
            existing.high = Math.max(existing.high, candle.high);
            existing.low = Math.min(existing.low, candle.low);
            existing.candles.push(candle);
        } else {
            dailyData.set(date, {
                high: candle.high,
                low: candle.low,
                open: candle.open,
                candles: [candle]
            });
        }
    }

    const days = Array.from(dailyData.values());
    if (days.length < period + 1) return [];

    // Step 2: Calculate ADR for each candle
    const result: AdrZones[] = [];
    let dayIndex = 0;

    for (const [dateStr, dayData] of dailyData) {
        dayIndex++;

        // Need at least 'period' previous days to calculate ADR
        if (dayIndex <= period) continue;

        // Calculate average range from previous 'period' days
        let sumRange = 0;
        let count = 0;
        const prevDays = Array.from(dailyData.values()).slice(dayIndex - period - 1, dayIndex - 1);

        for (const prevDay of prevDays) {
            sumRange += prevDay.high - prevDay.low;
            count++;
        }

        if (count < period) continue;

        const avgRange = sumRange / period;
        const halfRange = avgRange / 2;
        const sessionOpen = dayData.open;

        // Output ADR levels for EVERY candle in this day (creates horizontal lines)
        for (const candle of dayData.candles) {
            result.push({
                time: candle.time,
                resTop: sessionOpen + halfRange,          // Red zone top (14-period)
                resBottom: sessionOpen + halfRange * 0.5, // Red zone bottom (7-period approx)
                supTop: sessionOpen - halfRange * 0.5,    // Green zone top
                supBottom: sessionOpen - halfRange,       // Green zone bottom (14-period)
                sessionOpen,
            });
        }
    }

    return result;
}

// =============================================================================
// Custom Indicator Type
// =============================================================================

export interface CustomIndicator {
    id: string;
    type: 'ema' | 'sma';
    period: number;
    color: string;
}

// =============================================================================
// Indicator Settings Type
// =============================================================================

export interface IndicatorSettings {
    ema9: boolean;
    ema21: boolean;
    ema200: boolean;
    vwap: boolean;
    atrBands: boolean;
    bollingerBands: boolean;
    donchianChannels: boolean;
    adr: boolean;
    customIndicators?: CustomIndicator[];
}

export const DEFAULT_INDICATOR_SETTINGS: IndicatorSettings = {
    ema9: false,
    ema21: false,
    ema200: false,
    vwap: false,
    atrBands: false,
    bollingerBands: false,
    donchianChannels: false,
    adr: false,
    customIndicators: [],
};

```

### src/features/context.py

```python
"""
Context Features
Derived context vector (x_context) for MLP input.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any

from src.features.state import MarketState
from src.features.indicators import IndicatorValues
from src.features.levels import LevelValues
from src.features.time_features import TimeFeatures


@dataclass
class ContextFeatures:
    """
    Context feature vector for MLP input.
    
    These are scalar/low-dim features derived from indicators and state,
    separate from the raw OHLCV windows used by CNN.
    """
    # EMA distances (normalized by ATR)
    dist_ema_5m_200_atr: float = 0.0
    dist_ema_15m_200_atr: float = 0.0
    
    # VWAP distances
    dist_vwap_session_atr: float = 0.0
    dist_vwap_weekly_atr: float = 0.0
    
    # Level distances
    dist_nearest_1h_level_atr: float = 0.0
    dist_nearest_4h_level_atr: float = 0.0
    dist_pdh_atr: float = 0.0
    dist_pdl_atr: float = 0.0
    
    # Volatility
    adr_pct_used: float = 0.0
    
    # Momentum
    rsi_5m_14: float = 50.0
    rsi_15m_14: float = 50.0
    
    # Volume
    relative_volume: float = 1.0
    
    # Time (cyclical)
    hour_sin: float = 0.0
    hour_cos: float = 0.0
    dow_sin: float = 0.0
    dow_cos: float = 0.0
    
    # Time (flags)
    is_rth: float = 0.0
    is_first_hour: float = 0.0
    is_last_hour: float = 0.0
    mins_into_session: float = 0.0
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        return np.array([
            self.dist_ema_5m_200_atr,
            self.dist_ema_15m_200_atr,
            self.dist_vwap_session_atr,
            self.dist_vwap_weekly_atr,
            self.dist_nearest_1h_level_atr,
            self.dist_nearest_4h_level_atr,
            self.dist_pdh_atr,
            self.dist_pdl_atr,
            self.adr_pct_used,
            self.rsi_5m_14 / 100.0,  # Normalize to [0, 1]
            self.rsi_15m_14 / 100.0,
            self.relative_volume,
            self.hour_sin,
            self.hour_cos,
            self.dow_sin,
            self.dow_cos,
            self.is_rth,
            self.is_first_hour,
            self.is_last_hour,
            self.mins_into_session / 390.0,  # Normalize by RTH length
        ], dtype=np.float32)
    
    @staticmethod
    def feature_names() -> List[str]:
        """Get ordered feature names."""
        return [
            'dist_ema_5m_200_atr',
            'dist_ema_15m_200_atr',
            'dist_vwap_session_atr',
            'dist_vwap_weekly_atr',
            'dist_nearest_1h_level_atr',
            'dist_nearest_4h_level_atr',
            'dist_pdh_atr',
            'dist_pdl_atr',
            'adr_pct_used',
            'rsi_5m_14_norm',
            'rsi_15m_14_norm',
            'relative_volume',
            'hour_sin',
            'hour_cos',
            'dow_sin',
            'dow_cos',
            'is_rth',
            'is_first_hour',
            'is_last_hour',
            'mins_into_session_norm',
        ]
    
    @staticmethod
    def dim() -> int:
        """Get feature dimension."""
        return 20


def compute_context_features(
    current_price: float,
    indicators: IndicatorValues,
    levels: LevelValues,
    time_features: TimeFeatures,
    atr: float = 1.0
) -> ContextFeatures:
    """
    Compute context feature vector from component features.
    
    All distances are normalized by ATR for scale-independence.
    """
    if atr <= 0:
        atr = 1.0
    
    # EMA distances
    dist_ema_5m = (current_price - indicators.ema_5m_200) / atr if indicators.ema_5m_200 else 0
    dist_ema_15m = (current_price - indicators.ema_15m_200) / atr if indicators.ema_15m_200 else 0
    
    # VWAP distances
    dist_vwap_session = (current_price - indicators.vwap_session) / atr if indicators.vwap_session else 0
    dist_vwap_weekly = (current_price - indicators.vwap_weekly) / atr if indicators.vwap_weekly else 0
    
    # Level distances (use nearest, sign indicates above/below)
    dist_1h = min(abs(levels.dist_1h_high), abs(levels.dist_1h_low)) / atr
    if levels.dist_1h_high < levels.dist_1h_low:
        dist_1h = -dist_1h  # Closer to resistance (above)
    
    dist_4h = min(abs(levels.dist_4h_high), abs(levels.dist_4h_low)) / atr
    if levels.dist_4h_high < levels.dist_4h_low:
        dist_4h = -dist_4h
    
    return ContextFeatures(
        dist_ema_5m_200_atr=dist_ema_5m,
        dist_ema_15m_200_atr=dist_ema_15m,
        dist_vwap_session_atr=dist_vwap_session,
        dist_vwap_weekly_atr=dist_vwap_weekly,
        dist_nearest_1h_level_atr=dist_1h,
        dist_nearest_4h_level_atr=dist_4h,
        dist_pdh_atr=levels.dist_pdh / atr if levels.pdh else 0,
        dist_pdl_atr=levels.dist_pdl / atr if levels.pdl else 0,
        adr_pct_used=indicators.adr_pct_used,
        rsi_5m_14=indicators.rsi_5m_14,
        rsi_15m_14=indicators.rsi_15m_14,
        relative_volume=indicators.relative_volume,
        hour_sin=time_features.hour_sin,
        hour_cos=time_features.hour_cos,
        dow_sin=time_features.dow_sin,
        dow_cos=time_features.dow_cos,
        is_rth=float(time_features.is_rth),
        is_first_hour=float(time_features.is_first_hour),
        is_last_hour=float(time_features.is_last_hour),
        mins_into_session=float(time_features.mins_into_session),
    )

```

### src/features/engine.py

```python
"""
Unified Feature Engine

Single source of truth for feature computation.
Ensures TRAIN, SCAN, REPLAY, and INFER all use identical normalization.

This prevents "inference skew" - where training and production drift apart.
"""

import numpy as np
from typing import List, Dict, Any, Union
from dataclasses import dataclass


@dataclass
class FeatureConfig:
    """Configuration for feature computation."""
    lookback: int = 30
    normalization: str = "percent_change"  # "percent_change", "zscore", "minmax"
    volume_norm: str = "max"  # "max", "mean", "none"


def normalize_ohlcv_window(
    ohlcv_data: Union[np.ndarray, List[Dict[str, float]]],
    config: FeatureConfig = None
) -> np.ndarray:
    """
    Normalize OHLCV window for model input.
    
    THIS IS THE SINGLE SOURCE OF TRUTH FOR NORMALIZATION.
    Used by:
    - Training pipeline (train_ifvg_4class.py)
    - Inference endpoint (/infer)
    - Backend simulation (test_backend_simulation.py)
    - Frontend (via /infer API)
    
    Args:
        ohlcv_data: Either:
            - np.ndarray of shape (N, 5) [open, high, low, close, volume]
            - List of dicts [{open, high, low, close, volume}, ...]
        config: FeatureConfig (uses defaults if None)
    
    Returns:
        np.ndarray of shape (5, N) normalized for model input
    """
    if config is None:
        config = FeatureConfig()
    
    # Convert list of dicts to numpy array
    if isinstance(ohlcv_data, list):
        ohlcv_array = np.array([
            [b['open'], b['high'], b['low'], b['close'], b.get('volume', 0)]
            for b in ohlcv_data
        ], dtype=np.float32)
    else:
        ohlcv_array = np.asarray(ohlcv_data, dtype=np.float32)
    
    # Ensure correct shape (N, 5)
    if ohlcv_array.ndim == 1:
        ohlcv_array = ohlcv_array.reshape(-1, 5)
    
    # Transpose to (5, N) for model input
    x = ohlcv_array.T.copy()
    
    # Apply normalization to OHLC (indices 0-3)
    if config.normalization == "percent_change":
        # Normalize by first bar's close (percent change)
        first_close = x[3, 0]
        if first_close > 0:
            x[0:4] = (x[0:4] - first_close) / first_close * 100
    
    elif config.normalization == "zscore":
        # Z-score normalization
        for i in range(4):
            mean = x[i].mean()
            std = x[i].std()
            if std > 0:
                x[i] = (x[i] - mean) / std
    
    elif config.normalization == "minmax":
        # Min-max normalization to [0, 1]
        for i in range(4):
            min_val = x[i].min()
            max_val = x[i].max()
            if max_val > min_val:
                x[i] = (x[i] - min_val) / (max_val - min_val)
    
    # Apply volume normalization
    if config.volume_norm == "max":
        max_vol = x[4].max()
        if max_vol > 0:
            x[4] = x[4] / max_vol
        else:
            x[4] = 0
    elif config.volume_norm == "mean":
        mean_vol = x[4].mean()
        if mean_vol > 0:
            x[4] = x[4] / mean_vol
    elif config.volume_norm == "none":
        pass  # Keep raw volume
    
    return x


def compute_atr(bars: Union[np.ndarray, List[Dict[str, float]]], period: int = 14) -> float:
    """
    Compute Average True Range.
    
    Args:
        bars: OHLCV data (N, 5) or list of dicts
        period: ATR period
    
    Returns:
        ATR value
    """
    if isinstance(bars, list):
        highs = np.array([b['high'] for b in bars])
        lows = np.array([b['low'] for b in bars])
        closes = np.array([b['close'] for b in bars])
    else:
        highs = bars[:, 1]
        lows = bars[:, 2]
        closes = bars[:, 3]
    
    if len(bars) < 2:
        return highs[0] - lows[0] if len(bars) > 0 else 1.0
    
    # True Range = max(high-low, |high-prev_close|, |low-prev_close|)
    tr = np.zeros(len(bars))
    tr[0] = highs[0] - lows[0]
    
    for i in range(1, len(bars)):
        hl = highs[i] - lows[i]
        hpc = abs(highs[i] - closes[i-1])
        lpc = abs(lows[i] - closes[i-1])
        tr[i] = max(hl, hpc, lpc)
    
    # Use last 'period' bars for ATR
    atr = tr[-period:].mean() if len(tr) >= period else tr.mean()
    return float(atr)


def bars_to_model_input(
    bars: Union[np.ndarray, List[Dict[str, float]]],
    lookback: int = 30,
    config: FeatureConfig = None
) -> np.ndarray:
    """
    Convert bars to model input tensor.
    
    Takes last 'lookback' bars and normalizes them.
    
    Args:
        bars: OHLCV data
        lookback: Number of bars to use
        config: Feature configuration
    
    Returns:
        np.ndarray of shape (5, lookback) ready for model
    """
    if config is None:
        config = FeatureConfig(lookback=lookback)
    
    # Convert to numpy if needed
    if isinstance(bars, list):
        bars = np.array([
            [b['open'], b['high'], b['low'], b['close'], b.get('volume', 0)]
            for b in bars
        ], dtype=np.float32)
    
    # Take last N bars
    if len(bars) > lookback:
        bars = bars[-lookback:]
    elif len(bars) < lookback:
        # Pad with first bar if too few
        padding = np.tile(bars[0:1], (lookback - len(bars), 1))
        bars = np.vstack([padding, bars])
    
    return normalize_ohlcv_window(bars, config)

```

### src/features/fvg.py

```python
"""
Fair Value Gap (FVG) Detection

Identifies imbalances in price action on specified timeframes.
FVGs represent areas where price moved quickly, leaving a gap between candle wicks.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class FairValueGap:
    """Represents a Fair Value Gap in price."""
    high: float           # Top of the gap
    low: float            # Bottom of the gap
    midpoint: float       # 50% level for entry
    direction: str        # "BULLISH" or "BEARISH"
    bar_idx: int          # Bar index where it formed (the impulse candle)
    bar_time: pd.Timestamp  # Time of impulse candle
    gap_size: float       # Size of the gap in points
    
    def contains_price(self, price: float, min_pct: float = 0.5) -> bool:
        """
        Check if price has entered the FVG by at least min_pct.
        
        Args:
            price: Current price
            min_pct: Minimum percentage into FVG required (0.5 = 50%)
            
        Returns:
            True if price is at least min_pct into the FVG
        """
        if self.direction == "BULLISH":
            # For bullish FVG, price retracing down into gap
            if price > self.high:
                return False  # Above the gap
            if price <= self.low:
                return True   # Fully through gap
            # Calculate penetration percentage
            pct_into = (self.high - price) / self.gap_size if self.gap_size > 0 else 0
            return pct_into >= min_pct
        else:
            # For bearish FVG, price retracing up into gap
            if price < self.low:
                return False  # Below the gap
            if price >= self.high:
                return True   # Fully through gap
            pct_into = (price - self.low) / self.gap_size if self.gap_size > 0 else 0
            return pct_into >= min_pct


def find_fvg(
    df: pd.DataFrame,
    lookback: int = 20,
    min_gap_atr: float = 0.2,
    atr: float = 5.0
) -> List[FairValueGap]:
    """
    Find Fair Value Gaps in price data.
    
    FVG Definition:
    - Bullish FVG: Gap between candle[i-1].high and candle[i+1].low
      (price gapped up through the middle candle)
    - Bearish FVG: Gap between candle[i-1].low and candle[i+1].high
      (price gapped down through the middle candle)
    
    Args:
        df: OHLCV DataFrame with 'high', 'low', 'time' columns
        lookback: Number of bars to look back for FVGs
        min_gap_atr: Minimum gap size as ATR multiple
        atr: Current ATR for filtering
        
    Returns:
        List of FairValueGap objects, most recent first
    """
    if df is None or len(df) < 3:
        return []
    
    recent = df.tail(lookback + 2).copy()
    if len(recent) < 3:
        return []
    
    fvgs = []
    min_gap = min_gap_atr * atr
    
    # We need at least 3 candles to detect an FVG
    # The FVG forms between candle i-1 and i+1, with i being the impulse
    for i in range(1, len(recent) - 1):
        prev_bar = recent.iloc[i - 1]
        impulse_bar = recent.iloc[i]
        next_bar = recent.iloc[i + 1]
        
        # Get the original index for bar_idx
        bar_idx = recent.index[i]
        bar_time = impulse_bar.get('time', pd.Timestamp.now())
        
        # Check for BULLISH FVG (gap up)
        # Gap exists if next_bar.low > prev_bar.high
        bullish_gap = next_bar['low'] - prev_bar['high']
        if bullish_gap > min_gap:
            fvg = FairValueGap(
                high=next_bar['low'],      # Top of gap
                low=prev_bar['high'],       # Bottom of gap
                midpoint=(next_bar['low'] + prev_bar['high']) / 2,
                direction="BULLISH",
                bar_idx=bar_idx,
                bar_time=pd.Timestamp(bar_time) if not isinstance(bar_time, pd.Timestamp) else bar_time,
                gap_size=bullish_gap
            )
            fvgs.append(fvg)
        
        # Check for BEARISH FVG (gap down)
        # Gap exists if prev_bar.low > next_bar.high
        bearish_gap = prev_bar['low'] - next_bar['high']
        if bearish_gap > min_gap:
            fvg = FairValueGap(
                high=prev_bar['low'],       # Top of gap
                low=next_bar['high'],       # Bottom of gap
                midpoint=(prev_bar['low'] + next_bar['high']) / 2,
                direction="BEARISH",
                bar_idx=bar_idx,
                bar_time=pd.Timestamp(bar_time) if not isinstance(bar_time, pd.Timestamp) else bar_time,
                gap_size=bearish_gap
            )
            fvgs.append(fvg)
    
    # Return most recent first
    return list(reversed(fvgs))


def find_most_recent_fvg(
    df: pd.DataFrame,
    direction: str,
    lookback: int = 20,
    min_gap_atr: float = 0.2,
    atr: float = 5.0
) -> Optional[FairValueGap]:
    """
    Find the most recent FVG in the specified direction.
    
    Args:
        df: OHLCV DataFrame
        direction: "BULLISH" or "BEARISH"
        lookback: Bars to look back
        min_gap_atr: Minimum gap size
        atr: Current ATR
        
    Returns:
        Most recent FVG matching direction, or None
    """
    fvgs = find_fvg(df, lookback, min_gap_atr, atr)
    
    for fvg in fvgs:
        if fvg.direction == direction.upper():
            return fvg
    
    return None


def is_impulse_candle(
    candle: pd.Series,
    direction: str,
    min_body_pct: float = 0.6
) -> bool:
    """
    Check if a candle is an impulse candle (strong directional move).
    
    An impulse candle has:
    - Body at least min_body_pct of the total range
    - Close in the direction of the move
    
    Args:
        candle: Single candle row with OHLC
        direction: Expected direction ("BULLISH" or "BEARISH")
        min_body_pct: Minimum body percentage of range
        
    Returns:
        True if this is an impulse candle
    """
    body = abs(candle['close'] - candle['open'])
    full_range = candle['high'] - candle['low']
    
    if full_range <= 0:
        return False
    
    body_pct = body / full_range
    
    if body_pct < min_body_pct:
        return False
    
    if direction == "BULLISH":
        return candle['close'] > candle['open']
    else:
        return candle['close'] < candle['open']


def find_impulse_with_fvg(
    df_5m: pd.DataFrame,
    direction: str,
    lookback: int = 10,
    atr: float = 5.0
) -> Optional[tuple]:
    """
    Find an impulse candle that created an FVG.
    
    Used after a level break to identify structure change.
    
    Args:
        df_5m: 5-minute OHLCV data
        direction: Expected impulse direction
        lookback: Bars to look back
        atr: Current ATR
        
    Returns:
        Tuple of (impulse_bar_idx, FairValueGap) or None
    """
    if df_5m is None or len(df_5m) < 3:
        return None
    
    recent = df_5m.tail(lookback + 2).copy()
    
    for i in range(1, len(recent) - 1):
        impulse_bar = recent.iloc[i]
        
        if not is_impulse_candle(impulse_bar, direction):
            continue
        
        prev_bar = recent.iloc[i - 1]
        next_bar = recent.iloc[i + 1]
        
        # Check for FVG in the expected direction
        if direction == "BULLISH":
            gap = next_bar['low'] - prev_bar['high']
            if gap > 0.2 * atr:
                fvg = FairValueGap(
                    high=next_bar['low'],
                    low=prev_bar['high'],
                    midpoint=(next_bar['low'] + prev_bar['high']) / 2,
                    direction="BULLISH",
                    bar_idx=recent.index[i],
                    bar_time=pd.Timestamp(impulse_bar.get('time', pd.Timestamp.now())),
                    gap_size=gap
                )
                return (recent.index[i], fvg)
        else:
            gap = prev_bar['low'] - next_bar['high']
            if gap > 0.2 * atr:
                fvg = FairValueGap(
                    high=prev_bar['low'],
                    low=next_bar['high'],
                    midpoint=(prev_bar['low'] + next_bar['high']) / 2,
                    direction="BEARISH",
                    bar_idx=recent.index[i],
                    bar_time=pd.Timestamp(impulse_bar.get('time', pd.Timestamp.now())),
                    gap_size=gap
                )
                return (recent.index[i], fvg)
    
    return None

```

### src/features/index.ts

```typescript
/**
 * Features Index - Unified Export for Strategy Use
 * 
 * Import all indicator calculations from this single location:
 * 
 * ```typescript
 * import { calculateEMA, calculateADR, calculateBollingerBands } from '@/features';
 * ```
 */

// Chart Indicators (TypeScript - for frontend)
export * from './chart_indicators';

```

### src/features/indicator_registry_init.py

```python
"""
Indicator Registration  
Wire existing indicators into the IndicatorRegistry.
"""

from src.core.registries import IndicatorRegistry, IndicatorSeries
import pandas as pd
from typing import Any


# =============================================================================
# Register built-in indicators
# =============================================================================

@IndicatorRegistry.register(
    indicator_id="ema",
    name="Exponential Moving Average",
    output_type="line",
    description="EMA of closing prices",
    params_schema={
        "period": {"type": "integer", "default": 20, "min": 2}
    }
)
class EMAIndicator:
    """EMA indicator."""
    def __init__(self, period: int = 20):
        self.period = period
    
    def compute(self, stepper: Any) -> IndicatorSeries:
        """Compute EMA from stepper data."""
        # Extract price data
        df = stepper.df if hasattr(stepper, 'df') else pd.DataFrame()
        
        if len(df) == 0:
            return IndicatorSeries(
                indicator_id=f"ema_{self.period}",
                name=f"EMA {self.period}",
                type="line",
                points=[]
            )
        
        # Calculate EMA
        ema = df['close'].ewm(span=self.period, adjust=False).mean()
        
        # Convert to points
        points = [
            {
                'time': row['time'].isoformat() if hasattr(row['time'], 'isoformat') else str(row['time']),
                'value': float(val) if pd.notna(val) else None
            }
            for (idx, row), val in zip(df.iterrows(), ema)
        ]
        
        return IndicatorSeries(
            indicator_id=f"ema_{self.period}",
            name=f"EMA {self.period}",
            type="line",
            points=points,
            style={'color': '#00ff00', 'lineWidth': 2}
        )


@IndicatorRegistry.register(
    indicator_id="atr",
    name="Average True Range",
    output_type="line",
    description="ATR volatility indicator",
    params_schema={
        "period": {"type": "integer", "default": 14, "min": 2}
    }
)
class ATRIndicator:
    """ATR indicator."""
    def __init__(self, period: int = 14):
        self.period = period
    
    def compute(self, stepper: Any) -> IndicatorSeries:
        """Compute ATR from stepper data."""
        df = stepper.df if hasattr(stepper, 'df') else pd.DataFrame()
        
        if len(df) == 0:
            return IndicatorSeries(
                indicator_id=f"atr_{self.period}",
                name=f"ATR {self.period}",
                type="line",
                points=[]
            )
        
        # Calculate True Range
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.ewm(span=self.period, adjust=False).mean()
        
        # Convert to points
        points = [
            {
                'time': row['time'].isoformat() if hasattr(row['time'], 'isoformat') else str(row['time']),
                'value': float(val) if pd.notna(val) else None
            }
            for (idx, row), val in zip(df.iterrows(), atr)
        ]
        
        return IndicatorSeries(
            indicator_id=f"atr_{self.period}",
            name=f"ATR {self.period}",
            type="line",
            points=points,
            style={'color': '#ff9900', 'lineWidth': 1}
        )


@IndicatorRegistry.register(
    indicator_id="vwap",
    name="Volume Weighted Average Price",
    output_type="line",
    description="VWAP - resets daily",
    params_schema={}
)
class VWAPIndicator:
    """VWAP indicator."""
    def __init__(self):
        pass
    
    def compute(self, stepper: Any) -> IndicatorSeries:
        """Compute VWAP from stepper data."""
        df = stepper.df if hasattr(stepper, 'df') else pd.DataFrame()
        
        if len(df) == 0:
            return IndicatorSeries(
                indicator_id="vwap",
                name="VWAP",
                type="line",
                points=[]
            )
        
        # Simple VWAP (not session-aware for now)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        
        # Convert to points
        points = [
            {
                'time': row['time'].isoformat() if hasattr(row['time'], 'isoformat') else str(row['time']),
                'value': float(val) if pd.notna(val) else None
            }
            for (idx, row), val in zip(df.iterrows(), vwap)
        ]
        
        return IndicatorSeries(
            indicator_id="vwap",
            name="VWAP",
            type="line",
            points=points,
            style={'color': '#ffff00', 'lineWidth': 2}
        )


# Auto-register on import
def register_all_indicators():
    """
    Register all available indicators.
    Call this at startup to populate the registry.
    """
    pass

```

### src/features/indicators.py

```python
"""
Technical Indicators
EMA, RSI, ATR, ADR, VWAP calculations.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional
from zoneinfo import ZoneInfo

from src.config import (
    DEFAULT_EMA_PERIOD, DEFAULT_RSI_PERIOD, 
    DEFAULT_ATR_PERIOD, DEFAULT_ADR_PERIOD,
    NY_TZ, SESSION_RTH_START
)


# =============================================================================
# EMA
# =============================================================================

def calculate_ema(series: pd.Series, period: int = DEFAULT_EMA_PERIOD) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def add_ema(df: pd.DataFrame, period: int = DEFAULT_EMA_PERIOD, col: str = 'close') -> pd.Series:
    """Add EMA column to dataframe."""
    return calculate_ema(df[col], period)


# =============================================================================
# RSI
# =============================================================================

def calculate_rsi(series: pd.Series, period: int = DEFAULT_RSI_PERIOD) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = series.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.fillna(50)  # Neutral when undefined


# =============================================================================
# ATR
# =============================================================================

def calculate_atr(df: pd.DataFrame, period: int = DEFAULT_ATR_PERIOD) -> pd.Series:
    """
    Calculate Average True Range.
    
    Uses shifted ATR (value at T uses data up to T-1) to prevent look-ahead.
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    # Shift by 1 to make it causal
    return atr.shift(1)


# =============================================================================
# ADR (Average Daily Range)
# =============================================================================

def calculate_adr(
    df: pd.DataFrame, 
    period: int = DEFAULT_ADR_PERIOD,
    tz: ZoneInfo = NY_TZ
) -> pd.Series:
    """
    Calculate Average Daily Range.
    
    Returns ADR value aligned to each bar.
    """
    # Ensure we have a time column
    if 'time' in df.columns:
        df = df.copy()
        df['date'] = pd.to_datetime(df['time']).dt.date
    elif df.index.name == 'time':
        df = df.copy()
        df['date'] = df.index.date
    else:
        raise ValueError("DataFrame must have 'time' column or datetime index")
    
    # Calculate daily range
    daily = df.groupby('date').agg({
        'high': 'max',
        'low': 'min'
    })
    daily['daily_range'] = daily['high'] - daily['low']
    
    # Rolling average
    daily['adr'] = daily['daily_range'].rolling(window=period).mean().shift(1)
    
    # Map back to each bar
    df['adr'] = df['date'].map(daily['adr'])
    
    return df['adr']


def get_adr_percent_used(
    current_price: float,
    daily_open: float,
    adr: float
) -> float:
    """
    Calculate how much of ADR has been consumed.
    
    Returns value in [0, 1+] where 1.0 = full ADR used.
    """
    if adr <= 0:
        return 0.0
    movement = abs(current_price - daily_open)
    return movement / adr


# =============================================================================
# VWAP
# =============================================================================

def calculate_vwap(
    df: pd.DataFrame,
    period: str = 'session',  # 'session', 'weekly', 'daily'
    tz: ZoneInfo = NY_TZ,
    session_start: str = SESSION_RTH_START
) -> pd.Series:
    """
    Calculate Volume-Weighted Average Price.
    
    Args:
        df: DataFrame with time, high, low, close, volume
        period: 'session', 'weekly', or 'daily'
        tz: Timezone for period boundaries
        session_start: Session start time (for session VWAP)
        
    Returns:
        VWAP series aligned to each bar.
    """
    df = df.copy()
    
    # Ensure we have time
    if 'time' not in df.columns:
        raise ValueError("DataFrame must have 'time' column")
    
    # Convert to target timezone
    df['time_tz'] = pd.to_datetime(df['time']).dt.tz_convert(tz)
    
    # Typical price
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['tp_vol'] = df['typical_price'] * df['volume']
    
    # Determine period grouping
    if period == 'session':
        # Group by session (resets at session_start)
        hour, minute = map(int, session_start.split(':'))
        df['session_date'] = df['time_tz'].apply(
            lambda t: t.date() if t.hour >= hour else (t - pd.Timedelta(days=1)).date()
        )
        group_col = 'session_date'
        
    elif period == 'weekly':
        df['week'] = df['time_tz'].dt.isocalendar().week
        df['year'] = df['time_tz'].dt.year
        df['year_week'] = df['year'].astype(str) + '_' + df['week'].astype(str)
        group_col = 'year_week'
        
    else:  # daily
        df['date'] = df['time_tz'].dt.date
        group_col = 'date'
    
    # Cumulative VWAP within each period
    df['cum_tp_vol'] = df.groupby(group_col)['tp_vol'].cumsum()
    df['cum_vol'] = df.groupby(group_col)['volume'].cumsum()
    
    vwap = df['cum_tp_vol'] / df['cum_vol'].replace(0, np.nan)
    
    return vwap.fillna(method='ffill')


# =============================================================================
# Settlement Price
# =============================================================================

def calculate_settlement(
    df: pd.DataFrame,
    settlement_time: str = "15:00",  # 3 PM
    tz: ZoneInfo = NY_TZ
) -> pd.Series:
    """
    Calculate settlement price (typically 3 PM close).
    
    Args:
        df: DataFrame with close and time
        settlement_time: Time of settlement (HH:MM format)
    
    Returns:
        Series with settlement values (forward-filled)
    """
    df = df.copy()
    
    if 'time' not in df.columns:
        raise ValueError("DataFrame must have 'time' column")
    
    df['time_tz'] = pd.to_datetime(df['time']).dt.tz_convert(tz)
    hour, minute = map(int, settlement_time.split(':'))
    settlement_time_obj = df['time_tz'].iloc[0].replace(hour=hour, minute=minute).time()
    
    settlement = pd.Series(np.nan, index=df.index)
    current_settlement = None
    prev_hour = None
    
    for i in range(len(df)):
        t = df['time_tz'].iloc[i]
        
        # Check if crossed settlement time
        if prev_hour is not None:
            if prev_hour < hour <= t.hour or (prev_hour >= hour and t.hour >= hour and t.minute >= minute):
                current_settlement = df['close'].iloc[i]
        
        if current_settlement is not None:
            settlement.iloc[i] = current_settlement
        
        prev_hour = t.hour
    
    return settlement.ffill()


# =============================================================================
# Session Levels (PDH, PDL, PDC)
# =============================================================================

def calculate_session_levels(
    df: pd.DataFrame,
    tz: ZoneInfo = NY_TZ
) -> pd.DataFrame:
    """
    Calculate Previous Day High, Low, Close.
    
    Args:
        df: DataFrame with OHLC and time
    
    Returns:
        DataFrame with columns: pdh, pdl, pdc (Previous Day High/Low/Close)
    """
    df = df.copy()
    
    if 'time' not in df.columns:
        raise ValueError("DataFrame must have 'time' column")
    
    df['date'] = pd.to_datetime(df['time']).dt.date
    
    # Calculate daily stats
    daily = df.groupby('date').agg({
        'high': 'max',
        'low': 'min', 
        'close': 'last'
    }).rename(columns={
        'high': 'pdh',
        'low': 'pdl',
        'close': 'pdc'
    })
    
    # Shift by 1 day (previous day's values)
    daily = daily.shift(1)
    
    # Map back to each bar
    levels = pd.DataFrame(index=df.index)
    levels['pdh'] = df['date'].map(daily['pdh'])
    levels['pdl'] = df['date'].map(daily['pdl'])
    levels['pdc'] = df['date'].map(daily['pdc'])
    
    return levels


# =============================================================================
# Indicator Bundle
# =============================================================================

@dataclass
class IndicatorValues:
    """Bundle of indicator values at a point in time."""
    ema_5m_20: float = 0.0
    ema_15m_20: float = 0.0
    ema_5m_200: float = 0.0
    ema_15m_200: float = 0.0
    rsi_5m_14: float = 50.0
    rsi_15m_14: float = 50.0
    atr_5m_14: float = 0.0
    atr_15m_14: float = 0.0
    adr_14: float = 0.0
    adr_pct_used: float = 0.0
    vwap_session: float = 0.0
    vwap_weekly: float = 0.0
    relative_volume: float = 1.0


def compute_indicators_at_bar(
    bar_idx: int,
    df_1m: pd.DataFrame,
    df_5m: pd.DataFrame,
    df_15m: pd.DataFrame,
    current_time: pd.Timestamp,
    current_price: float,
    daily_open: float = None
) -> IndicatorValues:
    """
    Compute all indicators at a specific bar.
    
    Note: For efficiency, indicators should be pre-computed and looked up.
    This function is for reference/testing.
    """
    # This is a simplified version - in practice, use FeatureStore
    # to cache these computations
    
    values = IndicatorValues()
    
    # Get indices for lookups
    # ... (implementation would look up pre-computed values)
    
    return values

```

### src/features/indicators_pro.py

```python
"""
Professional Trading Indicators Library

Comprehensive set of trading primitives for the MLang2 platform:
- Bar Measurement: Heikin-Ashi, range metrics
- Time Series: MACD, Stochastic, ADX, Ichimoku
- Volume: OBV, VWMACD, Chaikin Money Flow
- Levels: Pivot Points, Fibonacci
- Breakouts: Donchian channels, patterns
- Filters: Time-of-day, risk sizing

All functions are registry-compatible and return pandas Series or DataFrames.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass


# =============================================================================
# 1. BAR MEASUREMENT PRIMITIVES
# =============================================================================

def calculate_heikin_ashi(df: pd.DataFrame, smoothing: float = 1.0) -> pd.DataFrame:
    """
    Calculate Heikin-Ashi candles for trend clarity.
    
    Args:
        df: DataFrame with OHLC columns
        smoothing: Smoothing factor (1.0 = standard, >1.0 = more smoothing)
    
    Returns:
        DataFrame with ha_open, ha_high, ha_low, ha_close
    """
    ha = pd.DataFrame(index=df.index)
    
    # HA Close = (O + H + L + C) / 4
    ha['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4.0
    
    # HA Open = (prev HA Open + prev HA Close) / 2
    ha['ha_open'] = 0.0
    ha.iloc[0, ha.columns.get_loc('ha_open')] = (df.iloc[0]['open'] + df.iloc[0]['close']) / 2.0
    
    for i in range(1, len(df)):
        ha.iloc[i, ha.columns.get_loc('ha_open')] = (
            (ha.iloc[i-1]['ha_open'] + ha.iloc[i-1]['ha_close']) / 2.0
        )
    
    # Apply smoothing if needed
    if smoothing != 1.0:
        ha['ha_open'] = ha['ha_open'].ewm(alpha=1.0/smoothing).mean()
        ha['ha_close'] = ha['ha_close'].ewm(alpha=1.0/smoothing).mean()
    
    # HA High = max(H, HA Open, HA Close)
    ha['ha_high'] = df[['high']].join(ha[['ha_open', 'ha_close']]).max(axis=1)
    
    # HA Low = min(L, HA Open, HA Close)
    ha['ha_low'] = df[['low']].join(ha[['ha_open', 'ha_close']]).min(axis=1)
    
    return ha


def calculate_bar_expansion(df: pd.DataFrame, atr_period: int = 14, threshold: float = 1.5) -> pd.Series:
    """
    Detect bars with expansion above threshold × ATR.
    
    Args:
        df: DataFrame with high, low columns
        atr_period: Period for ATR calculation
        threshold: Multiplier for expansion detection (e.g., 1.5 = 150% of ATR)
    
    Returns:
        Boolean Series indicating expansion bars
    """
    from src.features.indicators import calculate_atr
    
    atr = calculate_atr(df, period=atr_period)
    bar_range = df['high'] - df['low']
    
    return bar_range > (threshold * atr)


def calculate_average_bar_size(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Calculate average bar size over N periods.
    
    Args:
        df: DataFrame with high, low columns
        period: Lookback period
    
    Returns:
        Series of average bar sizes
    """
    bar_range = df['high'] - df['low']
    return bar_range.rolling(window=period).mean()


# =============================================================================
# 2. TIME SERIES PRIMITIVES
# =============================================================================

def calculate_macd(
    close: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        close: Close price series
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line EMA period
    
    Returns:
        (macd_line, signal_line, histogram)
    """
    fast_ema = close.ewm(span=fast_period, adjust=False).mean()
    slow_ema = close.ewm(span=slow_period, adjust=False).mean()
    
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_stochastic(
    df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
    smoothing: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Stochastic Oscillator.
    
    Args:
        df: DataFrame with high, low, close
        k_period: %K period (lookback for high/low)
        d_period: %D period (smoothing of %K)
        smoothing: Additional smoothing for %K
    
    Returns:
        (%K line, %D line)
    """
    # Calculate %K
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    
    k = 100 * (df['close'] - low_min) / (high_max - low_min)
    
    # Smooth %K
    k_smooth = k.rolling(window=smoothing).mean()
    
    # Calculate %D (moving average of %K)
    d = k_smooth.rolling(window=d_period).mean()
    
    return k_smooth, d


def calculate_adx(
    df: pd.DataFrame,
    period: int = 14
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate ADX (Average Directional Index) for trend strength.
    
    Args:
        df: DataFrame with high, low, close
        period: Period for ADX calculation
    
    Returns:
        (adx, plus_di, minus_di)
    """
    # True Range
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = df['high'] - df['high'].shift()
    down_move = df['low'].shift() - df['low']
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth with Wilder's method
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / atr
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    
    return adx, plus_di, minus_di


def calculate_ichimoku(
    df: pd.DataFrame,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
    displacement: int = 26
) -> Dict[str, pd.Series]:
    """
    Calculate Ichimoku Cloud components.
    
    Args:
        df: DataFrame with high, low, close
        tenkan_period: Conversion line period
        kijun_period: Base line period
        senkou_b_period: Leading span B period
        displacement: Cloud displacement forward
    
    Returns:
        Dictionary with tenkan, kijun, senkou_a, senkou_b, chikou
    """
    # Tenkan-sen (Conversion Line)
    high_tenkan = df['high'].rolling(window=tenkan_period).max()
    low_tenkan = df['low'].rolling(window=tenkan_period).min()
    tenkan = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line)
    high_kijun = df['high'].rolling(window=kijun_period).max()
    low_kijun = df['low'].rolling(window=kijun_period).min()
    kijun = (high_kijun + low_kijun) / 2
    
    # Senkou Span A (Leading Span A)
    senkou_a = ((tenkan + kijun) / 2).shift(displacement)
    
    # Senkou Span B (Leading Span B)
    high_senkou = df['high'].rolling(window=senkou_b_period).max()
    low_senkou = df['low'].rolling(window=senkou_b_period).min()
    senkou_b = ((high_senkou + low_senkou) / 2).shift(displacement)
    
    # Chikou Span (Lagging Span)
    chikou = df['close'].shift(-displacement)
    
    return {
        'tenkan': tenkan,
        'kijun': kijun,
        'senkou_a': senkou_a,
        'senkou_b': senkou_b,
        'chikou': chikou
    }


# =============================================================================
# 3. VOLUME PRIMITIVES
# =============================================================================

def calculate_obv(df: pd.DataFrame) -> pd.Series:
    """
    Calculate On-Balance Volume (OBV).
    
    Args:
        df: DataFrame with close, volume
    
    Returns:
        OBV series
    """
    obv = pd.Series(0, index=df.index)
    obv.iloc[0] = df.iloc[0]['volume']
    
    for i in range(1, len(df)):
        if df.iloc[i]['close'] > df.iloc[i-1]['close']:
            obv.iloc[i] = obv.iloc[i-1] + df.iloc[i]['volume']
        elif df.iloc[i]['close'] < df.iloc[i-1]['close']:
            obv.iloc[i] = obv.iloc[i-1] - df.iloc[i]['volume']
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv


def calculate_relative_volume(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Calculate relative volume (current vs average).
    
    Args:
        df: DataFrame with volume
        period: Lookback period for average
    
    Returns:
        Relative volume ratio
    """
    avg_volume = df['volume'].rolling(window=period).mean()
    return df['volume'] / avg_volume


def calculate_chaikin_money_flow(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Calculate Chaikin Money Flow.
    
    Args:
        df: DataFrame with high, low, close, volume
        period: Period for CMF calculation
    
    Returns:
        CMF series
    """
    # Money Flow Multiplier
    mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    mf_multiplier = mf_multiplier.fillna(0)  # Handle division by zero
    
    # Money Flow Volume
    mf_volume = mf_multiplier * df['volume']
    
    # CMF
    cmf = mf_volume.rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
    
    return cmf


def calculate_vwmacd(
    df: pd.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Volume-Weighted MACD.
    
    Args:
        df: DataFrame with close, volume
        fast_period: Fast period
        slow_period: Slow period
        signal_period: Signal period
    
    Returns:
        (vwmacd_line, signal_line, histogram)
    """
    # Volume-weighted price
    vwp = (df['close'] * df['volume']).ewm(span=1).mean() / df['volume'].ewm(span=1).mean()
    
    # MACD on volume-weighted price
    fast_vwma = vwp.ewm(span=fast_period, adjust=False).mean()
    slow_vwma = vwp.ewm(span=slow_period, adjust=False).mean()
    
    vwmacd_line = fast_vwma - slow_vwma
    signal_line = vwmacd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = vwmacd_line - signal_line
    
    return vwmacd_line, signal_line, histogram


# =============================================================================
# 4. LEVELS PRIMITIVES
# =============================================================================

@dataclass
class PivotLevels:
    """Pivot point levels."""
    pivot: float
    r1: float
    r2: float
    r3: float
    s1: float
    s2: float
    s3: float


def calculate_pivot_points(
    high: float,
    low: float,
    close: float,
    method: str = 'standard'
) -> PivotLevels:
    """
    Calculate pivot points.
    
    Args:
        high: Previous period high
        low: Previous period low
        close: Previous period close
        method: 'standard', 'woodie', or 'camarilla'
    
    Returns:
        PivotLevels dataclass
    """
    if method == 'standard':
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)
        s1 = 2 * pivot - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)
    
    elif method == 'woodie':
        pivot = (high + low + 2 * close) / 4
        r1 = 2 * pivot - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)
        s1 = 2 * pivot - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)
    
    elif method == 'camarilla':
        pivot = (high + low + close) / 3
        range_hl = high - low
        r1 = close + range_hl * 1.1 / 12
        r2 = close + range_hl * 1.1 / 6
        r3 = close + range_hl * 1.1 / 4
        s1 = close - range_hl * 1.1 / 12
        s2 = close - range_hl * 1.1 / 6
        s3 = close - range_hl * 1.1 / 4
    
    else:
        raise ValueError(f"Unknown pivot method: {method}")
    
    return PivotLevels(pivot, r1, r2, r3, s1, s2, s3)


def calculate_fibonacci_levels(
    swing_high: float,
    swing_low: float,
    direction: str = 'retracement'
) -> Dict[str, float]:
    """
    Calculate Fibonacci retracement or extension levels.
    
    Args:
        swing_high: Swing high price
        swing_low: Swing low price
        direction: 'retracement' or 'extension'
    
    Returns:
        Dictionary of Fibonacci levels
    """
    diff = swing_high - swing_low
    
    if direction == 'retracement':
        return {
            '0.0': swing_high,
            '23.6': swing_high - 0.236 * diff,
            '38.2': swing_high - 0.382 * diff,
            '50.0': swing_high - 0.500 * diff,
            '61.8': swing_high - 0.618 * diff,
            '78.6': swing_high - 0.786 * diff,
            '100.0': swing_low
        }
    else:  # extension
        return {
            '0.0': swing_low,
            '61.8': swing_low + 0.618 * diff,
            '100.0': swing_low + diff,
            '161.8': swing_low + 1.618 * diff,
            '261.8': swing_low + 2.618 * diff,
            '423.6': swing_low + 4.236 * diff
        }


def calculate_round_levels(price: float, increment: float = 50.0) -> List[float]:
    """
    Calculate nearby round/psychological levels.
    
    Args:
        price: Current price
        increment: Round number increment (50, 100, etc.)
    
    Returns:
        List of nearby round levels
    """
    base = round(price / increment) * increment
    return [
        base - 2 * increment,
        base - increment,
        base,
        base + increment,
        base + 2 * increment
    ]


# =============================================================================
# 5. BREAKOUTS AND CONTINUATIONS
# =============================================================================

def calculate_donchian_channels(
    df: pd.DataFrame,
    period: int = 20
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Donchian Channels.
    
    Args:
        df: DataFrame with high, low
        period: Lookback period
    
    Returns:
        (upper_band, lower_band, middle_band)
    """
    upper = df['high'].rolling(window=period).max()
    lower = df['low'].rolling(window=period).min()
    middle = (upper + lower) / 2
    
    return upper, lower, middle


def detect_channel_breakout(
    df: pd.DataFrame,
    period: int = 20,
    confirmation_bars: int = 1
) -> pd.Series:
    """
    Detect breakouts above/below Donchian channels.
    
    Args:
        df: DataFrame with high, low, close
        period: Channel period
        confirmation_bars: Bars to confirm breakout
    
    Returns:
        Series with 1 (bullish breakout), -1 (bearish), 0 (no breakout)
    """
    upper, lower, _ = calculate_donchian_channels(df, period)
    
    breakouts = pd.Series(0, index=df.index)
    
    # Bullish breakout: close above upper band for N bars
    bullish = df['close'] > upper
    breakouts[bullish.rolling(window=confirmation_bars).sum() >= confirmation_bars] = 1
    
    # Bearish breakout: close below lower band for N bars
    bearish = df['close'] < lower
    breakouts[bearish.rolling(window=confirmation_bars).sum() >= confirmation_bars] = -1
    
    return breakouts


def detect_momentum_burst(
    df: pd.DataFrame,
    rsi_threshold: float = 70,
    volume_threshold: float = 2.0,
    volume_period: int = 20
) -> pd.Series:
    """
    Detect momentum bursts (RSI spike with volume).
    
    Args:
        df: DataFrame with close, volume
        rsi_threshold: RSI level to trigger
        volume_threshold: Volume multiplier vs average
        volume_period: Period for average volume
    
    Returns:
        Boolean series of momentum bursts
    """
    from src.features.indicators import calculate_rsi
    
    rsi = calculate_rsi(df['close'], period=14)
    rel_vol = calculate_relative_volume(df, period=volume_period)
    
    return (rsi > rsi_threshold) & (rel_vol > volume_threshold)


# =============================================================================
# 6. FILTERS AND RISK PRIMITIVES
# =============================================================================

def filter_time_of_day(
    timestamp: pd.Timestamp,
    allowed_hours: List[Tuple[int, int]],
    timezone: str = 'America/New_York'
) -> bool:
    """
    Check if timestamp is within allowed trading hours.
    
    Args:
        timestamp: Timestamp to check
        allowed_hours: List of (start_hour, end_hour) tuples in 24h format
        timezone: Timezone for hour check
    
    Returns:
        True if timestamp is in allowed hours
    """
    if timestamp.tz is None:
        timestamp = timestamp.tz_localize('UTC')
    
    local_time = timestamp.tz_convert(timezone)
    hour = local_time.hour
    
    for start_hour, end_hour in allowed_hours:
        if start_hour <= hour < end_hour:
            return True
    
    return False


def calculate_kelly_criterion(
    win_rate: float,
    avg_win: float,
    avg_loss: float
) -> float:
    """
    Calculate Kelly Criterion for position sizing.
    
    Args:
        win_rate: Probability of winning (0-1)
        avg_win: Average win amount
        avg_loss: Average loss amount (positive)
    
    Returns:
        Kelly percentage (fraction of capital to risk)
    """
    if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
        return 0.0
    
    # Kelly = (bp - q) / b
    # where b = avg_win/avg_loss, p = win_rate, q = 1-p
    b = avg_win / avg_loss
    p = win_rate
    q = 1 - p
    
    kelly = (b * p - q) / b
    
    # Cap at 25% for safety
    return max(0, min(kelly, 0.25))


def calculate_position_size(
    account_balance: float,
    risk_percent: float,
    entry_price: float,
    stop_price: float,
    contract_value: float = 5.0
) -> int:
    """
    Calculate position size in contracts based on risk.
    
    Args:
        account_balance: Total account balance
        risk_percent: Percentage of account to risk (0-100)
        entry_price: Entry price
        stop_price: Stop loss price
        contract_value: Dollar value per point
    
    Returns:
        Number of contracts to trade
    """
    risk_dollars = account_balance * (risk_percent / 100.0)
    risk_per_contract = abs(entry_price - stop_price) * contract_value
    
    if risk_per_contract == 0:
        return 0
    
    contracts = int(risk_dollars / risk_per_contract)
    
    return max(1, contracts)  # At least 1 contract


def check_risk_reward_ratio(
    entry_price: float,
    stop_price: float,
    target_price: float,
    min_rr: float = 2.0
) -> bool:
    """
    Check if trade meets minimum risk/reward ratio.
    
    Args:
        entry_price: Entry price
        stop_price: Stop loss price
        target_price: Take profit price
        min_rr: Minimum risk/reward ratio
    
    Returns:
        True if RR meets minimum
    """
    risk = abs(entry_price - stop_price)
    reward = abs(target_price - entry_price)
    
    if risk == 0:
        return False
    
    rr_ratio = reward / risk
    
    return rr_ratio >= min_rr

```

### src/features/levels.py

```python
"""
Price Levels
Support/resistance level detection and distance calculation.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from zoneinfo import ZoneInfo

from src.config import NY_TZ


@dataclass
class LevelValues:
    """Bundle of level-related values."""
    # 1h timeframe levels
    nearest_1h_high: float = 0.0
    nearest_1h_low: float = 0.0
    dist_1h_high: float = 0.0
    dist_1h_low: float = 0.0
    
    # 4h timeframe levels
    nearest_4h_high: float = 0.0
    nearest_4h_low: float = 0.0
    dist_4h_high: float = 0.0
    dist_4h_low: float = 0.0
    
    # Previous day levels
    pdh: float = 0.0   # Previous Day High
    pdl: float = 0.0   # Previous Day Low
    pdc: float = 0.0   # Previous Day Close
    dist_pdh: float = 0.0
    dist_pdl: float = 0.0
    
    # Current day
    current_day_high: float = 0.0
    current_day_low: float = 0.0


def get_htf_levels(
    df_htf: pd.DataFrame,
    current_time: pd.Timestamp,
    lookback_bars: int = 10
) -> List[Tuple[float, str]]:
    """
    Get high/low levels from higher timeframe bars.
    
    Returns list of (price, type) tuples where type is 'high' or 'low'.
    """
    # Filter to bars before current time
    mask = df_htf['time'] <= current_time
    recent = df_htf.loc[mask].tail(lookback_bars)
    
    levels = []
    for _, row in recent.iterrows():
        levels.append((row['high'], 'high'))
        levels.append((row['low'], 'low'))
    
    return levels


def get_nearest_level(
    price: float,
    levels: List[float]
) -> Tuple[float, float, str]:
    """
    Find nearest level to current price.
    
    Returns:
        (level_price, distance, 'above' or 'below')
    """
    if not levels:
        return (0.0, 0.0, 'none')
    
    above_levels = [l for l in levels if l >= price]
    below_levels = [l for l in levels if l < price]
    
    nearest_above = min(above_levels) if above_levels else None
    nearest_below = max(below_levels) if below_levels else None
    
    if nearest_above is None:
        return (nearest_below, price - nearest_below, 'below')
    if nearest_below is None:
        return (nearest_above, nearest_above - price, 'above')
    
    dist_above = nearest_above - price
    dist_below = price - nearest_below
    
    if dist_above <= dist_below:
        return (nearest_above, dist_above, 'above')
    else:
        return (nearest_below, dist_below, 'below')


def get_previous_day_levels(
    df: pd.DataFrame,
    current_time: pd.Timestamp,
    tz: ZoneInfo = NY_TZ
) -> dict:
    """
    Get previous day high, low, close.
    
    Uses New York timezone for day boundaries.
    """
    df = df.copy()
    
    # Convert to NY time
    if 'time' in df.columns:
        df['time_ny'] = pd.to_datetime(df['time']).dt.tz_convert(tz)
    else:
        df['time_ny'] = df.index.tz_convert(tz)
    
    df['date'] = df['time_ny'].dt.date
    current_date = current_time.astimezone(tz).date()
    
    # Get previous trading day
    unique_dates = sorted(df['date'].unique())
    if current_date not in unique_dates:
        # Find most recent date before current
        prev_dates = [d for d in unique_dates if d < current_date]
        if not prev_dates:
            return {'pdh': None, 'pdl': None, 'pdc': None}
        prev_date = max(prev_dates)
    else:
        idx = unique_dates.index(current_date)
        if idx == 0:
            return {'pdh': None, 'pdl': None, 'pdc': None}
        prev_date = unique_dates[idx - 1]
    
    # Get previous day data
    prev_day_data = df[df['date'] == prev_date]
    
    if prev_day_data.empty:
        return {'pdh': None, 'pdl': None, 'pdc': None}
    
    return {
        'pdh': prev_day_data['high'].max(),
        'pdl': prev_day_data['low'].min(),
        'pdc': prev_day_data['close'].iloc[-1],
    }


def compute_level_distances(
    current_price: float,
    levels: LevelValues,
    atr: float = 1.0
) -> dict:
    """
    Compute distances to all levels, normalized by ATR.
    
    Returns dict with distance values (positive = above, negative = below).
    """
    if atr <= 0:
        atr = 1.0
    
    return {
        'dist_1h_high_atr': (levels.nearest_1h_high - current_price) / atr if levels.nearest_1h_high else 0,
        'dist_1h_low_atr': (levels.nearest_1h_low - current_price) / atr if levels.nearest_1h_low else 0,
        'dist_4h_high_atr': (levels.nearest_4h_high - current_price) / atr if levels.nearest_4h_high else 0,
        'dist_4h_low_atr': (levels.nearest_4h_low - current_price) / atr if levels.nearest_4h_low else 0,
        'dist_pdh_atr': (levels.pdh - current_price) / atr if levels.pdh else 0,
        'dist_pdl_atr': (levels.pdl - current_price) / atr if levels.pdl else 0,
    }


def compute_levels_at_bar(
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    df_1m: pd.DataFrame,
    current_time: pd.Timestamp,
    current_price: float
) -> LevelValues:
    """
    Compute all level values at a specific point in time.
    """
    levels = LevelValues()
    
    # 1h levels
    if df_1h is not None and not df_1h.empty:
        h1_levels = get_htf_levels(df_1h, current_time, lookback_bars=10)
        highs_1h = [l[0] for l in h1_levels if l[1] == 'high']
        lows_1h = [l[0] for l in h1_levels if l[1] == 'low']
        
        if highs_1h:
            above = [h for h in highs_1h if h >= current_price]
            levels.nearest_1h_high = min(above) if above else max(highs_1h)
            levels.dist_1h_high = levels.nearest_1h_high - current_price
        
        if lows_1h:
            below = [l for l in lows_1h if l <= current_price]
            levels.nearest_1h_low = max(below) if below else min(lows_1h)
            levels.dist_1h_low = current_price - levels.nearest_1h_low
    
    # 4h levels
    if df_4h is not None and not df_4h.empty:
        h4_levels = get_htf_levels(df_4h, current_time, lookback_bars=6)
        highs_4h = [l[0] for l in h4_levels if l[1] == 'high']
        lows_4h = [l[0] for l in h4_levels if l[1] == 'low']
        
        if highs_4h:
            above = [h for h in highs_4h if h >= current_price]
            levels.nearest_4h_high = min(above) if above else max(highs_4h)
            levels.dist_4h_high = levels.nearest_4h_high - current_price
        
        if lows_4h:
            below = [l for l in lows_4h if l <= current_price]
            levels.nearest_4h_low = max(below) if below else min(lows_4h)
            levels.dist_4h_low = current_price - levels.nearest_4h_low
    
    # Previous day levels
    if df_1m is not None:
        pd_levels = get_previous_day_levels(df_1m, current_time)
        levels.pdh = pd_levels.get('pdh', 0) or 0
        levels.pdl = pd_levels.get('pdl', 0) or 0
        levels.pdc = pd_levels.get('pdc', 0) or 0
        levels.dist_pdh = levels.pdh - current_price if levels.pdh else 0
        levels.dist_pdl = current_price - levels.pdl if levels.pdl else 0
    
    return levels

```

### src/features/patterns.py

```python
"""
Chart Pattern Recognition Features

Identifies intraday chart patterns like flags, wedges, and pullbacks.
"""

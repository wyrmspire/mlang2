"""
Discovery Tools
Tools for the agent to discover available atomic components (triggers, scanners, levels).
"""
import inspect
import pandas as pd
from typing import Dict, Any, List, Optional

from src.core.tool_registry import ToolRegistry, ToolCategory
from src.policy.triggers.factory import TRIGGER_REGISTRY
# from src.policy.library import SCANNER_REGISTRY # Does not exist, we will generic discovery
from src.features.levels import LevelValues
from src.features.session_levels import SessionLevels
from src.policy.scanners import Scanner
import pkgutil
import importlib
import src.policy.library

@ToolRegistry.register(
    tool_id="list_triggers",
    category=ToolCategory.UTILITY,
    name="List Triggers",
    description="List all available trigger types for strategy composition.",
    output_schema={
        "type": "object",
        "properties": {
            "triggers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "class": {"type": "string"},
                        "description": {"type": "string"}
                    }
                }
            }
        }
    }
)
class ListTriggersTool:
    def execute(self, **kwargs):
        triggers = []
        for tid, tcls in TRIGGER_REGISTRY.items():
            doc = inspect.getdoc(tcls) or ""
            # First line of docstring as description
            desc = doc.split("\n")[0] if doc else ""
            triggers.append({
                "id": tid,
                "class": tcls.__name__,
                "description": desc
            })
        
        return {"triggers": triggers}


@ToolRegistry.register(
    tool_id="get_trigger_info",
    category=ToolCategory.UTILITY,
    name="Get Trigger Info",
    description="Get detailed schema and description for a specific trigger type.",
    input_schema={
        "type": "object",
        "properties": {
            "trigger_type": {"type": "string", "description": "ID of the trigger (e.g., 'ema_cross')"}
        },
        "required": ["trigger_type"]
    }
)
class GetTriggerInfoTool:
    def execute(self, trigger_type: str, **kwargs):
        if trigger_type not in TRIGGER_REGISTRY:
            return {"error": f"Trigger type '{trigger_type}' not found. Use list_triggers to see available types."}
        
        tcls = TRIGGER_REGISTRY[trigger_type]
        doc = inspect.getdoc(tcls) or ""
        
        # Determine params from __init__ (simplified)
        init_sig = inspect.signature(tcls.__init__)
        params = []
        for name, param in init_sig.parameters.items():
            if name == "self": continue
            default = param.default if param.default != inspect.Parameter.empty else None
            annotation = str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any"
            params.append({
                "name": name,
                "type": annotation,
                "default": str(default) if default is not None else "Required"
            })

        return {
            "id": trigger_type,
            "class": tcls.__name__,
            "description": doc,
            "parameters": params
        }


@ToolRegistry.register(
    tool_id="list_scanners",
    category=ToolCategory.UTILITY,
    name="List Library Scanners",
    description="List pre-built scanners available in the library.",
    output_schema={
        "type": "object",
        "properties": {
            "scanners": {"type": "array"}
        }
    }
)
class ListScannersTool:
    def execute(self, **kwargs):
        scanners = []
        
        # Dynamic discovery of scanners in src.policy.library
        package = src.policy.library
        path = package.__path__
        prefix = package.__name__ + "."

        for _, name, _ in pkgutil.iter_modules(path, prefix):
            try:
                module = importlib.import_module(name)
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (inspect.isclass(attr) and 
                        issubclass(attr, Scanner) and 
                        attr is not Scanner and 
                        attr.__module__ == module.__name__):
                        
                        doc = inspect.getdoc(attr) or ""
                        desc = doc.split("\n")[0] if doc else ""
                        sid = getattr(attr, "scanner_id", name.split(".")[-1])
                        # If scanner_id is a property, we might need to instantiate or guess
                        # For now, use class name as fallback ID or snake_case conversion
                        
                        scanners.append({
                            "id": sid if isinstance(sid, str) else attr.__name__,
                            "class": attr.__name__,
                            "description": desc
                        })
            except Exception as e:
                print(f"Error inspecting {name}: {e}")
            
        return {"scanners": scanners}


@ToolRegistry.register(
    tool_id="list_levels",
    category=ToolCategory.UTILITY,
    name="List Available Levels",
    description="List the types of price levels available for strategy context (e.g., PDH, Asian Low).",
    output_schema={
        "type": "object",
        "properties": {
            "levels": {"type": "array"}
        }
    }
)
class ListLevelsTool:
    def execute(self, **kwargs):
        # Inspect LevelValues and SessionLevels dataclasses to find available fields
        levels = []
        
        # From Levels (Daily/HTF)
        for field in LevelValues.__dataclass_fields__:
            levels.append({
                "id": field,
                "category": "Daily/HTF",
                "description": f"Standard level: {field}"
            })
            
        # From Session Levels
        for field in SessionLevels.__dataclass_fields__:
            levels.append({
                "id": field,
                "category": "Session",
                "description": f"Session level: {field}"
            })
            
        # Add dynamic ones (FVG, etc if we had a dedicated structure)
        levels.append({"id": "fvg_bullish", "category": "Dynamic", "description": "Nearest bullish FVG"})
        levels.append({"id": "fvg_bearish", "category": "Dynamic", "description": "Nearest bearish FVG"})
        levels.append({"id": "vwap", "category": "Indicator", "description": "Volume Weighted Average Price"})
        
        return {"levels": levels}


@ToolRegistry.register(
    tool_id="list_brackets",
    category=ToolCategory.UTILITY,
    name="List Bracket Types",
    description="List available bracket types for stop loss / take profit configuration.",
    output_schema={
        "type": "object",
        "properties": {
            "brackets": {"type": "array"}
        }
    }
)
class ListBracketsTool:
    def execute(self, **kwargs):
        """List available bracket types for OCO configuration."""
        brackets = [
            {
                "id": "atr",
                "name": "ATR-Based",
                "description": "Stop and target as multiples of ATR",
                "params": ["stop_atr", "tp_atr", "atr_period"]
            },
            {
                "id": "percent",
                "name": "Percentage-Based",
                "description": "Stop and target as percentage of entry price",
                "params": ["stop_pct", "tp_pct"]
            },
            {
                "id": "fixed",
                "name": "Fixed Points",
                "description": "Stop and target as fixed point values",
                "params": ["stop_points", "tp_points"]
            },
            {
                "id": "ict",
                "name": "ICT Levels",
                "description": "Use ICT-style levels (FVG, swing points) for targets",
                "params": ["use_fvg", "use_swing"]
            },
            {
                "id": "level",
                "name": "Level-Based",
                "description": "Target specific price levels (PDH, PDL, etc.)",
                "params": ["target_level", "stop_level"]
            }
        ]
        return {"brackets": brackets}

"""
Composition Tools
Tools for composing strategies from atomic primitives (triggers, brackets, filters).
This allows agents to build "Modular Strategies" dynamically.
"""
from typing import Dict, Any, List, Optional
import json

from src.core.tool_registry import ToolRegistry, ToolCategory
from src.policy.triggers.factory import TRIGGER_REGISTRY, trigger_from_dict

@ToolRegistry.register(
    tool_id="compose_scan",
    category=ToolCategory.STRATEGY,
    name="Compose Scan Configuration",
    description="Helper to compose a validate scan configuration for the 'run_modular_strategy' tool.",
    input_schema={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Name of the strategy"
            },
            "trigger_type": {
                "type": "string",
                "description": "Type of trigger (use list_triggers to find available types)"
            },
            "trigger_params": {
                "type": "object",
                "description": "Parameters for the trigger"
            },
            "trigger_config": {
                 "type": "object",
                 "description": "Alternative: Full trigger configuration object (recursive AND/OR)"
            },
            "bracket": {
                "type": "object",
                "description": "Bracket configuration (stop loss / take profit)",
                "properties": {
                    "type": {"type": "string", "enum": ["atr", "percent", "fixed", "ict"]},
                    "stop_atr": {"type": "number"},
                    "tp_atr": {"type": "number"},
                    "stop_pct": {"type": "number"},
                    "tp_pct": {"type": "number"}
                }
            },
            "filters": {
                "type": "array",
                "items": {"type": "object"},
                "description": "List of filters (e.g., time, session)"
            }
        },
        "required": ["name"]
    }
)
class ComposeScanTool:
    def execute(self, **inputs):
        name = inputs.get("name")
        
        # Build Trigger Config
        trigger_config = inputs.get("trigger_config")
        
        if not trigger_config:
            # Build from type/params
            t_type = inputs.get("trigger_type")
            t_params = inputs.get("trigger_params", {})
            if not t_type:
               return {"error": "Must provide either 'trigger_config' or 'trigger_type'"}
            
            trigger_config = {"type": t_type, **t_params}
            
        # Validate Trigger
        try:
            # We try to create it to validate params
            # Note: recursive triggers might need children
            # For now, simplistic validation
            if trigger_config["type"] not in TRIGGER_REGISTRY:
                 return {"error": f"Unknown trigger type: {trigger_config['type']}"}
        except Exception as e:
            return {"error": f"Invalid trigger config: {e}"}

        # Build Bracket Config
        bracket_config = inputs.get("bracket", {"type": "atr", "stop_atr": 2.0, "tp_atr": 3.0})
        
        # Assemble Full Spec
        scan_spec = {
            "trigger": trigger_config,
            "bracket": bracket_config,
            "filters": inputs.get("filters", []),
            "name": name
        }
        
        # In the future, we might save this to a DB
        # For now, we return it so the agent can pass it to run_modular_strategy
        
        return {
            "status": "success",
            "message": "Scan configuration composed successfully. Pass 'scan_spec' to run_modular_strategy.",
            "scan_spec": scan_spec
        }

@ToolRegistry.register(
    tool_id="save_scan_spec",
    category=ToolCategory.UTILITY,
    name="Save Scan Spec",
    description="Save a scan specification to the library for reuse.",
    input_schema={
        "type": "object",
        "properties": {
             "name": {"type": "string"},
             "scan_spec": {"type": "object"}
        },
        "required": ["name", "scan_spec"]
    }
)
class SaveScanSpecTool:
    def execute(self, **inputs):
        # Placeholder for saving to file system
        # src/policy/library/user/{name}.json
        return {"status": "success", "message": "Scan saved (mock)"}

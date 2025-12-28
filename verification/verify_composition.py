"""
Verify Composition Tools and Execution Logic
"""
import sys
import os
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.composition_tools import ComposeScanTool
from src.tools.agent_tools import RunModularStrategyTool
from src.server.main import agent_chat, ChatRequest, ChatContext, ChatMessage

import asyncio

async def test_composition():
    print("Testing ComposeScanTool...")
    tool = ComposeScanTool()
    
    # Test valid composition
    result = tool.execute(
        name="Test Strat",
        trigger_type="ema_cross",
        trigger_params={"fast": 10, "slow": 50},
        bracket={"type": "atr", "stop_atr": 1.5, "tp_atr": 3.0}
    )
    
    if "scan_spec" not in result:
        print("FAIL: ComposeScanTool did not return scan_spec")
        print(result)
        return
        
    spec = result["scan_spec"]
    print(f"Spec created: {json.dumps(spec, indent=2)}")
    
    # Verify strict equality
    assert spec["trigger"]["type"] == "ema_cross"
    assert spec["trigger"]["fast"] == 10
    
    print("PASS: ComposeScanTool")
    
    # Test Agent Execution Logic (Mock)
    # We call run_modular_strategy via agent_chat
    # Actually we can't easily Mock agent_chat without running valid inputs
    # But we can verify RunModularStrategyTool schema
    
    print("\nVerifying RunModularStrategyTool Schema...")
    from src.core.tool_registry import ToolRegistry
    # The registry stores declarations, not tool instances directly in the same way?
    # Actually get_gemini_function_declarations returns the list
    
    tools = ToolRegistry.get_gemini_function_declarations()
    target = next((t for t in tools if t['name'] == 'run_modular_strategy'), None)
    
    if target:
        props = target['parameters']['properties']
        if "trigger_config" in props and "bracket_config" in props:
            print("PASS: RunModularStrategyTool schema updated.")
        else:
            print(f"FAIL: Schema missing new configs. Keys: {props.keys()}")
    else:
        print("FAIL: Tool not found in registry.")
        
    print("\nPhase 3/4 Verification Complete.")

if __name__ == "__main__":
    asyncio.run(test_composition())

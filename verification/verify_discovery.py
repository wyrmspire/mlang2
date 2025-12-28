import sys
import os
import json

# Add project root to path
sys.path.append(os.getcwd())

from src.core.tool_registry import ToolRegistry
import src.server.main  # Trigger registration via imports

def verify_tool(tool_id):
    print(f"\n--- Verifying {tool_id} ---")
    try:
        tool = ToolRegistry.get_tool(tool_id)
        if not tool:
            print(f"❌ Tool {tool_id} NOT found in registry")
            return
        
        print(f"✅ Tool {tool_id} found")
        result = tool.execute()
        print(f"Output: {json.dumps(result, indent=2)[:500]}...") # Truncate output
        
        # specific checks
        if tool_id == "list_triggers":
            triggers = result.get("triggers", [])
            print(f"Found {len(triggers)} triggers")
            has_vwap = any(t["id"] == "vwap_reclaim" for t in triggers)
            print(f"Has 'vwap_reclaim': {has_vwap}")
            
        if tool_id == "list_scanners":
            scanners = result.get("scanners", [])
            print(f"Found {len(scanners)} scanners")
            
        if tool_id == "list_levels":
            levels = result.get("levels", [])
            print(f"Found {len(levels)} levels")
            
    except Exception as e:
        print(f"❌ Error executing {tool_id}: {e}")

if __name__ == "__main__":
    verify_tool("list_triggers")
    verify_tool("list_scanners")
    verify_tool("list_levels")

from typing import List, Dict, Any, Optional
from src.core.tool_registry import ToolRegistry, ToolCategory

@ToolRegistry.register(
    tool_id="create_plan",
    category=ToolCategory.UTILITY,
    name="Create Plan",
    description="Outline a multi-step plan before executing it. This helps you think clearly and break down complex tasks.",
    input_schema={
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of steps to execute"
            },
            "goal": {
                "type": "string",
                "description": "The overall goal of this plan"
            }
        },
        "required": ["steps"]
    }
)
class CreatePlanTool:
    def execute(self, steps: List[str], goal: str = "") -> str:
        plan_str = f"## Plan for: {goal}\n" if goal else "## Plan\n"
        for i, step in enumerate(steps):
            plan_str += f"{i+1}. {step}\n"
        return plan_str


@ToolRegistry.register(
    tool_id="execute_plan",
    category=ToolCategory.UTILITY,
    name="Execute Plan",
    description="Execute a list of tool calls sequentially. Use this for complex workflows, comparisons, or multi-step analysis.",
    input_schema={
        "type": "object",
        "properties": {
            "tool_calls": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "tool_id": {"type": "string", "description": "The ID of the tool to execute"},
                        "args": {"type": "object", "description": "The arguments for the tool"}
                    },
                    "required": ["tool_id", "args"]
                },
                "description": "List of tool calls to execute in order"
            }
        },
        "required": ["tool_calls"]
    }
)
class ExecutePlanTool:
    def execute(self, tool_calls: List[Dict[str, Any]]) -> str:
        results = []
        for i, call in enumerate(tool_calls):
            tool_id = call.get('tool_id')
            args = call.get('args', {})

            # Special handling for tools that might not be in registry but handled by main.py
            # If main.py handles them, we might be unable to invoke them directly here unless they are registered.
            # Fortunately, most important tools are registered or can be invoked via ToolRegistry if we ensure they are.

            try:
                # Try to create and execute the tool
                # Note: run_strategy and others are registered in agent_tools.py,
                # but their execute() method just returns a dict saying "queued".
                # Real execution happens in main.py loop.
                # HOWEVER, for analysis tools (cluster_trades, etc.), they execute immediately.

                tool_instance = ToolRegistry.get_tool(tool_id)
                if not tool_instance:
                     # Attempt to create with params if needed, but get_tool creates empty
                     tool_instance = ToolRegistry.create(tool_id)

                if tool_instance:
                    res = tool_instance.execute(**args)

                    # If it's a "queued" result (like run_strategy), we might want to actually run it?
                    # The current architecture relies on main.py interception for 'run_strategy'.
                    # But 'run_strategy' tool in agent_tools.py returns a dict.
                    # If we want to actually run it, we need to call the logic that main.py calls.
                    # This is tricky without refactoring main.py logic into the tool.
                    # For now, we assume this tool is best for ANALYSIS tools (find_price_opportunities, etc.)
                    # If the user wants to run 3 strategies, they will get 3 "queued" messages.

                    import json
                    res_str = json.dumps(res, indent=2, default=str)
                    results.append(f"### Step {i+1}: {tool_id}\n```json\n{res_str}\n```")
                else:
                    results.append(f"### Step {i+1}: {tool_id}\nError: Tool not found in registry.")

            except Exception as e:
                results.append(f"### Step {i+1}: {tool_id}\nError: {str(e)}")

        return "\n\n".join(results)

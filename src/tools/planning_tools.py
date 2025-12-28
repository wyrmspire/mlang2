from typing import List, Dict, Any, Optional
from src.core.tool_registry import ToolRegistry, ToolCategory

@ToolRegistry.register(
    tool_id="CreatePlanTool",
    name="CreatePlanTool",
    category=ToolCategory.UTILITY,
    description="Plan a sequence of steps for complex tasks. Use this when the user asks for a 'plan' or a multi-step analysis."
)
class CreatePlanTool:
    def execute(self, steps: List[str], goal: str) -> Dict[str, Any]:
        """
        Create a structured plan.

        Args:
            steps: List of steps to execute.
            goal: The overall goal of the plan.

        Returns:
            The structured plan.
        """
        return {
            "goal": goal,
            "steps": steps,
            "status": "planned"
        }

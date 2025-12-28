import pytest
from src.core.tool_registry import ToolRegistry, ToolCategory
from src.tools.planning_tools import ExecutePlanTool, CreatePlanTool

# Mock tool for testing
@ToolRegistry.register(
    tool_id="mock_tool",
    category=ToolCategory.UTILITY,
    name="Mock Tool",
    description="A mock tool for testing",
    input_schema={"type": "object", "properties": {"msg": {"type": "string"}}}
)
class MockTool:
    def execute(self, msg: str):
        return f"Echo: {msg}"

def test_create_plan():
    tool = CreatePlanTool()
    steps = ["Step A", "Step B"]
    result = tool.execute(steps=steps, goal="Test Goal")
    assert "Plan for: Test Goal" in result
    assert "1. Step A" in result
    assert "2. Step B" in result

def test_execute_plan():
    tool = ExecutePlanTool()
    tool_calls = [
        {"tool_id": "mock_tool", "args": {"msg": "Hello"}},
        {"tool_id": "mock_tool", "args": {"msg": "World"}}
    ]

    result = tool.execute(tool_calls)

    assert "Step 1: mock_tool" in result
    assert "Echo: Hello" in result
    assert "Step 2: mock_tool" in result
    assert "Echo: World" in result

def test_execute_plan_invalid_tool():
    tool = ExecutePlanTool()
    tool_calls = [
        {"tool_id": "invalid_tool_xyz", "args": {}}
    ]

    result = tool.execute(tool_calls)
    assert "Step 1: invalid_tool_xyz" in result
    assert "Error" in result

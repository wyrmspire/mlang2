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

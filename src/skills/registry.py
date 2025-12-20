"""
Skill Registry
Discover and access all available agent skills.
"""

from typing import Dict, List, Callable
import inspect

from src.skills.data import ingest_raw_data, get_data_summary
from src.skills.research import run_research_experiment, run_simple_walkforward
from src.skills.model import train_agent_model, evaluate_model_performance

# Import new agent tools (lazy import to avoid circular dependencies)
def _get_strategy_tools():
    """Lazy import of strategy tools."""
    from src.tools.catalog import list_all_tools
    from src.tools.strategy_builder import StrategyBuilder
    from src.tools.simulation_runner import run_simulation
    from src.tools.pattern_scanner import scan_pattern
    from src.tools.trigger_composer import TriggerComposer
    from src.tools.validation import StrategyValidator
    return {
        'list_all_tools': list_all_tools,
        'StrategyBuilder': StrategyBuilder,
        'run_simulation': run_simulation,
        'scan_pattern': scan_pattern,
        'TriggerComposer': TriggerComposer,
        'StrategyValidator': StrategyValidator,
    }

class SkillRegistry:
    """
    Registry of all available skills.
    Agents can query this to understand what they can do.
    """
    
    def __init__(self):
        self._skills: Dict[str, Dict] = {}
        self._register_defaults()
    
    def _register_defaults(self):
        # Data Skills
        self.register("ingest_raw_data", ingest_raw_data, "Ingests raw JSON data into processed Parquet.")
        self.register("get_data_summary", get_data_summary, "Provides a summary of available data.")
        
        # Research Skills
        self.register("run_experiment", run_research_experiment, "Runs a standard research experiment.")
        self.register("run_walkforward", run_simple_walkforward, "Runs a walk-forward research session.")
        
        # Model Skills
        self.register("train_model", train_agent_model, "Trains a neural model on sharded data.")
        self.register("evaluate_model", evaluate_model_performance, "Evaluates a model's performance.")
        
        # Strategy Building Skills (lazy loaded)
        tools = _get_strategy_tools()
        self.register("list_all_tools", tools['list_all_tools'], "Lists all available agent tools with descriptions and examples.")
        self.register("run_simulation", tools['run_simulation'], "Runs strategy simulation with model inference on playback data.")
        self.register("scan_pattern", tools['scan_pattern'], "Scans historical data for pattern occurrences and success rates.")
        
        # Register class-based tools with wrapper functions
        self.register("build_strategy_from_template", 
                     lambda template_id, **kwargs: tools['StrategyBuilder'].from_template(template_id, **kwargs),
                     "Builds a strategy from a template (mean_reversion, opening_range, time_based, etc.)")
        self.register("build_custom_strategy",
                     lambda name, scanner_id, **kwargs: tools['StrategyBuilder'].create(name, scanner_id, **kwargs),
                     "Builds a custom strategy from components (scanner, trigger, bracket)")
        self.register("list_strategy_templates",
                     lambda: [t.template_id for t in tools['StrategyBuilder'].list_templates()],
                     "Lists all available strategy templates")
        self.register("validate_strategy",
                     lambda strategy: tools['StrategyValidator'].validate(strategy),
                     "Validates a strategy configuration before execution")
        self.register("compose_triggers_and",
                     lambda triggers: tools['TriggerComposer'].AND(triggers),
                     "Combines multiple triggers with AND logic (all must fire)")
        self.register("compose_triggers_or",
                     lambda triggers: tools['TriggerComposer'].OR(triggers),
                     "Combines multiple triggers with OR logic (any can fire)")

    def register(self, name: str, func: Callable, description: str):
        self._skills[name] = {
            "func": func,
            "description": description,
            "signature": str(inspect.signature(func))
        }

    def list_skills(self) -> List[Dict]:
        """Returns a list of all skills with descriptions."""
        return [
            {"name": name, "description": data["description"], "signature": data["signature"]}
            for name, data in self._skills.items()
        ]

    def get_skill(self, name: str) -> Callable:
        if name not in self._skills:
            raise ValueError(f"Skill '{name}' not found.")
        return self._skills[name]["func"]

# Global registry instance
registry = SkillRegistry()

def list_available_skills():
    """Helper function for agents to see what they can do."""
    skills = registry.list_skills()
    output = ["Available Agent Skills in mlang2:"]
    for s in skills:
        output.append(f"- {s['name']}{s['signature']}: {s['description']}")
    return "\n".join(output)

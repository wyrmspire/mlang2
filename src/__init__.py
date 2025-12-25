# MLang2 - Trade Simulation & Research Platform
"""
A deterministic, causal-correct platform for simulating trades,
logging decisions, and training models to predict counterfactual outcomes.
"""

__version__ = "0.1.0"

# Lazy imports to avoid loading heavy dependencies (torch, etc.) on import
def _get_skills_registry():
    """Lazy import of skills registry to avoid torch dependency on module import."""
    from src.skills.registry import registry
    return registry

def _list_available_skills():
    """Lazy import of skills list."""
    from src.skills.registry import list_available_skills
    return list_available_skills()

# For backward compatibility
skills = None  # Will be loaded on first access if needed

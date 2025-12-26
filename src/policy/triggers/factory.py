"""
Trigger Factory
Separated from __init__ to avoid circular imports.
"""

from typing import Dict, Any, List

from .base import Trigger
from .time_trigger import TimeTrigger
from .candle_patterns import CandlePatternTrigger
from .indicator_triggers import EMACrossTrigger, RSIThresholdTrigger
from .structure_break import StructureBreakTrigger
from .fakeout import FakeoutTrigger
from .ema_rejection import EMA200RejectionTrigger
from .parametric import RejectionTrigger, ComparisonTrigger
from .sweep import SweepTrigger
from .or_false_break import ORFalseBreakTrigger
from .vwap_reclaim import VWAPReclaimTrigger
# Logic triggers imported safely or lazily if needed
# To avoid cycle, we'll import logic classes inside the factory/registry if they depend on this factory
# But AndTrigger needs trigger_from_dict...

# Solution:
# 1. Define Registry here (empty or populated with leaves)
# 2. Logic triggers import trigger_from_dict
# 3. We import Logic triggers here to populate registry

# Wait, if logic.py imports trigger_from_dict, then trigger_from_dict cannot import logic.py at top level
# unless we use lazy import inside the function.

TRIGGER_REGISTRY = {}

def register_triggers():
    """Populate registry. Call this once or ensure imports happen."""
    # Leaf triggers (no recursion)
    TRIGGER_REGISTRY["time"] = TimeTrigger
    TRIGGER_REGISTRY["candle_pattern"] = CandlePatternTrigger
    TRIGGER_REGISTRY["ema_cross"] = EMACrossTrigger
    TRIGGER_REGISTRY["rsi_threshold"] = RSIThresholdTrigger
    TRIGGER_REGISTRY["structure_break"] = StructureBreakTrigger
    TRIGGER_REGISTRY["fakeout"] = FakeoutTrigger
    TRIGGER_REGISTRY["ema200_rejection"] = EMA200RejectionTrigger
    TRIGGER_REGISTRY["rejection"] = RejectionTrigger
    TRIGGER_REGISTRY["comparison"] = ComparisonTrigger
    TRIGGER_REGISTRY["sweep"] = SweepTrigger
    TRIGGER_REGISTRY["or_false_break"] = ORFalseBreakTrigger
    TRIGGER_REGISTRY["vwap_reclaim"] = VWAPReclaimTrigger
    
    # Logic triggers (recursive)
    from .logic import AndTrigger, OrTrigger, NotTrigger
    TRIGGER_REGISTRY["AND"] = AndTrigger
    TRIGGER_REGISTRY["OR"] = OrTrigger
    TRIGGER_REGISTRY["NOT"] = NotTrigger

def trigger_from_dict(config: dict) -> Trigger:
    """
    Factory function to create trigger from config dict.
    """
    config = config.copy()
    trigger_type = config.pop("type")
    
    if trigger_type not in TRIGGER_REGISTRY:
        # Try re-registering just in case import order messed it up
        register_triggers()
        if trigger_type not in TRIGGER_REGISTRY:
             raise ValueError(f"Unknown trigger type: {trigger_type}. Available: {list(TRIGGER_REGISTRY.keys())}")
    
    return TRIGGER_REGISTRY[trigger_type](**config)


def list_triggers() -> list:
    """List available trigger types for agent discovery."""
    return list(TRIGGER_REGISTRY.keys())


# Pre-populate registry after functions are defined
register_triggers()



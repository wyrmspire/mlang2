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
from .parametric import ComparisonTrigger # Removed RejectionTrigger to use new one
from .price_action_triggers import (
    RejectionTrigger, 
    PinBarTrigger, 
    EngulfingTrigger, 
    InsideBarTrigger, 
    DoubleTopBottomTrigger, 
    FlagPatternTrigger
)
from .sweep import SweepTrigger
from .or_false_break import ORFalseBreakTrigger
from .vwap_reclaim import VWAPReclaimTrigger

# ...

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
    
    # New Price Action Triggers
    TRIGGER_REGISTRY["rejection"] = RejectionTrigger
    TRIGGER_REGISTRY["pin_bar"] = PinBarTrigger
    TRIGGER_REGISTRY["engulfing"] = EngulfingTrigger
    TRIGGER_REGISTRY["inside_bar"] = InsideBarTrigger
    TRIGGER_REGISTRY["double_top_bottom"] = DoubleTopBottomTrigger
    TRIGGER_REGISTRY["flag_pattern"] = FlagPatternTrigger
    
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



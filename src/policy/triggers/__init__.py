"""
Trigger Components Package

Simple, atomic entry signals for agent-friendly strategy building.
"""

from .base import Trigger, TriggerResult, TriggerDirection
from .time_trigger import TimeTrigger
from .candle_patterns import CandlePatternTrigger, CandlePattern
from .indicator_triggers import EMACrossTrigger, RSIThresholdTrigger
from .structure_break import StructureBreakTrigger
from .fakeout import FakeoutTrigger
from .ema_rejection import EMA200RejectionTrigger
from .parametric import RejectionTrigger, ComparisonTrigger
from .sweep import SweepTrigger
from .or_false_break import ORFalseBreakTrigger
from .vwap_reclaim import VWAPReclaimTrigger

# Registry of all available triggers
TRIGGER_REGISTRY = {
    "time": TimeTrigger,
    "candle_pattern": CandlePatternTrigger,
    "ema_cross": EMACrossTrigger,
    "rsi_threshold": RSIThresholdTrigger,
    "structure_break": StructureBreakTrigger,
    "fakeout": FakeoutTrigger,
    "ema200_rejection": EMA200RejectionTrigger,
    # Parametric triggers - the preferred way
    "rejection": RejectionTrigger,
    "comparison": ComparisonTrigger,
    "sweep": SweepTrigger,
    "or_false_break": ORFalseBreakTrigger,
    "vwap_reclaim": VWAPReclaimTrigger,
}


def trigger_from_dict(config: dict) -> Trigger:
    """
    Factory function to create trigger from config dict.
    
    Agent-friendly interface:
        trigger_from_dict({"type": "time", "hour": 10, "minute": 0})
        trigger_from_dict({"type": "ema_cross", "fast": 9, "slow": 21})
    """
    config = config.copy()
    trigger_type = config.pop("type")
    
    if trigger_type not in TRIGGER_REGISTRY:
        raise ValueError(f"Unknown trigger type: {trigger_type}. Available: {list(TRIGGER_REGISTRY.keys())}")
    
    return TRIGGER_REGISTRY[trigger_type](**config)


def list_triggers() -> list:
    """List available trigger types for agent discovery."""
    return list(TRIGGER_REGISTRY.keys())

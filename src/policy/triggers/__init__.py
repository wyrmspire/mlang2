"""
Trigger Components Package

Simple, atomic entry signals for agent-friendly strategy building.
"""

from .base import Trigger, TriggerResult, TriggerDirection
# Export classes for direct usage if needed
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
from .logic import AndTrigger, OrTrigger, NotTrigger

# Export factory
from .factory import trigger_from_dict, TRIGGER_REGISTRY, list_triggers

# Alias list_triggers if needed or just use keys
def list_triggers() -> list:
    return list(TRIGGER_REGISTRY.keys())

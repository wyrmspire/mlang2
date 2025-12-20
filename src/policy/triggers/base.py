"""
Trigger Components - Base Classes and Types

Triggers are the simplest atomic entry signals.
They answer: "Should we consider entering right now?"
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from enum import Enum


class TriggerDirection(Enum):
    """Direction bias from trigger."""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"  # Trigger fires but no directional bias


@dataclass
class TriggerResult:
    """Result from checking a trigger."""
    trigger_id: str
    triggered: bool
    direction: TriggerDirection = TriggerDirection.NEUTRAL
    context: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0  # 0-1 confidence in signal


class Trigger(ABC):
    """
    Base class for atomic entry triggers.
    
    Triggers are simpler than Scanners:
    - No state management required (stateless preferred)
    - Single responsibility: "Does condition X hold right now?"
    - Composable: Multiple triggers can be ANDed/ORed together
    
    Agents can specify triggers via simple config dicts:
    {"type": "time", "hour": 10, "minute": 0}
    {"type": "ema_cross", "fast": 9, "slow": 21}
    """
    
    @property
    @abstractmethod
    def trigger_id(self) -> str:
        """Unique identifier for this trigger type."""
        pass
    
    @property
    def params(self) -> Dict[str, Any]:
        """Serializable parameters for this trigger instance."""
        return {}
    
    @abstractmethod
    def check(self, features: 'FeatureBundle') -> TriggerResult:
        """
        Check if trigger condition is met.
        
        Args:
            features: Current market features
            
        Returns:
            TriggerResult with triggered flag and context
        """
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize trigger to dict for agent use."""
        return {
            "type": self.trigger_id,
            **self.params
        }
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'Trigger':
        """Factory method - implemented by registry."""
        raise NotImplementedError("Use trigger_from_dict() factory function")


# Import FeatureBundle for type hints (avoid circular import)
from src.features.pipeline import FeatureBundle

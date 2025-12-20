"""
Trigger Composer - Combine multiple triggers with logical operators.

Allows agents to build complex entry conditions by composing simple triggers.
"""

from typing import List, Dict, Any
from dataclasses import dataclass

from src.policy.triggers.base import Trigger, TriggerResult, TriggerDirection
from src.features.pipeline import FeatureBundle


class CompositeTrigger(Trigger):
    """Base class for composite triggers."""
    
    def __init__(self, triggers: List[Trigger]):
        self._triggers = triggers
    
    @property
    def params(self) -> Dict[str, Any]:
        return {
            'triggers': [t.to_dict() for t in self._triggers]
        }
    
    def reset(self):
        """Reset all child triggers."""
        for trigger in self._triggers:
            if hasattr(trigger, 'reset'):
                trigger.reset()


class ANDTrigger(CompositeTrigger):
    """Triggers when ALL child triggers fire."""
    
    @property
    def trigger_id(self) -> str:
        return "and_composite"
    
    def check(self, features: FeatureBundle) -> TriggerResult:
        """Check all triggers - must all be true."""
        results = [t.check(features) for t in self._triggers]
        
        # All must trigger
        if not all(r.triggered for r in results):
            return TriggerResult(
                trigger_id=self.trigger_id,
                triggered=False,
            )
        
        # Combine context from all triggers
        combined_context = {}
        for i, result in enumerate(results):
            combined_context[f'trigger_{i}'] = {
                'id': result.trigger_id,
                'direction': result.direction.value,
                **result.context
            }
        
        # Direction: use first trigger's direction (could be smarter)
        direction = results[0].direction
        
        # Confidence: minimum of all confidences
        confidence = min(r.confidence for r in results)
        
        return TriggerResult(
            trigger_id=self.trigger_id,
            triggered=True,
            direction=direction,
            context=combined_context,
            confidence=confidence,
        )


class ORTrigger(CompositeTrigger):
    """Triggers when ANY child trigger fires."""
    
    @property
    def trigger_id(self) -> str:
        return "or_composite"
    
    def check(self, features: FeatureBundle) -> TriggerResult:
        """Check all triggers - at least one must be true."""
        results = [t.check(features) for t in self._triggers]
        
        # At least one must trigger
        triggered_results = [r for r in results if r.triggered]
        if not triggered_results:
            return TriggerResult(
                trigger_id=self.trigger_id,
                triggered=False,
            )
        
        # Use first triggered result
        first_triggered = triggered_results[0]
        
        # Combine context
        combined_context = {
            'triggered_count': len(triggered_results),
            'triggers': [r.trigger_id for r in triggered_results],
            'primary': first_triggered.to_dict() if hasattr(first_triggered, 'to_dict') else {},
        }
        
        return TriggerResult(
            trigger_id=self.trigger_id,
            triggered=True,
            direction=first_triggered.direction,
            context=combined_context,
            confidence=first_triggered.confidence,
        )


class SequenceTrigger(CompositeTrigger):
    """
    Triggers when conditions fire in sequence over multiple bars.
    
    Example: First RSI oversold, then hammer candle within 3 bars.
    """
    
    def __init__(self, triggers: List[Trigger], max_bars: int = 5):
        super().__init__(triggers)
        self._max_bars = max_bars
        self._sequence_state = []  # Track which triggers have fired
        self._last_check_idx = -1
    
    @property
    def trigger_id(self) -> str:
        return "sequence_composite"
    
    def check(self, features: FeatureBundle) -> TriggerResult:
        """Check triggers in sequence."""
        current_idx = features.bar_idx if hasattr(features, 'bar_idx') else 0
        
        # Reset if too much time has passed
        if current_idx - self._last_check_idx > self._max_bars:
            self._sequence_state = []
        
        self._last_check_idx = current_idx
        
        # Check next trigger in sequence
        next_trigger_idx = len(self._sequence_state)
        if next_trigger_idx >= len(self._triggers):
            # Sequence complete - fire and reset
            triggered = True
            self._sequence_state = []
        else:
            # Check next trigger
            result = self._triggers[next_trigger_idx].check(features)
            if result.triggered:
                self._sequence_state.append({
                    'bar_idx': current_idx,
                    'trigger_id': result.trigger_id,
                    'direction': result.direction.value,
                })
            
            # Complete?
            triggered = len(self._sequence_state) == len(self._triggers)
            if triggered:
                # Reset for next sequence
                context = {'sequence': self._sequence_state.copy()}
                self._sequence_state = []
                return TriggerResult(
                    trigger_id=self.trigger_id,
                    triggered=True,
                    direction=TriggerDirection.NEUTRAL,
                    context=context,
                )
        
        return TriggerResult(
            trigger_id=self.trigger_id,
            triggered=False,
        )
    
    def reset(self):
        """Reset sequence state."""
        super().reset()
        self._sequence_state = []
        self._last_check_idx = -1


class TriggerComposer:
    """
    High-level API for composing triggers.
    
    Usage:
        # AND composition
        time_and_rsi = TriggerComposer.AND([
            {"type": "time", "hour": 10, "minute": 0},
            {"type": "rsi_threshold", "threshold": 30, "direction": "below"}
        ])
        
        # OR composition
        pattern_or_level = TriggerComposer.OR([
            {"type": "candle_pattern", "patterns": ["hammer"]},
            {"type": "level_proximity", "atr_threshold": 0.3}
        ])
        
        # Sequence composition
        rsi_then_pattern = TriggerComposer.SEQUENCE([
            {"type": "rsi_threshold", "threshold": 30, "direction": "below"},
            {"type": "candle_pattern", "patterns": ["hammer"]}
        ], max_bars=3)
    """
    
    @staticmethod
    def AND(trigger_configs: List[Dict[str, Any]]) -> ANDTrigger:
        """Create AND composite trigger."""
        from src.policy.triggers import trigger_from_dict
        triggers = [trigger_from_dict(cfg) for cfg in trigger_configs]
        return ANDTrigger(triggers)
    
    @staticmethod
    def OR(trigger_configs: List[Dict[str, Any]]) -> ORTrigger:
        """Create OR composite trigger."""
        from src.policy.triggers import trigger_from_dict
        triggers = [trigger_from_dict(cfg) for cfg in trigger_configs]
        return ORTrigger(triggers)
    
    @staticmethod
    def SEQUENCE(trigger_configs: List[Dict[str, Any]], max_bars: int = 5) -> SequenceTrigger:
        """Create sequence composite trigger."""
        from src.policy.triggers import trigger_from_dict
        triggers = [trigger_from_dict(cfg) for cfg in trigger_configs]
        return SequenceTrigger(triggers, max_bars)
    
    @staticmethod
    def validate_composition(trigger_configs: List[Dict[str, Any]]) -> bool:
        """Validate that all triggers in composition are valid."""
        from src.policy.triggers import trigger_from_dict
        try:
            for cfg in trigger_configs:
                trigger_from_dict(cfg)
            return True
        except Exception as e:
            raise ValueError(f"Invalid trigger in composition: {e}")
    
    @staticmethod
    def explain_composition(composite_trigger: CompositeTrigger) -> str:
        """Explain a composite trigger."""
        output = [f"Composite Trigger: {composite_trigger.trigger_id}"]
        output.append("=" * 60)
        output.append(f"\nChild Triggers ({len(composite_trigger._triggers)}):")
        
        for i, trigger in enumerate(composite_trigger._triggers):
            output.append(f"  {i+1}. {trigger.trigger_id}")
            output.append(f"     Config: {trigger.to_dict()}")
        
        logic = "ALL must trigger" if isinstance(composite_trigger, ANDTrigger) else "ANY can trigger"
        if isinstance(composite_trigger, SequenceTrigger):
            logic = f"Must trigger in sequence within {composite_trigger._max_bars} bars"
        
        output.append(f"\nLogic: {logic}")
        
        return "\n".join(output)


# Register composite triggers in the registry
from src.policy.triggers import TRIGGER_REGISTRY

TRIGGER_REGISTRY['and'] = ANDTrigger
TRIGGER_REGISTRY['or'] = ORTrigger
TRIGGER_REGISTRY['sequence'] = SequenceTrigger

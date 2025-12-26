"""
Logic Triggers
Boolean logic for composing triggers: AND, OR, NOT.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from src.policy.triggers.base import Trigger, TriggerResult, TriggerDirection

from src.features.pipeline import FeatureBundle


class AndTrigger(Trigger):
    """
    Fires only if ALL child triggers fire.
    
    Direction logic:
    - If all children agree on direction (or are NEUTRAL), inherit that direction.
    - If children conflict (LONG vs SHORT), returns NEUTRAL (but still triggered).
    """
    
    def __init__(self, children: List[Dict[str, Any]]):
        from src.policy.triggers.factory import trigger_from_dict
        self.children_configs = children
        self._children: List[Trigger] = [trigger_from_dict(c) for c in children]
        
    @property
    def trigger_id(self) -> str:
        return "AND"
        
    @property
    def params(self) -> Dict[str, Any]:
        return {"children": self.children_configs}
        
    def check(self, features: FeatureBundle) -> TriggerResult:
        results = []
        for child in self._children:
            res = child.check(features)
            if not res.triggered:
                # Short circuit
                return TriggerResult(
                    trigger_id=self.trigger_id, 
                    triggered=False
                )
            results.append(res)
            
        # If we got here, all triggered
        
        # Determine direction
        directions = {r.direction for r in results if r.direction != TriggerDirection.NEUTRAL}
        
        final_dir = TriggerDirection.NEUTRAL
        if len(directions) == 1:
            final_dir = list(directions)[0]
        # If len > 1, conflict -> NEUTRAL (but triggered)
        
        # Merge context
        merged_context = {}
        for i, res in enumerate(results):
            # Prefix keys to avoid collisions? Or just merge?
            # Merging allows downstream to see "rsi": 30
            merged_context.update(res.context)
            
        return TriggerResult(
            trigger_id=self.trigger_id,
            triggered=True,
            direction=final_dir,
            context=merged_context,
            confidence=min(r.confidence for r in results)  # Weakest link
        )


class OrTrigger(Trigger):
    """
    Fires if ANY child trigger fires.
    """
    
    def __init__(self, children: List[Dict[str, Any]]):
        from src.policy.triggers.factory import trigger_from_dict
        self.children_configs = children
        self._children: List[Trigger] = [trigger_from_dict(c) for c in children]
        
    @property
    def trigger_id(self) -> str:
        return "OR"
        
    @property
    def params(self) -> Dict[str, Any]:
        return {"children": self.children_configs}
        
    def check(self, features: FeatureBundle) -> TriggerResult:
        fired = []
        for child in self._children:
            res = child.check(features)
            if res.triggered:
                fired.append(res)
                
        if not fired:
            return TriggerResult(trigger_id=self.trigger_id, triggered=False)
            
        # Use the most confident one, or the first one
        best_res = max(fired, key=lambda r: r.confidence)
        
        return TriggerResult(
            trigger_id=self.trigger_id,
            triggered=True,
            direction=best_res.direction,
            context=best_res.context,
            confidence=best_res.confidence
        )


class NotTrigger(Trigger):
    """
    Inverts the triggered status of a child.
    
    Note: Direction is inverted if possible (LONG -> SHORT).
    """
    
    def __init__(self, child: Dict[str, Any]):
        from src.policy.triggers.factory import trigger_from_dict
        self.child_config = child
        self._child = trigger_from_dict(child)
        
    @property
    def trigger_id(self) -> str:
        return "NOT"
        
    @property
    def params(self) -> Dict[str, Any]:
        return {"child": self.child_config}
        
    def check(self, features: FeatureBundle) -> TriggerResult:
        res = self._child.check(features)
        
        # Invert triggered status
        triggered = not res.triggered
        
        if not triggered:
            return TriggerResult(trigger_id=self.trigger_id, triggered=False)
            
        # Invert direction if valid
        direction = TriggerDirection.NEUTRAL
        if res.direction == TriggerDirection.LONG:
            direction = TriggerDirection.SHORT
        elif res.direction == TriggerDirection.SHORT:
            direction = TriggerDirection.LONG
            
        return TriggerResult(
            trigger_id=self.trigger_id,
            triggered=True,
            direction=direction,
            context=res.context
        )

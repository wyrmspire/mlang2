"""
Modular Scanner

Wraps Triggers to provide a standard Scanner interface.
"""

from typing import Dict, Any, Optional
import inspect

from src.policy.scanners import Scanner, ScanResult
from src.policy.triggers import Trigger, trigger_from_dict
from src.policy.triggers.base import TriggerDirection


class ModularScanner(Scanner):
    """
    Scanner that uses a modular Trigger to detect decision points.
    
    This bridges the high-level Scanner interface used by backtesters
    with the atomic Trigger components.
    """
    
    def __init__(self, trigger_config: Dict[str, Any], cooldown_bars: int = 20):
        self._trigger = trigger_from_dict(trigger_config)
        self._cooldown_bars = cooldown_bars
        self._last_trigger_idx = -1000
    
    @property
    def scanner_id(self) -> str:
        return f"modular_{self._trigger.trigger_id}"
    
    def scan(self, state, features, **kwargs) -> ScanResult:
        # Check cooldown
        current_idx = features.bar_idx if hasattr(features, 'bar_idx') else 0
        if current_idx - self._last_trigger_idx < self._cooldown_bars:
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        # Check if trigger.check accepts kwargs
        sig = inspect.signature(self._trigger.check)
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        
        # Check trigger (pass kwargs only if trigger supports them)
        if accepts_kwargs:
            res = self._trigger.check(features, **kwargs)
        else:
            res = self._trigger.check(features)
        
        if res.triggered:
            self._last_trigger_idx = current_idx
            return ScanResult(
                scanner_id=self.scanner_id,
                triggered=True,
                context={
                    "direction": res.direction.value,
                    "trigger_id": res.trigger_id,
                    **res.context
                }
            )
        
        return ScanResult(scanner_id=self.scanner_id, triggered=False)

    def reset(self):
        self._trigger.reset()
        self._last_trigger_idx = -1000

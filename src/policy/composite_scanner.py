"""
Composite Scanner (The Strategy Engine)

This scanner interprets a JSON Recipe to build a dynamic strategy on the fly.
It replaces the need to write custom Python classes for every new strategy idea.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from src.features.state import MarketState
from src.features.pipeline import FeatureBundle
from src.policy.scanners import Scanner, ScanResult
from src.policy.triggers.factory import trigger_from_dict
from src.policy.triggers.base import TriggerResult

@dataclass
class CompositeConfig:
    """
    Configuration for a composed strategy.
    
    Example:
    {
        "name": "My Composed Strategy",
        "entry_trigger": {
            "type": "AND",
            "children": [
                {"type": "ema_cross", "fast": 9, "slow": 21},
                {"type": "rsi_threshold", "threshold": 30, "direction": "lt"}
            ]
        },
        "cooldown_bars": 10
    }
    """
    name: str
    entry_trigger: Dict[str, Any]
    cooldown_bars: int = 10


class CompositeScanner(Scanner):
    """
    A Scanner that executes a dynamic configuration.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = CompositeConfig(
            name=config.get("name", "composite_strategy"),
            entry_trigger=config["entry_trigger"],
            cooldown_bars=config.get("cooldown_bars", 10)
        )
        
        # Build the Trigger Tree
        self._trigger = trigger_from_dict(self.config.entry_trigger)
        
        # State
        self._last_trigger_idx = -1000
    
    @property
    def scanner_id(self) -> str:
        # Use the name from the config as the ID
        # This ensures it shows up nicely in Trade Viz
        return self.config.name.lower().replace(" ", "_")
    
    def scan(
        self,
        state: MarketState,
        features: FeatureBundle
    ) -> ScanResult:
        """
        Evaluate the trigger tree against current market features.
        """
        current_idx = features.bar_idx
        
        # 1. Check Cooldown
        if current_idx - self._last_trigger_idx < self.config.cooldown_bars:
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
            
        # 2. Check Trigger
        res: TriggerResult = self._trigger.check(features)
        
        if res.triggered:
            self._last_trigger_idx = current_idx
            
            return ScanResult(
                scanner_id=self.scanner_id,
                triggered=True,
                context={
                    "direction": res.direction.value,
                    "confidence": res.confidence,
                    **res.context
                },
                score=res.confidence
            )
            
        return ScanResult(scanner_id=self.scanner_id, triggered=False)

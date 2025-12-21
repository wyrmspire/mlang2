"""
Bracket Components (Exit Strategy)

Define stop-loss and take-profit levels for OCO orders.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Tuple
from enum import Enum


class BracketType(Enum):
    ATR = "atr"
    PERCENT = "percent"
    FIXED = "fixed"
    LEVEL = "level"


@dataclass
class BracketLevels:
    """Computed stop and TP prices."""
    stop_price: float
    tp_price: float
    risk_points: float  # Distance from entry to stop
    reward_points: float  # Distance from entry to TP
    r_multiple: float  # reward / risk


class Bracket(ABC):
    """
    Base class for exit bracket strategies.
    
    Brackets compute stop-loss and take-profit prices
    given an entry price, direction, and market context.
    """
    
    @property
    @abstractmethod
    def bracket_type(self) -> BracketType:
        pass
    
    @property
    def params(self) -> Dict[str, Any]:
        """Serializable parameters."""
        return {}
    
    @abstractmethod
    def compute(
        self,
        entry_price: float,
        direction: str,  # "LONG" or "SHORT"
        atr: float,
        **kwargs
    ) -> BracketLevels:
        """
        Compute stop and TP levels.
        
        Args:
            entry_price: Entry price
            direction: Trade direction
            atr: Current ATR value
            **kwargs: Additional context (levels, etc.)
        """
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.bracket_type.value,
            **self.params
        }


class ATRBracket(Bracket):
    """
    Stop and TP as ATR multiples.
    
    Agent config:
        {"type": "atr", "stop_atr": 2.0, "tp_atr": 3.0}
    """
    
    def __init__(self, stop_atr: float = 2.0, tp_atr: float = 3.0):
        self._stop_atr = stop_atr
        self._tp_atr = tp_atr
    
    @property
    def bracket_type(self) -> BracketType:
        return BracketType.ATR
    
    @property
    def params(self) -> Dict[str, Any]:
        return {
            "stop_atr": self._stop_atr,
            "tp_atr": self._tp_atr,
        }
    
    def compute(
        self,
        entry_price: float,
        direction: str,
        atr: float,
        **kwargs
    ) -> BracketLevels:
        risk_points = self._stop_atr * atr
        reward_points = self._tp_atr * atr
        
        if direction.upper() == "LONG":
            stop_price = entry_price - risk_points
            tp_price = entry_price + reward_points
        else:  # SHORT
            stop_price = entry_price + risk_points
            tp_price = entry_price - reward_points
        
        return BracketLevels(
            stop_price=stop_price,
            tp_price=tp_price,
            risk_points=risk_points,
            reward_points=reward_points,
            r_multiple=reward_points / risk_points if risk_points > 0 else 0
        )


class PercentBracket(Bracket):
    """
    Stop and TP as percentage of entry price.
    
    Agent config:
        {"type": "percent", "stop_pct": 0.5, "tp_pct": 1.0}
    """
    
    def __init__(self, stop_pct: float = 0.5, tp_pct: float = 1.0):
        self._stop_pct = stop_pct / 100  # Convert to decimal
        self._tp_pct = tp_pct / 100
    
    @property
    def bracket_type(self) -> BracketType:
        return BracketType.PERCENT
    
    @property
    def params(self) -> Dict[str, Any]:
        return {
            "stop_pct": self._stop_pct * 100,
            "tp_pct": self._tp_pct * 100,
        }
    
    def compute(
        self,
        entry_price: float,
        direction: str,
        atr: float,
        **kwargs
    ) -> BracketLevels:
        risk_points = entry_price * self._stop_pct
        reward_points = entry_price * self._tp_pct
        
        if direction.upper() == "LONG":
            stop_price = entry_price - risk_points
            tp_price = entry_price + reward_points
        else:
            stop_price = entry_price + risk_points
            tp_price = entry_price - reward_points
        
        return BracketLevels(
            stop_price=stop_price,
            tp_price=tp_price,
            risk_points=risk_points,
            reward_points=reward_points,
            r_multiple=reward_points / risk_points if risk_points > 0 else 0
        )


class FixedBracket(Bracket):
    """
    Fixed point stop and TP.
    
    Agent config:
        {"type": "fixed", "stop_points": 5.0, "tp_points": 10.0}
    """
    
    def __init__(self, stop_points: float = 5.0, tp_points: float = 10.0):
        self._stop_points = stop_points
        self._tp_points = tp_points
    
    @property
    def bracket_type(self) -> BracketType:
        return BracketType.FIXED
    
    @property
    def params(self) -> Dict[str, Any]:
        return {
            "stop_points": self._stop_points,
            "tp_points": self._tp_points,
        }
    
    def compute(
        self,
        entry_price: float,
        direction: str,
        atr: float,
        **kwargs
    ) -> BracketLevels:
        if direction.upper() == "LONG":
            stop_price = entry_price - self._stop_points
            tp_price = entry_price + self._tp_points
        else:
            stop_price = entry_price + self._stop_points
            tp_price = entry_price - self._tp_points
        
        return BracketLevels(
            stop_price=stop_price,
            tp_price=tp_price,
            risk_points=self._stop_points,
            reward_points=self._tp_points,
            r_multiple=self._tp_points / self._stop_points if self._stop_points > 0 else 0
        )


class ICTBracket(Bracket):
    """
    ICT-style bracket with PDH/PDL targeting.
    
    Uses pre-computed stop from scanner context (wick-based).
    Targets PDH/PDL if R:R is favorable, otherwise uses min_rr.
    
    Agent config:
        {"type": "ict", "min_rr": 1.5, "use_pdh_pdl": true}
    """
    
    def __init__(self, min_rr: float = 1.5, use_pdh_pdl: bool = True):
        self._min_rr = min_rr
        self._use_pdh_pdl = use_pdh_pdl
    
    @property
    def bracket_type(self) -> BracketType:
        return BracketType.LEVEL
    
    @property
    def params(self) -> Dict[str, Any]:
        return {
            "min_rr": self._min_rr,
            "use_pdh_pdl": self._use_pdh_pdl,
        }
    
    def compute(
        self,
        entry_price: float,
        direction: str,
        atr: float,
        stop_price: float = None,  # Pre-computed from scanner
        pdh: float = None,
        pdl: float = None,
        **kwargs
    ) -> BracketLevels:
        """
        Compute ICT bracket levels.
        
        Stop is expected from scanner context (at penetrating wick).
        TP targets PDH/PDL if favorable R:R, else min_rr.
        """
        # Use pre-computed stop if provided, otherwise fallback to ATR
        if stop_price is not None:
            computed_stop = stop_price
        else:
            # Fallback if no pre-computed stop
            if direction.upper() == "LONG":
                computed_stop = entry_price - (2.0 * atr)
            else:
                computed_stop = entry_price + (2.0 * atr)
        
        risk_points = abs(entry_price - computed_stop)
        min_reward = self._min_rr * risk_points
        
        if direction.upper() == "LONG":
            # Target PDH if favorable
            if self._use_pdh_pdl and pdh and (pdh - entry_price) >= min_reward:
                tp_price = pdh
            else:
                tp_price = entry_price + min_reward
        else:
            # Target PDL if favorable
            if self._use_pdh_pdl and pdl and (entry_price - pdl) >= min_reward:
                tp_price = pdl
            else:
                tp_price = entry_price - min_reward
        
        reward_points = abs(tp_price - entry_price)
        
        return BracketLevels(
            stop_price=computed_stop,
            tp_price=tp_price,
            risk_points=risk_points,
            reward_points=reward_points,
            r_multiple=reward_points / risk_points if risk_points > 0 else 0
        )


# Registry and factory
BRACKET_REGISTRY = {
    "atr": ATRBracket,
    "percent": PercentBracket,
    "fixed": FixedBracket,
    "ict": ICTBracket,
}


def bracket_from_dict(config: dict) -> Bracket:
    """
    Factory function to create bracket from config dict.
    
    Agent-friendly:
        bracket_from_dict({"type": "atr", "stop_atr": 2.0, "tp_atr": 3.0})
    """
    config = config.copy()
    bracket_type = config.pop("type")
    
    if bracket_type not in BRACKET_REGISTRY:
        raise ValueError(f"Unknown bracket type: {bracket_type}. Available: {list(BRACKET_REGISTRY.keys())}")
    
    return BRACKET_REGISTRY[bracket_type](**config)


def list_brackets() -> list:
    """List available bracket types."""
    return list(BRACKET_REGISTRY.keys())

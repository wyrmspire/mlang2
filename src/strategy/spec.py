"""
Declarative Strategy Specification

Replaces ad-hoc strategy scripts with a unified, declarative specification.
Agents create StrategySpec objects instead of writing random Python files.

This ensures:
- Consistent strategy definition
- Reproducible runs (stored in manifest)
- Validation before execution
- No "strategy snowflakes"

Usage:
    from src.strategy.spec import StrategySpec, TriggerConfig, BracketConfig
    
    spec = StrategySpec(
        strategy_id="ema_cross_2025",
        trigger=TriggerConfig(
            type="ema_cross",
            params={"fast": 9, "slow": 21}
        ),
        bracket=BracketConfig(
            type="atr",
            stop_atr=2.0,
            tp_atr=3.0
        ),
        sizing=SizingConfig(
            risk_percent=0.02,
            max_contracts=5
        )
    )
    
    # Stored in manifest
    manifest = {
        'strategy_spec': spec.to_dict(),
        ...
    }
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum
import hashlib
import json


class TriggerType(Enum):
    """Supported trigger types."""
    EMA_CROSS = "ema_cross"
    EMA_BOUNCE = "ema_bounce"
    RSI_THRESHOLD = "rsi_threshold"
    IFVG = "ifvg"
    ORB = "orb"
    CANDLE_PATTERN = "candle_pattern"
    TIME = "time"
    MODEL = "model"  # ML model prediction


class BracketType(Enum):
    """Supported bracket types."""
    ATR = "atr"
    PERCENT = "percent"
    FIXED = "fixed"
    RISK_REWARD = "risk_reward"


class SizingMethod(Enum):
    """Position sizing methods."""
    FIXED_CONTRACTS = "fixed_contracts"
    FIXED_RISK = "fixed_risk"  # % of account
    KELLY = "kelly"
    VOLATILITY_SCALED = "volatility_scaled"


@dataclass
class TriggerConfig:
    """
    Trigger configuration.
    Defines when to enter a trade.
    """
    type: TriggerType
    params: Dict[str, Any] = field(default_factory=dict)
    
    # Optional ML model for prediction
    model_id: Optional[str] = None
    
    # Filters (AND logic)
    filters: List[str] = field(default_factory=list)  # ["session:rth", "time:09:30-15:30"]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type.value if isinstance(self.type, TriggerType) else self.type,
            'params': self.params,
            'model_id': self.model_id,
            'filters': self.filters,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TriggerConfig':
        """Create from dictionary."""
        return cls(
            type=TriggerType(data['type']) if isinstance(data['type'], str) else data['type'],
            params=data.get('params', {}),
            model_id=data.get('model_id'),
            filters=data.get('filters', []),
        )


@dataclass
class BracketConfig:
    """
    Bracket configuration.
    Defines stop loss and take profit.
    """
    type: BracketType
    
    # For ATR brackets
    stop_atr: Optional[float] = None
    tp_atr: Optional[float] = None
    
    # For percent brackets
    stop_percent: Optional[float] = None
    tp_percent: Optional[float] = None
    
    # For fixed brackets
    stop_points: Optional[float] = None
    tp_points: Optional[float] = None
    
    # For risk/reward brackets
    risk_reward_ratio: Optional[float] = None
    
    # Entry type
    entry_type: str = "LIMIT"  # or "MARKET"
    entry_offset_atr: float = 0.25
    
    # Max bars in trade
    max_bars: int = 200
    
    # Reference (for indicator-based levels)
    reference: str = "PRICE"  # or "EMA_5M", "VWAP", etc.
    reference_offset_atr: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type.value if isinstance(self.type, BracketType) else self.type,
            'stop_atr': self.stop_atr,
            'tp_atr': self.tp_atr,
            'stop_percent': self.stop_percent,
            'tp_percent': self.tp_percent,
            'stop_points': self.stop_points,
            'tp_points': self.tp_points,
            'risk_reward_ratio': self.risk_reward_ratio,
            'entry_type': self.entry_type,
            'entry_offset_atr': self.entry_offset_atr,
            'max_bars': self.max_bars,
            'reference': self.reference,
            'reference_offset_atr': self.reference_offset_atr,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BracketConfig':
        """Create from dictionary."""
        return cls(
            type=BracketType(data['type']) if isinstance(data['type'], str) else data['type'],
            stop_atr=data.get('stop_atr'),
            tp_atr=data.get('tp_atr'),
            stop_percent=data.get('stop_percent'),
            tp_percent=data.get('tp_percent'),
            stop_points=data.get('stop_points'),
            tp_points=data.get('tp_points'),
            risk_reward_ratio=data.get('risk_reward_ratio'),
            entry_type=data.get('entry_type', 'LIMIT'),
            entry_offset_atr=data.get('entry_offset_atr', 0.25),
            max_bars=data.get('max_bars', 200),
            reference=data.get('reference', 'PRICE'),
            reference_offset_atr=data.get('reference_offset_atr', 0.0),
        )


@dataclass
class SizingConfig:
    """
    Position sizing configuration.
    Defines how many contracts to trade.
    """
    method: SizingMethod
    
    # For fixed contracts
    contracts: Optional[int] = None
    
    # For risk-based sizing
    risk_percent: Optional[float] = None  # e.g., 0.02 = 2% risk
    max_contracts: Optional[int] = None
    
    # Account size (optional, can be passed at runtime)
    account_size: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'method': self.method.value if isinstance(self.method, SizingMethod) else self.method,
            'contracts': self.contracts,
            'risk_percent': self.risk_percent,
            'max_contracts': self.max_contracts,
            'account_size': self.account_size,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SizingConfig':
        """Create from dictionary."""
        return cls(
            method=SizingMethod(data['method']) if isinstance(data['method'], str) else data['method'],
            contracts=data.get('contracts'),
            risk_percent=data.get('risk_percent'),
            max_contracts=data.get('max_contracts'),
            account_size=data.get('account_size'),
        )


@dataclass
class FilterConfig:
    """
    Entry filter configuration.
    Additional conditions that must be met.
    """
    filter_type: str  # "session", "time", "trend", "volatility"
    params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'filter_type': self.filter_type,
            'params': self.params,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FilterConfig':
        return cls(
            filter_type=data['filter_type'],
            params=data.get('params', {}),
        )


@dataclass
class StrategySpec:
    """
    Complete strategy specification.
    
    This is the declarative definition that replaces ad-hoc scripts.
    Stored in run manifest for reproducibility.
    """
    strategy_id: str
    trigger: TriggerConfig
    bracket: BracketConfig
    sizing: SizingConfig
    
    # Optional components
    filters: List[FilterConfig] = field(default_factory=list)
    indicators: List[str] = field(default_factory=list)  # Indicator IDs to compute
    
    # Metadata
    name: Optional[str] = None
    description: Optional[str] = None
    version: str = "1.0"
    author: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    # Date range
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    
    # Walk-forward settings
    walk_forward: bool = False
    train_weeks: Optional[int] = None
    test_weeks: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'strategy_id': self.strategy_id,
            'trigger': self.trigger.to_dict(),
            'bracket': self.bracket.to_dict(),
            'sizing': self.sizing.to_dict(),
            'filters': [f.to_dict() for f in self.filters],
            'indicators': self.indicators,
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'author': self.author,
            'tags': self.tags,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'walk_forward': self.walk_forward,
            'train_weeks': self.train_weeks,
            'test_weeks': self.test_weeks,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategySpec':
        """Create from dictionary."""
        return cls(
            strategy_id=data['strategy_id'],
            trigger=TriggerConfig.from_dict(data['trigger']),
            bracket=BracketConfig.from_dict(data['bracket']),
            sizing=SizingConfig.from_dict(data['sizing']),
            filters=[FilterConfig.from_dict(f) for f in data.get('filters', [])],
            indicators=data.get('indicators', []),
            name=data.get('name'),
            description=data.get('description'),
            version=data.get('version', '1.0'),
            author=data.get('author'),
            tags=data.get('tags', []),
            start_date=data.get('start_date'),
            end_date=data.get('end_date'),
            walk_forward=data.get('walk_forward', False),
            train_weeks=data.get('train_weeks'),
            test_weeks=data.get('test_weeks'),
        )
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'StrategySpec':
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def fingerprint(self) -> str:
        """
        Generate deterministic fingerprint for the strategy.
        
        Used for:
        - Run identification
        - Deduplication
        - Caching
        
        Returns:
            SHA256 hash of canonical JSON representation
        """
        # Canonical JSON (sorted keys, no whitespace)
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]
    
    def validate(self) -> List[str]:
        """
        Validate strategy specification.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate trigger
        if not self.trigger.type:
            errors.append("Trigger type is required")
        
        # Validate bracket based on type
        if self.bracket.type == BracketType.ATR:
            if self.bracket.stop_atr is None or self.bracket.tp_atr is None:
                errors.append("ATR bracket requires stop_atr and tp_atr")
        elif self.bracket.type == BracketType.PERCENT:
            if self.bracket.stop_percent is None or self.bracket.tp_percent is None:
                errors.append("Percent bracket requires stop_percent and tp_percent")
        elif self.bracket.type == BracketType.FIXED:
            if self.bracket.stop_points is None or self.bracket.tp_points is None:
                errors.append("Fixed bracket requires stop_points and tp_points")
        
        # Validate sizing
        if self.sizing.method == SizingMethod.FIXED_CONTRACTS:
            if self.sizing.contracts is None:
                errors.append("Fixed contracts sizing requires contracts value")
        elif self.sizing.method == SizingMethod.FIXED_RISK:
            if self.sizing.risk_percent is None:
                errors.append("Fixed risk sizing requires risk_percent")
        
        # Validate dates if provided
        if self.start_date and self.end_date:
            if self.start_date >= self.end_date:
                errors.append("start_date must be before end_date")
        
        return errors


# =============================================================================
# Convenience Functions
# =============================================================================

def create_ema_cross_strategy(
    fast: int = 9,
    slow: int = 21,
    stop_atr: float = 2.0,
    tp_atr: float = 3.0,
    risk_percent: float = 0.02,
    **kwargs
) -> StrategySpec:
    """Convenience function to create EMA cross strategy."""
    return StrategySpec(
        strategy_id=f"ema_cross_{fast}_{slow}",
        trigger=TriggerConfig(
            type=TriggerType.EMA_CROSS,
            params={'fast': fast, 'slow': slow}
        ),
        bracket=BracketConfig(
            type=BracketType.ATR,
            stop_atr=stop_atr,
            tp_atr=tp_atr
        ),
        sizing=SizingConfig(
            method=SizingMethod.FIXED_RISK,
            risk_percent=risk_percent
        ),
        **kwargs
    )


def create_ifvg_strategy(
    stop_atr: float = 2.0,
    tp_atr: float = 3.0,
    risk_percent: float = 0.02,
    **kwargs
) -> StrategySpec:
    """Convenience function to create IFVG strategy."""
    return StrategySpec(
        strategy_id="ifvg",
        trigger=TriggerConfig(
            type=TriggerType.IFVG,
            params={}
        ),
        bracket=BracketConfig(
            type=BracketType.ATR,
            stop_atr=stop_atr,
            tp_atr=tp_atr
        ),
        sizing=SizingConfig(
            method=SizingMethod.FIXED_RISK,
            risk_percent=risk_percent
        ),
        **kwargs
    )


def create_model_strategy(
    model_id: str,
    stop_atr: float = 2.0,
    tp_atr: float = 3.0,
    risk_percent: float = 0.02,
    **kwargs
) -> StrategySpec:
    """Convenience function to create ML model strategy."""
    return StrategySpec(
        strategy_id=f"model_{model_id}",
        trigger=TriggerConfig(
            type=TriggerType.MODEL,
            model_id=model_id
        ),
        bracket=BracketConfig(
            type=BracketType.ATR,
            stop_atr=stop_atr,
            tp_atr=tp_atr
        ),
        sizing=SizingConfig(
            method=SizingMethod.FIXED_RISK,
            risk_percent=risk_percent
        ),
        **kwargs
    )

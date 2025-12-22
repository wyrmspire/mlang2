"""
Plugin Registries
Central registration for scanners, models, and indicators.
"""

from typing import Dict, Callable, Any, List, Protocol
from dataclasses import dataclass


# =============================================================================
# Scanner Registry
# =============================================================================

class Scanner(Protocol):
    """Protocol for scanner implementations."""
    def scan(self, step_result: Any) -> bool:
        """Return True if conditions are met."""
        ...


@dataclass
class ScannerInfo:
    """Metadata about a registered scanner."""
    scanner_id: str
    name: str
    description: str
    params_schema: Dict[str, Any]  # JSON schema for params


class ScannerRegistry:
    """
    Registry for scanner implementations.
    
    Usage:
        @ScannerRegistry.register("ema_cross", "EMA Cross", "Trigger on EMA crossover")
        class EMACrossScanner:
            def __init__(self, fast=12, slow=26):
                self.fast = fast
                self.slow = slow
            
            def scan(self, step_result):
                # Implementation
                pass
    """
    
    _registry: Dict[str, Callable] = {}
    _info: Dict[str, ScannerInfo] = {}
    
    @classmethod
    def register(
        cls,
        scanner_id: str,
        name: str,
        description: str = "",
        params_schema: Dict[str, Any] = None
    ):
        """Decorator to register a scanner."""
        def decorator(scanner_class):
            cls._registry[scanner_id] = scanner_class
            cls._info[scanner_id] = ScannerInfo(
                scanner_id=scanner_id,
                name=name,
                description=description,
                params_schema=params_schema or {},
            )
            return scanner_class
        return decorator
    
    @classmethod
    def create(cls, scanner_id: str, **params) -> Scanner:
        """Create scanner instance by ID."""
        if scanner_id not in cls._registry:
            raise ValueError(f"Unknown scanner: {scanner_id}")
        return cls._registry[scanner_id](**params)
    
    @classmethod
    def list_all(cls) -> List[ScannerInfo]:
        """List all registered scanners."""
        return list(cls._info.values())
    
    @classmethod
    def get_info(cls, scanner_id: str) -> ScannerInfo:
        """Get info for a specific scanner."""
        if scanner_id not in cls._info:
            raise ValueError(f"Unknown scanner: {scanner_id}")
        return cls._info[scanner_id]


# =============================================================================
# Model Registry
# =============================================================================

class PolicyModel(Protocol):
    """Protocol for policy model implementations."""
    def predict(self, features: Any) -> Dict[str, Any]:
        """Return model prediction."""
        ...


@dataclass
class ModelInfo:
    """Metadata about a registered model."""
    model_id: str
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]


class ModelRegistry:
    """
    Registry for model implementations.
    
    Usage:
        @ModelRegistry.register("fusion_cnn", "Fusion CNN Model")
        class FusionModelWrapper:
            def __init__(self, model_path):
                self.model = load_model(model_path)
            
            def predict(self, features):
                return self.model.forward(**features)
    """
    
    _registry: Dict[str, Callable] = {}
    _info: Dict[str, ModelInfo] = {}
    
    @classmethod
    def register(
        cls,
        model_id: str,
        name: str,
        description: str = "",
        input_schema: Dict[str, Any] = None,
        output_schema: Dict[str, Any] = None
    ):
        """Decorator to register a model."""
        def decorator(model_class):
            cls._registry[model_id] = model_class
            cls._info[model_id] = ModelInfo(
                model_id=model_id,
                name=name,
                description=description,
                input_schema=input_schema or {},
                output_schema=output_schema or {},
            )
            return model_class
        return decorator
    
    @classmethod
    def create(cls, model_id: str, **params) -> PolicyModel:
        """Create model instance by ID."""
        if model_id not in cls._registry:
            raise ValueError(f"Unknown model: {model_id}")
        return cls._registry[model_id](**params)
    
    @classmethod
    def list_all(cls) -> List[ModelInfo]:
        """List all registered models."""
        return list(cls._info.values())
    
    @classmethod
    def get_info(cls, model_id: str) -> ModelInfo:
        """Get info for a specific model."""
        if model_id not in cls._info:
            raise ValueError(f"Unknown model: {model_id}")
        return cls._info[model_id]


# =============================================================================
# Indicator Registry
# =============================================================================

@dataclass
class IndicatorSeries:
    """
    First-class indicator series for visualization.
    Not hardcoded in chart - generic overlay rendering.
    """
    indicator_id: str
    name: str
    type: str  # 'line', 'histogram', 'band', 'marker'
    points: List[Dict[str, Any]]  # [{time, value}, ...] or specific format per type
    style: Dict[str, Any] = None  # Color, line width, etc.
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'indicator_id': self.indicator_id,
            'name': self.name,
            'type': self.type,
            'points': self.points,
            'style': self.style or {},
        }


@dataclass
class IndicatorInfo:
    """Metadata about a registered indicator."""
    indicator_id: str
    name: str
    description: str
    output_type: str  # 'line', 'histogram', 'band', 'marker'
    params_schema: Dict[str, Any]


class IndicatorRegistry:
    """
    Registry for indicator implementations.
    
    Usage:
        @IndicatorRegistry.register("ema", "EMA", output_type="line")
        class EMAIndicator:
            def __init__(self, period=20):
                self.period = period
            
            def compute(self, stepper) -> IndicatorSeries:
                # Calculate EMA
                return IndicatorSeries(...)
    """
    
    _registry: Dict[str, Callable] = {}
    _info: Dict[str, IndicatorInfo] = {}
    
    @classmethod
    def register(
        cls,
        indicator_id: str,
        name: str,
        output_type: str = "line",
        description: str = "",
        params_schema: Dict[str, Any] = None
    ):
        """Decorator to register an indicator."""
        def decorator(indicator_class):
            cls._registry[indicator_id] = indicator_class
            cls._info[indicator_id] = IndicatorInfo(
                indicator_id=indicator_id,
                name=name,
                description=description,
                output_type=output_type,
                params_schema=params_schema or {},
            )
            return indicator_class
        return decorator
    
    @classmethod
    def create(cls, indicator_id: str, **params):
        """Create indicator instance by ID."""
        if indicator_id not in cls._registry:
            raise ValueError(f"Unknown indicator: {indicator_id}")
        return cls._registry[indicator_id](**params)
    
    @classmethod
    def list_all(cls) -> List[IndicatorInfo]:
        """List all registered indicators."""
        return list(cls._info.values())
    
    @classmethod
    def get_info(cls, indicator_id: str) -> IndicatorInfo:
        """Get info for a specific indicator."""
        if indicator_id not in cls._info:
            raise ValueError(f"Unknown indicator: {indicator_id}")
        return cls._info[indicator_id]
    
    @classmethod
    def compute_all(cls, stepper: Any, indicator_ids: List[str]) -> List[IndicatorSeries]:
        """Compute multiple indicators at once."""
        results = []
        for indicator_id in indicator_ids:
            indicator = cls.create(indicator_id)
            series = indicator.compute(stepper)
            results.append(series)
        return results

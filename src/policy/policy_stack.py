"""
Policy Stack - Multi-Stage Strategy Evaluation

Solves the "Strategy Chaining" problem by enabling:
1. Generator → Filter → Qualifier → Sizer pipeline
2. Configurable evaluator chains
3. Support for filter models (e.g., CNN on losing trades)

This allows complex strategy logic without hardcoding.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Protocol
from enum import Enum
import numpy as np


class EvaluatorStage(Enum):
    """Stages in the policy evaluation pipeline."""
    GENERATOR = "generator"  # Identifies candidate events (Scanner)
    FILTER = "filter"        # Quick pass/fail check (high recall, low precision)
    QUALIFIER = "qualifier"  # Deep evaluation (model inference)
    SIZER = "sizer"          # Determines position size


@dataclass
class EvaluatorResult:
    """
    Result from a single evaluator.
    
    Each stage can pass (continue) or reject (stop pipeline).
    """
    passed: bool
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'passed': self.passed,
            'confidence': self.confidence,
            'metadata': self.metadata,
        }


class Evaluator(Protocol):
    """
    Protocol for policy evaluators.
    
    All stages (Generator, Filter, Qualifier, Sizer) implement this.
    """
    
    stage: EvaluatorStage
    
    def evaluate(self, features: Any, context: Dict[str, Any]) -> EvaluatorResult:
        """
        Evaluate features and return result.
        
        Args:
            features: FeatureBundle from pipeline
            context: Context dict passed through pipeline
            
        Returns:
            EvaluatorResult with pass/fail decision
        """
        ...


@dataclass
class PolicyStackConfig:
    """
    Configuration for a policy stack.
    
    Example JSON:
    {
        "evaluators": [
            {
                "stage": "generator",
                "type": "opening_range_scanner",
                "params": {"minutes": 30}
            },
            {
                "stage": "filter",
                "type": "model",
                "params": {"model_path": "models/loss_filter.pt", "threshold": 0.7}
            },
            {
                "stage": "qualifier",
                "type": "model",
                "params": {"model_path": "models/main_cnn.pt", "threshold": 0.8}
            },
            {
                "stage": "sizer",
                "type": "fixed",
                "params": {"size": 1}
            }
        ]
    }
    """
    evaluators: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {'evaluators': self.evaluators}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PolicyStackConfig':
        return cls(evaluators=data.get('evaluators', []))


@dataclass
class PolicyStackResult:
    """
    Result from running the complete policy stack.
    
    Includes results from each stage and final decision.
    """
    passed: bool
    final_confidence: float
    position_size: float
    stage_results: List[EvaluatorResult] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'passed': self.passed,
            'final_confidence': self.final_confidence,
            'position_size': self.position_size,
            'stage_results': [r.to_dict() for r in self.stage_results],
            'context': self.context,
        }


class PolicyStack:
    """
    Multi-stage policy evaluation pipeline.
    
    Evaluators are run in sequence. If any stage fails, pipeline stops.
    
    Usage:
        # Create stack
        stack = PolicyStack()
        stack.add_evaluator(generator_evaluator)
        stack.add_evaluator(filter_evaluator)
        stack.add_evaluator(qualifier_evaluator)
        stack.add_evaluator(sizer_evaluator)
        
        # Run on features
        result = stack.evaluate(features)
        if result.passed:
            # Execute trade with result.position_size
            pass
    """
    
    def __init__(self):
        """Initialize empty policy stack."""
        self.evaluators: List[Evaluator] = []
    
    def add_evaluator(self, evaluator: Evaluator):
        """
        Add an evaluator to the stack.
        
        Args:
            evaluator: Evaluator implementing the Evaluator protocol
        """
        self.evaluators.append(evaluator)
    
    def evaluate(
        self,
        features: Any,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> PolicyStackResult:
        """
        Run the complete policy stack.
        
        Args:
            features: FeatureBundle from pipeline
            initial_context: Optional initial context dict
            
        Returns:
            PolicyStackResult with final decision and stage results
        """
        context = initial_context or {}
        stage_results = []
        final_confidence = 0.0
        position_size = 1.0  # Default
        
        # Run each evaluator in sequence
        for evaluator in self.evaluators:
            result = evaluator.evaluate(features, context)
            stage_results.append(result)
            
            # Update context with stage metadata
            context[f"{evaluator.stage.value}_metadata"] = result.metadata
            
            # If stage rejects, stop pipeline
            if not result.passed:
                return PolicyStackResult(
                    passed=False,
                    final_confidence=result.confidence,
                    position_size=0.0,
                    stage_results=stage_results,
                    context=context,
                )
            
            # Update confidence (use minimum across stages for conservative approach)
            if final_confidence == 0.0:
                final_confidence = result.confidence
            else:
                final_confidence = min(final_confidence, result.confidence)
            
            # If this is a sizer, capture position size
            if evaluator.stage == EvaluatorStage.SIZER:
                position_size = result.metadata.get('size', 1.0)
        
        # All stages passed
        return PolicyStackResult(
            passed=True,
            final_confidence=final_confidence,
            position_size=position_size,
            stage_results=stage_results,
            context=context,
        )
    
    def reset(self):
        """Reset all evaluators."""
        for evaluator in self.evaluators:
            if hasattr(evaluator, 'reset'):
                evaluator.reset()


# =============================================================================
# Built-in Evaluators
# =============================================================================

class GeneratorEvaluator:
    """
    Generator stage - wraps a Scanner to identify candidate events.
    """
    
    def __init__(self, scanner: Any):
        """
        Initialize generator evaluator.
        
        Args:
            scanner: Scanner instance (from src.policy.scanners)
        """
        self.stage = EvaluatorStage.GENERATOR
        self.scanner = scanner
    
    def evaluate(self, features: Any, context: Dict[str, Any]) -> EvaluatorResult:
        """Check if scanner triggers."""
        result = self.scanner.scan(None, features)
        
        return EvaluatorResult(
            passed=result.triggered,
            confidence=1.0 if result.triggered else 0.0,
            metadata=result.context or {},
        )
    
    def reset(self):
        """Reset scanner state."""
        if hasattr(self.scanner, 'reset'):
            self.scanner.reset()


class ModelEvaluator:
    """
    Filter or Qualifier stage - uses a model for evaluation.
    
    Can be used as:
    - Filter (high recall, low precision): threshold = 0.6
    - Qualifier (high precision): threshold = 0.8
    """
    
    def __init__(
        self,
        model: Any,
        threshold: float = 0.7,
        stage: EvaluatorStage = EvaluatorStage.QUALIFIER
    ):
        """
        Initialize model evaluator.
        
        Args:
            model: PyTorch model
            threshold: Confidence threshold to pass
            stage: FILTER or QUALIFIER
        """
        self.stage = stage
        self.model = model
        self.threshold = threshold
    
    def evaluate(self, features: Any, context: Dict[str, Any]) -> EvaluatorResult:
        """Run model inference and check threshold."""
        import torch
        
        self.model.eval()
        
        with torch.no_grad():
            # Convert to tensors
            x_price_1m = torch.from_numpy(features.x_price_1m).float().unsqueeze(0)
            x_price_5m = torch.from_numpy(features.x_price_5m).float().unsqueeze(0)
            x_price_15m = torch.from_numpy(features.x_price_15m).float().unsqueeze(0)
            x_context = torch.from_numpy(features.x_context).float().unsqueeze(0)
            
            # Transpose for CNN
            x_price_1m = x_price_1m.transpose(1, 2)
            x_price_5m = x_price_5m.transpose(1, 2)
            x_price_15m = x_price_15m.transpose(1, 2)
            
            # Forward pass
            logits = self.model(
                x_price_1m=x_price_1m,
                x_price_5m=x_price_5m,
                x_price_15m=x_price_15m,
                x_context=x_context,
            )
            
            # Get probabilities
            probs = torch.softmax(logits, dim=-1)
            pred_class = torch.argmax(logits, dim=-1).item()
            confidence = probs[0, pred_class].item()
            
            passed = confidence >= self.threshold
            
            return EvaluatorResult(
                passed=passed,
                confidence=confidence,
                metadata={
                    'predicted_class': int(pred_class),
                    'probabilities': probs[0].cpu().numpy().tolist(),
                    'threshold': self.threshold,
                },
            )


class FixedSizeEvaluator:
    """
    Sizer stage - returns fixed position size.
    """
    
    def __init__(self, size: float = 1.0):
        """
        Initialize fixed size evaluator.
        
        Args:
            size: Fixed position size
        """
        self.stage = EvaluatorStage.SIZER
        self.size = size
    
    def evaluate(self, features: Any, context: Dict[str, Any]) -> EvaluatorResult:
        """Always passes with fixed size."""
        return EvaluatorResult(
            passed=True,
            confidence=1.0,
            metadata={'size': self.size},
        )


class ConfidenceBasedSizeEvaluator:
    """
    Sizer stage - scales position size by confidence.
    """
    
    def __init__(self, base_size: float = 1.0, min_confidence: float = 0.5):
        """
        Initialize confidence-based size evaluator.
        
        Args:
            base_size: Maximum position size
            min_confidence: Minimum confidence required
        """
        self.stage = EvaluatorStage.SIZER
        self.base_size = base_size
        self.min_confidence = min_confidence
    
    def evaluate(self, features: Any, context: Dict[str, Any]) -> EvaluatorResult:
        """Scale size by confidence from previous stages."""
        # Get confidence from qualifier stage
        qualifier_meta = context.get('qualifier_metadata', {})
        confidence = qualifier_meta.get('confidence', 0.5)
        
        if confidence < self.min_confidence:
            return EvaluatorResult(
                passed=False,
                confidence=confidence,
                metadata={'reason': 'confidence below minimum'},
            )
        
        # Scale size by confidence
        scaled_size = self.base_size * confidence
        
        return EvaluatorResult(
            passed=True,
            confidence=confidence,
            metadata={'size': scaled_size, 'base_size': self.base_size},
        )

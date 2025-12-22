"""
Model Registry - Dynamic Model Architecture Management

Solves the "Hardcoded Model Architecture" problem by:
1. Saving complete checkpoint bundles (config + manifest + weights)
2. Dynamic model instantiation via ModelFactory
3. Support for multiple architectures (CNN, LSTM, Transformer, etc.)

This allows agents to train different architectures without code changes.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Type, Callable, List
from pathlib import Path
import json
import torch
import torch.nn as nn

from src.core.enums import ModelRole


@dataclass
class FeatureManifest:
    """
    Feature manifest describing what the model expects.
    
    Ensures feature computation matches model training.
    """
    inputs: List[str]  # e.g., ["close_1m", "rsi_14", "ema_200"]
    normalization: str  # e.g., "zscore", "minmax", "none"
    lookback_1m: int = 120
    lookback_5m: int = 24
    lookback_15m: int = 8
    context_dim: int = 20
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'inputs': self.inputs,
            'normalization': self.normalization,
            'lookback_1m': self.lookback_1m,
            'lookback_5m': self.lookback_5m,
            'lookback_15m': self.lookback_15m,
            'context_dim': self.context_dim,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureManifest':
        return cls(
            inputs=data.get('inputs', []),
            normalization=data.get('normalization', 'zscore'),
            lookback_1m=data.get('lookback_1m', 120),
            lookback_5m=data.get('lookback_5m', 24),
            lookback_15m=data.get('lookback_15m', 8),
            context_dim=data.get('context_dim', 20),
        )


@dataclass
class ModelArchitectureConfig:
    """
    Complete model architecture specification.
    
    Allows dynamic instantiation of different model types.
    """
    type: str  # "fusion_cnn", "lstm", "transformer", etc.
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    role: str = "TRAINING_ONLY"  # ModelRole enum value
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type,
            'hyperparameters': self.hyperparameters,
            'role': self.role,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelArchitectureConfig':
        return cls(
            type=data.get('type', 'fusion_cnn'),
            hyperparameters=data.get('hyperparameters', {}),
            role=data.get('role', 'TRAINING_ONLY'),
        )


@dataclass
class ModelCheckpoint:
    """
    Complete model checkpoint bundle.
    
    Contains everything needed to recreate and use a model:
    - Architecture configuration
    - Feature manifest (what inputs it expects)
    - Trained weights
    - Metadata (training info, performance, etc.)
    """
    architecture: ModelArchitectureConfig
    feature_manifest: FeatureManifest
    weights: Dict[str, Any]  # state_dict
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def save(self, path: Path):
        """
        Save complete checkpoint bundle.
        
        Creates:
        - {path}/config.json - Architecture and feature manifest
        - {path}/weights.pt - Model weights
        - {path}/metadata.json - Training metadata
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config = {
            'architecture': self.architecture.to_dict(),
            'feature_manifest': self.feature_manifest.to_dict(),
        }
        with open(path / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save weights
        torch.save(self.weights, path / 'weights.pt')
        
        # Save metadata
        with open(path / 'metadata.json', 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    @classmethod
    def load(cls, path: Path) -> 'ModelCheckpoint':
        """
        Load complete checkpoint bundle.
        
        Args:
            path: Directory containing config.json, weights.pt, metadata.json
            
        Returns:
            ModelCheckpoint with all components
        """
        path = Path(path)
        
        # Load config
        with open(path / 'config.json') as f:
            config = json.load(f)
        
        architecture = ModelArchitectureConfig.from_dict(config['architecture'])
        feature_manifest = FeatureManifest.from_dict(config['feature_manifest'])
        
        # Load weights
        weights = torch.load(path / 'weights.pt', map_location='cpu')
        
        # Load metadata
        metadata = {}
        metadata_path = path / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
        
        return cls(
            architecture=architecture,
            feature_manifest=feature_manifest,
            weights=weights,
            metadata=metadata,
        )


class ModelFactory:
    """
    Factory for creating models dynamically based on configuration.
    
    Supports pluggable architecture types.
    
    Usage:
        # Register a new architecture
        @ModelFactory.register("lstm")
        def create_lstm(config: ModelArchitectureConfig) -> nn.Module:
            return MyLSTM(**config.hyperparameters)
        
        # Create model from config
        model = ModelFactory.create(architecture_config)
    """
    
    _registry: Dict[str, Callable[[ModelArchitectureConfig], nn.Module]] = {}
    
    @classmethod
    def register(cls, model_type: str):
        """
        Decorator to register a model creation function.
        
        Args:
            model_type: String identifier for this model type
            
        Returns:
            Decorator function
        """
        def decorator(create_fn: Callable[[ModelArchitectureConfig], nn.Module]):
            cls._registry[model_type] = create_fn
            return create_fn
        return decorator
    
    @classmethod
    def create(cls, config: ModelArchitectureConfig) -> nn.Module:
        """
        Create model instance from configuration.
        
        Args:
            config: Model architecture configuration
            
        Returns:
            Instantiated model
            
        Raises:
            ValueError: If model type is not registered
        """
        if config.type not in cls._registry:
            raise ValueError(
                f"Unknown model type: {config.type}. "
                f"Registered types: {list(cls._registry.keys())}"
            )
        
        model = cls._registry[config.type](config)
        
        # Set role if model supports it
        if hasattr(model, 'role'):
            model.role = ModelRole[config.role]
        
        return model
    
    @classmethod
    def list_types(cls) -> List[str]:
        """List all registered model types."""
        return list(cls._registry.keys())


def load_model_from_checkpoint(checkpoint_path: Path) -> nn.Module:
    """
    Load a complete model from checkpoint bundle.
    
    This is the recommended way to load models in REPLAY mode.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        
    Returns:
        Instantiated model with loaded weights
        
    Example:
        model = load_model_from_checkpoint("models/my_model")
        manager = SimulationManager(df, model=model, run_mode=RunMode.REPLAY)
    """
    # Load checkpoint bundle
    checkpoint = ModelCheckpoint.load(checkpoint_path)
    
    # Create model from architecture config
    model = ModelFactory.create(checkpoint.architecture)
    
    # Load weights
    model.load_state_dict(checkpoint.weights)
    
    return model


def save_model_checkpoint(
    model: nn.Module,
    save_path: Path,
    architecture_config: ModelArchitectureConfig,
    feature_manifest: FeatureManifest,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Save a complete model checkpoint bundle.
    
    Args:
        model: Trained model to save
        save_path: Directory to save checkpoint
        architecture_config: Model architecture configuration
        feature_manifest: Feature configuration used for training
        metadata: Optional training metadata (metrics, dates, etc.)
        
    Example:
        save_model_checkpoint(
            model=my_model,
            save_path="models/my_model",
            architecture_config=ModelArchitectureConfig(
                type="fusion_cnn",
                hyperparameters={"dropout": 0.3, "num_classes": 2},
                role="REPLAY_ONLY"
            ),
            feature_manifest=FeatureManifest(
                inputs=["close", "rsi_14", "ema_200"],
                normalization="zscore"
            ),
            metadata={
                "train_accuracy": 0.85,
                "val_accuracy": 0.78,
                "trained_on": "2025-01-15"
            }
        )
    """
    checkpoint = ModelCheckpoint(
        architecture=architecture_config,
        feature_manifest=feature_manifest,
        weights=model.state_dict(),
        metadata=metadata or {},
    )
    
    checkpoint.save(save_path)


# =============================================================================
# Register Built-in Model Types
# =============================================================================

@ModelFactory.register("fusion_cnn")
def create_fusion_cnn(config: ModelArchitectureConfig) -> nn.Module:
    """Create FusionModel (CNN + MLP)."""
    from src.models.fusion import FusionModel
    
    params = config.hyperparameters
    role = ModelRole[config.role] if isinstance(config.role, str) else config.role
    
    return FusionModel(
        context_dim=params.get('context_dim', 20),
        price_embedding_per_tf=params.get('price_embedding_per_tf', 32),
        context_embedding=params.get('context_embedding', 32),
        num_classes=params.get('num_classes', 2),
        dropout=params.get('dropout', 0.3),
        role=role,
    )


@ModelFactory.register("lstm")
def create_lstm(config: ModelArchitectureConfig) -> nn.Module:
    """
    Create LSTM model (placeholder for future implementation).
    
    This demonstrates how to add new architectures.
    """
    # TODO: Implement LSTM architecture
    raise NotImplementedError("LSTM architecture not yet implemented")


@ModelFactory.register("transformer")
def create_transformer(config: ModelArchitectureConfig) -> nn.Module:
    """
    Create Transformer model (placeholder for future implementation).
    
    This demonstrates how to add new architectures.
    """
    # TODO: Implement Transformer architecture
    raise NotImplementedError("Transformer architecture not yet implemented")

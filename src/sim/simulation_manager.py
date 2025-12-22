"""
Simulation Manager - Unified Feature Engine

This is the single source of truth for feature computation across all modes:
- TRAIN mode: Uses this to generate training data
- SCAN mode: Uses this to evaluate models on historical data  
- REPLAY mode: Uses this to run interactive simulations

Solves the "Inference Skew" problem by ensuring the exact same
FeaturePipeline logic is used in all contexts.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
import torch

from src.sim.stepper import MarketStepper
from src.features.pipeline import FeaturePipeline, FeatureConfig, FeatureBundle, compute_features
from src.core.enums import RunMode, ModelRole


@dataclass
class SimulationState:
    """Current state of the simulation."""
    bar_idx: int
    timestamp: pd.Timestamp
    current_price: float
    atr: float
    features: FeatureBundle
    model_output: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'bar_idx': self.bar_idx,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'current_price': self.current_price,
            'atr': self.atr,
            'model_output': self.model_output,
        }


class SimulationManager:
    """
    Unified simulation manager that ensures identical feature computation
    across TRAIN, SCAN, and REPLAY modes.
    
    This class wraps:
    - MarketStepper: Causal data access
    - FeaturePipeline: Feature computation
    - Model: Inference (optional)
    
    Usage:
        # Training mode
        manager = SimulationManager(df, run_mode=RunMode.TRAIN)
        for _ in range(num_steps):
            state = manager.next_bar()
            # Use state.features for training
        
        # Replay mode with model
        manager = SimulationManager(
            df, 
            run_mode=RunMode.REPLAY,
            model=my_model,
            feature_config=saved_config
        )
        for _ in range(num_steps):
            state = manager.next_bar()
            # state.model_output contains predictions
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        run_mode: RunMode = RunMode.TRAIN,
        feature_config: Optional[FeatureConfig] = None,
        model: Optional[torch.nn.Module] = None,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
        df_5m: Optional[pd.DataFrame] = None,
        df_15m: Optional[pd.DataFrame] = None,
        df_1h: Optional[pd.DataFrame] = None,
        df_4h: Optional[pd.DataFrame] = None,
        precomputed_indicators: Optional[Dict[int, Any]] = None,
        precomputed_levels: Optional[Dict[int, Any]] = None,
    ):
        """
        Initialize simulation manager.
        
        Args:
            df: 1-minute OHLCV data
            run_mode: TRAIN, SCAN, or REPLAY
            feature_config: Feature computation configuration
            model: Optional model for inference
            start_idx: Starting bar index
            end_idx: Ending bar index (None = end of data)
            df_5m, df_15m, df_1h, df_4h: Higher timeframe data
            precomputed_indicators: Cached indicator values
            precomputed_levels: Cached level values
        """
        self.df = df
        self.run_mode = run_mode
        self.feature_config = feature_config or FeatureConfig()
        self.model = model
        
        # Validate model role
        if self.model is not None and hasattr(self.model, 'role'):
            if run_mode == RunMode.REPLAY and self.model.role == ModelRole.TRAINING_ONLY:
                raise ValueError(
                    f"Cannot use TRAINING_ONLY model in REPLAY mode. "
                    f"This prevents future leakage."
                )
            if run_mode == RunMode.TRAIN and self.model.role == ModelRole.REPLAY_ONLY:
                raise ValueError(
                    f"Cannot use REPLAY_ONLY model in TRAIN mode."
                )
        
        # Initialize stepper
        self.stepper = MarketStepper(
            df=df,
            start_idx=start_idx,
            end_idx=end_idx
        )
        
        # Higher timeframe data
        self.df_5m = df_5m
        self.df_15m = df_15m
        self.df_1h = df_1h
        self.df_4h = df_4h
        
        # Cached computations
        self.precomputed_indicators = precomputed_indicators or {}
        self.precomputed_levels = precomputed_levels or {}
        
        # State tracking
        self.current_state: Optional[SimulationState] = None
        self._step_count = 0
    
    def next_bar(self) -> Optional[SimulationState]:
        """
        Advance to next bar and compute features.
        
        This is the ONLY way to get features in any mode.
        Guarantees identical computation across TRAIN/SCAN/REPLAY.
        
        Returns:
            SimulationState with features and optional model output,
            or None if simulation is complete.
        """
        # Step the market
        if not self.stepper.step():
            return None
        
        # Compute features using the EXACT SAME pipeline
        features = compute_features(
            stepper=self.stepper,
            config=self.feature_config,
            df_5m=self.df_5m,
            df_15m=self.df_15m,
            df_1h=self.df_1h,
            df_4h=self.df_4h,
            precomputed_indicators=self.precomputed_indicators,
            precomputed_levels=self.precomputed_levels,
        )
        
        # Run model inference if provided
        model_output = None
        if self.model is not None:
            model_output = self._run_inference(features)
        
        # Create state
        self.current_state = SimulationState(
            bar_idx=features.bar_idx,
            timestamp=features.timestamp,
            current_price=features.current_price,
            atr=features.atr,
            features=features,
            model_output=model_output,
        )
        
        self._step_count += 1
        return self.current_state
    
    def _run_inference(self, features: FeatureBundle) -> Dict[str, Any]:
        """
        Run model inference on current features.
        
        Args:
            features: FeatureBundle from pipeline
            
        Returns:
            Model output dictionary with predictions
        """
        self.model.eval()
        
        with torch.no_grad():
            # Convert to tensors
            x_price_1m = torch.from_numpy(features.x_price_1m).float().unsqueeze(0)  # (1, channels, length)
            x_price_5m = torch.from_numpy(features.x_price_5m).float().unsqueeze(0)
            x_price_15m = torch.from_numpy(features.x_price_15m).float().unsqueeze(0)
            x_context = torch.from_numpy(features.x_context).float().unsqueeze(0)  # (1, context_dim)
            
            # Transpose for CNN (batch, length, channels) -> (batch, channels, length)
            x_price_1m = x_price_1m.transpose(1, 2)
            x_price_5m = x_price_5m.transpose(1, 2)
            x_price_15m = x_price_15m.transpose(1, 2)
            
            # Forward pass with run_mode enforcement
            if hasattr(self.model, 'check_can_run'):
                self.model.check_can_run(self.run_mode)
            
            logits = self.model(
                x_price_1m=x_price_1m,
                x_price_5m=x_price_5m,
                x_price_15m=x_price_15m,
                x_context=x_context,
                run_mode=self.run_mode if hasattr(self.model, 'forward') else None
            )
            
            # Get probabilities
            probs = torch.softmax(logits, dim=-1)
            pred_class = torch.argmax(logits, dim=-1).item()
            confidence = probs[0, pred_class].item()
            
            return {
                'logits': logits[0].cpu().numpy().tolist(),
                'probabilities': probs[0].cpu().numpy().tolist(),
                'predicted_class': int(pred_class),
                'confidence': float(confidence),
            }
    
    def get_prediction(self) -> Optional[Dict[str, Any]]:
        """
        Get model prediction for current state.
        
        Returns:
            Model output dict or None if no model or no current state.
        """
        if self.current_state is None:
            return None
        return self.current_state.model_output
    
    def get_features(self) -> Optional[FeatureBundle]:
        """
        Get features for current state.
        
        Returns:
            FeatureBundle or None if no current state.
        """
        if self.current_state is None:
            return None
        return self.current_state.features
    
    def reset(self):
        """Reset simulation to beginning."""
        self.stepper.reset()
        self.current_state = None
        self._step_count = 0
    
    def get_progress(self) -> float:
        """Get simulation progress as fraction (0.0 to 1.0)."""
        return self.stepper.progress()
    
    def is_done(self) -> bool:
        """Check if simulation is complete."""
        return self.stepper.is_done()
    
    @property
    def step_count(self) -> int:
        """Number of bars processed."""
        return self._step_count
    
    def get_state_dict(self) -> Dict[str, Any]:
        """
        Get complete state as dictionary.
        
        Returns:
            Dictionary with current state information.
        """
        if self.current_state is None:
            return {
                'initialized': False,
                'step_count': self._step_count,
                'progress': self.get_progress(),
            }
        
        return {
            'initialized': True,
            'step_count': self._step_count,
            'progress': self.get_progress(),
            'current_state': self.current_state.to_dict(),
            'has_model': self.model is not None,
            'run_mode': self.run_mode.value,
        }

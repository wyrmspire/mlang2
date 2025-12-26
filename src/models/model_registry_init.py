"""
Model Registration
Wire existing models into the ModelRegistry.
"""

from src.core.registries import ModelRegistry


# =============================================================================
# Register built-in models
# =============================================================================

@ModelRegistry.register(
    model_id="fusion_cnn",
    name="Fusion CNN Model",
    description="Multi-timeframe CNN with MLP context fusion",
    input_schema={
        "x_price_1m": {"type": "array", "shape": [None, 5]},
        "x_price_5m": {"type": "array", "shape": [None, 5]},
        "x_price_15m": {"type": "array", "shape": [None, 5]},
        "x_context": {"type": "array", "shape": [None]},
    },
    output_schema={
        "logits": {"type": "array", "shape": [3]},
        "probs": {"type": "array", "shape": [3]},
    }
)
class FusionCNNWrapper:
    """Wrapper for FusionModel."""
    def __init__(self, model_path: str):
        from src.models.fusion import FusionModel
        from src.core.enums import ModelRole
        import torch
        
        # Load model
        self.model = FusionModel(role=ModelRole.REPLAY_ONLY)
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
    
    def predict(self, features):
        import torch
        from src.core.enums import RunMode
        
        # Extract features
        x_1m = torch.tensor(features['x_price_1m'], dtype=torch.float32).unsqueeze(0)
        x_5m = torch.tensor(features['x_price_5m'], dtype=torch.float32).unsqueeze(0)
        x_15m = torch.tensor(features['x_price_15m'], dtype=torch.float32).unsqueeze(0)
        x_context = torch.tensor(features['x_context'], dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            logits = self.model(x_1m, x_5m, x_15m, x_context, run_mode=RunMode.REPLAY)
            probs = torch.softmax(logits, dim=1)
        
        return {
            'logits': logits[0].numpy().tolist(),
            'probs': probs[0].numpy().tolist(),
        }


# Auto-register on import
def register_all_models():
    """
    Register all available models.
    Call this at startup to populate the registry.
    """
    pass


@ModelRegistry.register(
    model_id="ifvg_4class",
    name="IFVG 4-Class CNN",
    description="4-class CNN for IFVG pattern detection (LONG_WIN, LONG_LOSS, SHORT_WIN, SHORT_LOSS)",
    input_schema={
        "ohlcv": {"type": "array", "shape": [5, 30], "normalization": "percent_change"},
    },
    output_schema={
        "probs": {"type": "array", "shape": [4]},  # [LONG_WIN, LONG_LOSS, SHORT_WIN, SHORT_LOSS]
    }
)
class IFVG4ClassWrapper:
    """Wrapper for IFVG4ClassCNN."""
    
    def __init__(self, model_path: str = None, **kwargs):
        import torch
        import torch.nn as nn
        
        # Define architecture inline (same as train_ifvg_4class.py)
        class IFVG4ClassCNN(nn.Module):
            def __init__(self, input_channels=5, seq_length=30, num_classes=4):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Conv1d(32, 64, kernel_size=3, padding=1),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Conv1d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(32, num_classes),
                )
            
            def forward(self, x):
                x = self.features(x)
                return self.classifier(x)
        
        self.model = IFVG4ClassCNN(**kwargs)
        if model_path:
            state = torch.load(model_path, map_location='cpu', weights_only=False)
            # Handle both raw state_dict and checkpoint bundles
            if 'model_state_dict' in state:
                self.model.load_state_dict(state['model_state_dict'])
            else:
                self.model.load_state_dict(state)
        self.model.eval()
    
    def predict(self, features):
        import torch
        
        # Expects features['ohlcv'] as (5, 30) normalized array
        x = torch.tensor(features['ohlcv'], dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
        
        probs_list = probs[0].numpy().tolist()
        
        # Determine triggered direction
        long_win, long_loss, short_win, short_loss = probs_list
        
        return {
            'probs': probs_list,
            'long_win_prob': long_win,
            'short_win_prob': short_win,
            'direction': 'LONG' if long_win > short_win else 'SHORT',
            'triggered': True,  # Always true - let caller apply threshold
        }


@ModelRegistry.register(
    model_id="puller_xgb_4class",
    name="Puller XGBoost 4-Class",
    description="XGBoost model for Puller pattern (LONG_WIN, LONG_LOSS, SHORT_WIN, SHORT_LOSS)",
    input_schema={
        "bars": {"type": "array", "description": "OHLCV bars"},
        "ohlcv": {"type": "array", "shape": [5, 30], "description": "Normalized OHLCV"},
    },
    output_schema={
        "probs": {"type": "array", "shape": [4]},
        "direction": {"type": "string"},
        "triggered": {"type": "boolean"},
    }
)
class PullerXGBoostWrapper:
    """Wrapper for Puller XGBoost 4-class model.
    
    Computes features from raw OHLCV bars for inference.
    """
    
    def __init__(self, model_path: str = None, **kwargs):
        import xgboost as xgb
        
        self.model = xgb.XGBClassifier()
        if model_path:
            self.model.load_model(model_path)
        else:
            self.model.load_model('models/puller_xgb_4class.json')
    
    def predict(self, features):
        import numpy as np
        
        # Extract bars or use pre-computed ohlcv
        bars = features.get('bars', [])
        
        # Compute features from bars (pattern indicators that predict win/loss)
        if len(bars) >= 10:
            # Compute pattern-based features from price action
            closes = np.array([b['close'] for b in bars[-30:]])
            highs = np.array([b['high'] for b in bars[-30:]])
            lows = np.array([b['low'] for b in bars[-30:]])
            
            # Feature 1: Recent volatility (normalized range)
            atr = np.mean(highs - lows)
            volatility = atr / closes[-1] if closes[-1] > 0 else 0
            
            # Feature 2: Momentum (close change over last 10 bars)
            momentum = (closes[-1] - closes[-10]) / atr if atr > 0 else 0
            
            # Feature 3: Range position (where close is in recent range)
            range_high = np.max(highs[-20:])
            range_low = np.min(lows[-20:])
            range_pos = (closes[-1] - range_low) / (range_high - range_low) if range_high > range_low else 0.5
            
            x = np.array([volatility * 100, momentum, range_pos], dtype=np.float32).reshape(1, -1)
        else:
            # Fallback: default features
            x = np.array([50, 0, 0.5], dtype=np.float32).reshape(1, -1)
        
        probs = self.model.predict_proba(x)[0].tolist()
        
        # 0=LONG_WIN, 1=LONG_LOSS, 2=SHORT_WIN, 3=SHORT_LOSS
        long_win_prob = probs[0]
        short_win_prob = probs[2]
        
        # Determine direction based on which WIN class has higher prob
        if long_win_prob > short_win_prob:
            direction = 'LONG'
            prob = long_win_prob
        else:
            direction = 'SHORT'
            prob = short_win_prob
        
        return {
            'probs': probs,
            'long_win_prob': long_win_prob,
            'short_win_prob': short_win_prob,
            'direction': direction,
            'triggered': prob >= 0.35,
        }

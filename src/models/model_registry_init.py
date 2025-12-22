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


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

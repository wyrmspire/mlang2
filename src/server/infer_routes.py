"""
Simple Inference Endpoint

POST /infer - Takes price window, runs CNN, returns signal
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/infer", tags=["inference"])

# Cached model
_model = None
_model_path = None


class InferRequest(BaseModel):
    """Request for CNN inference."""
    bars: list  # List of {open, high, low, close, volume}
    model_path: str = "models/ifvg_4class_cnn.pth"
    threshold: float = 0.2


class InferResponse(BaseModel):
    """Response from CNN inference."""
    triggered: bool
    direction: str  # LONG, SHORT, or NONE
    probability: float
    entry_price: float
    stop_price: float
    tp_price: float
    
    
class IFVG4ClassCNN(nn.Module):
    """CNN for 4-class IFVG pattern detection (embedded copy)."""
    
    def __init__(self, input_channels: int = 5, seq_length: int = 30, num_classes: int = 4):
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def load_model(model_path: Path):
    """Load model with auto-detected architecture."""
    global _model, _model_path
    
    if _model is not None and _model_path == str(model_path):
        return _model
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Detect model architecture by looking at layer sizes
    # IFVG4ClassCNN: features.0.weight is (32, 5, 3), classifier.6.weight exists
    # SimpleCNN: features.0.weight is (16, 5, 3), classifier.4.weight exists
    
    if 'features.0.weight' in state_dict:
        first_conv_out = state_dict['features.0.weight'].shape[0]
        
        if first_conv_out == 32 and 'classifier.7.weight' in state_dict:
            # This is IFVG4ClassCNN (8 classifier layers: 0-7)
            num_classes = state_dict['classifier.7.weight'].shape[0]
            model = IFVG4ClassCNN(num_classes=num_classes)
        elif first_conv_out == 16 and 'classifier.4.weight' in state_dict:
            # This is SimpleCNN
            num_classes = state_dict['classifier.4.weight'].shape[0]
            from src.models.fusion import SimpleCNN
            model = SimpleCNN(num_classes=num_classes)
        else:
            raise ValueError(f"Unknown CNN architecture: first_conv_out={first_conv_out}")
    else:
        raise ValueError("Unknown model architecture - no features.0.weight found")
    
    model.load_state_dict(state_dict)
    model.eval()
    
    _model = model
    _model_path = str(model_path)
    
    return model


def normalize_window(ohlcv_array):
    """
    Normalize to match training format.
    
    Training uses:
    - OHLC: Percent change from first close
    - Volume: Max-normalized
    
    Input: (length, 5) array [open, high, low, close, volume]
    Output: (5, length) array normalized
    """
    # Transpose to (5, length) 
    x = ohlcv_array.T.copy()
    
    # Normalize OHLC by first close (percent change)
    first_close = x[3, 0]  # Close of first bar
    if first_close > 0:
        x[0:4] = (x[0:4] - first_close) / first_close * 100
    
    # Normalize volume by max
    max_vol = x[4].max()
    if max_vol > 0:
        x[4] = x[4] / max_vol
    else:
        x[4] = 0
    
    return x  # (5, length)


@router.post("", response_model=InferResponse)
async def infer(request: InferRequest) -> InferResponse:
    """
    Run CNN inference on price window.
    
    Bars should be last 120 1-minute bars (or at least 60).
    Returns whether to trigger a trade and at what levels.
    """
    model_path = Path(request.model_path)
    if not model_path.exists():
        raise HTTPException(404, f"Model not found: {model_path}")
    
    if len(request.bars) < 30:
        raise HTTPException(400, f"Need at least 30 bars, got {len(request.bars)}")
    
    # Load model
    model = load_model(model_path)
    
    # Convert bars to numpy array - use last 30 bars (training used 30)
    bars_to_use = request.bars[-30:] if len(request.bars) >= 30 else request.bars
    bars_array = np.array([
        [b['open'], b['high'], b['low'], b['close'], b.get('volume', 0)]
        for b in bars_to_use
    ])
    
    # Normalize - returns (5, length) already
    x_norm = normalize_window(bars_array)
    
    # To tensor: (1, channels, length)
    x_t = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0)
    
    # For SimpleCNN, just pass the tensor
    # For FusionModel, we'd need multi-timeframe data (not available here)
    with torch.no_grad():
        if hasattr(model, 'features'):
            # SimpleCNN or IFVG4ClassCNN
            logits = model(x_t)
            probs = torch.softmax(logits, dim=-1)
        else:
            # FusionModel - would need more data, fallback
            probs = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
    
    # Interpret output based on num_classes
    # 4-class model: 0=LONG_WIN, 1=LONG_LOSS, 2=SHORT_WIN, 3=SHORT_LOSS
    # 2-class model: 0=LOSS, 1=WIN (original binary)
    num_classes = probs.shape[-1]
    
    current_price = request.bars[-1]['close']
    atr = abs(request.bars[-1]['high'] - request.bars[-1]['low']) * 2  # Simple ATR proxy
    if atr < 0.5: atr = current_price * 0.001  # Fallback
    
    if num_classes == 4:
        # 4-class: 0=LONG_WIN, 1=LONG_LOSS, 2=SHORT_WIN, 3=SHORT_LOSS
        long_win = float(probs[0, 0])
        long_loss = float(probs[0, 1])
        short_win = float(probs[0, 2])
        short_loss = float(probs[0, 3])
        
        # Trigger on the WIN class with highest probability
        if long_win >= request.threshold and long_win > short_win:
            triggered = True
            direction = 'LONG'
            prob = long_win
        elif short_win >= request.threshold:
            triggered = True
            direction = 'SHORT'
            prob = short_win
        else:
            triggered = False
            direction = 'NONE'
            prob = max(long_win, short_win)
    elif num_classes == 2:
        # Binary: 0=LOSS, 1=WIN
        prob = float(probs[0, 1])
        triggered = prob >= request.threshold
        direction = 'LONG' if triggered else 'NONE'  # Assume binary model is Long-only
    else:
        # 3-class or other: assume 0=No, 1=Long, 2=Short
        long_prob = float(probs[0, 1])
        short_prob = float(probs[0, 2]) if num_classes > 2 else 0
        
        if long_prob >= request.threshold and long_prob > short_prob:
            triggered = True
            direction = 'LONG'
            prob = long_prob
        elif short_prob >= request.threshold:
            triggered = True
            direction = 'SHORT'
            prob = short_prob
        else:
            triggered = False
            direction = 'NONE'
            prob = max(long_prob, short_prob)
    
    # Calculate levels
    if direction == 'LONG':
        entry = current_price
        stop = entry - (2 * atr)
        tp = entry + (4 * atr)
    elif direction == 'SHORT':
        entry = current_price
        stop = entry + (2 * atr)
        tp = entry - (4 * atr)
    else:
        entry = current_price
        stop = 0
        tp = 0
    
    return InferResponse(
        triggered=triggered,
        direction=direction,
        probability=round(prob, 4),
        entry_price=round(entry, 2),
        stop_price=round(stop, 2),
        tp_price=round(tp, 2)
    )

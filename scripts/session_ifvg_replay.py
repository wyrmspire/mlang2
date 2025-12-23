#!/usr/bin/env python3
"""
IFVG CNN Replay Runner

Run simulation/replay mode using the trained IFVG CNN.
When the CNN detects a pattern with high confidence, it triggers a trade
using the IFVG entry rules (limit order at FVG midpoint).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import time
import torch
import numpy as np
import pandas as pd
from datetime import timedelta

from src.config import MODELS_DIR, NY_TZ
from src.data.loader import load_continuous_contract
from src.data.resample import resample_all_timeframes
from src.features.fvg import find_fvg
from src.features.swings import find_swings, count_levels_swept
from src.sim.stepper import MarketStepper


# ============================================================================
# MODEL
# ============================================================================

class IFVGPatternCNN(torch.nn.Module):
    """CNN for IFVG pattern detection (must match training architecture)."""
    
    def __init__(self, input_channels: int = 5, seq_length: int = 30, num_classes: int = 2):
        super().__init__()
        
        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(input_channels, 16, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            
            torch.nn.Conv1d(16, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            
            torch.nn.Conv1d(32, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),
        )
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(32, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
    
    def predict_proba(self, x):
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)


def load_ifvg_model(model_path: Path):
    """Load trained IFVG CNN."""
    model = IFVGPatternCNN()
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ============================================================================
# HELPERS
# ============================================================================

def emit_event(event_type: str, data: dict):
    """Emit event as JSON line to stdout."""
    event = {'type': event_type, **data}
    print(json.dumps(event), flush=True)


def normalize_window(ohlcv: np.ndarray) -> np.ndarray:
    """Normalize OHLCV window for CNN input."""
    ohlcv = ohlcv.copy().astype(np.float32)
    
    # Normalize price columns by first close
    first_close = ohlcv[3, 0]
    if first_close > 0:
        ohlcv[0:4] = (ohlcv[0:4] - first_close) / first_close * 100
    
    # Normalize volume by max
    max_vol = ohlcv[4].max() if ohlcv[4].max() > 0 else 1
    ohlcv[4] = ohlcv[4] / max_vol
    
    return ohlcv


def get_price_window(df_1m: pd.DataFrame, bar_idx: int, lookback: int = 30) -> np.ndarray:
    """Extract (5, lookback) price window from 1m data."""
    start_idx = max(0, bar_idx - lookback)
    window = df_1m.iloc[start_idx:bar_idx]
    
    if len(window) < lookback:
        return None
    
    ohlcv = np.array([
        window['open'].values,
        window['high'].values,
        window['low'].values,
        window['close'].values,
        window['volume'].values
    ], dtype=np.float32)
    
    return ohlcv


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run IFVG CNN Replay")
    parser.add_argument("--model", type=str, default="models/ifvg_cnn.pth",
                        help="Path to trained IFVG model")
    parser.add_argument("--start-date", type=str, default="2025-03-18",
                        help="Start date for replay")
    parser.add_argument("--days", type=int, default=1,
                        help="Number of days to replay")
    parser.add_argument("--speed", type=float, default=10.0,
                        help="Speed multiplier")
    parser.add_argument("--threshold", type=float, default=0.65,
                        help="Confidence threshold for triggering")
    
    args = parser.parse_args()
    
    model_path = Path(args.model)
    if not model_path.exists():
        emit_event('ERROR', {'message': f'Model not found: {model_path}'})
        sys.exit(1)
    
    # Load model
    emit_event('STATUS', {'message': 'Loading IFVG CNN model...'})
    model = load_ifvg_model(model_path)
    
    # Load data
    emit_event('STATUS', {'message': 'Loading market data...'})
    df = load_continuous_contract()
    
    start_date = pd.Timestamp(args.start_date, tz=NY_TZ)
    end_date = start_date + timedelta(days=args.days)
    
    df = df[(df['time'] >= start_date) & (df['time'] < end_date)].reset_index(drop=True)
    if len(df) < 60:
        emit_event('ERROR', {'message': f'Not enough data: {len(df)} bars'})
        sys.exit(1)
    
    emit_event('STATUS', {'message': f'Loaded {len(df)} bars'})
    
    # Resample
    htf_data = resample_all_timeframes(df)
    df_5m = htf_data.get('5m')
    
    # Initialize stepper
    stepper = MarketStepper(df, start_idx=30, end_idx=len(df) - 10)
    
    # Emit replay start
    emit_event('REPLAY_START', {
        'start_date': start_date.isoformat(),
        'end_date': end_date.isoformat(),
        'total_bars': len(df),
        'model': str(model_path),
        'strategy': 'IFVG CNN'
    })
    
    bar_delay = 1.0 / args.speed
    decision_count = 0
    trigger_count = 0
    
    # Track recent FVGs for IFVG detection
    recent_fvgs = []
    
    while True:
        step = stepper.step()
        if step.is_done:
            break
        
        bar_idx = step.bar_idx
        current_bar = step.bar
        timestamp = current_bar['time']
        
        # Emit bar update
        emit_event('BAR', {
            'bar_idx': bar_idx,
            'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
            'open': float(current_bar['open']),
            'high': float(current_bar['high']),
            'low': float(current_bar['low']),
            'close': float(current_bar['close']),
            'volume': float(current_bar['volume'])
        })
        
        # Only check every 5 bars for efficiency
        if bar_idx % 5 != 0:
            time.sleep(bar_delay)
            continue
        
        # Get price window
        price_window = get_price_window(df, bar_idx, lookback=30)
        if price_window is None:
            time.sleep(bar_delay)
            continue
        
        # Normalize and convert to tensor
        x = normalize_window(price_window)
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # (1, 5, 30)
        
        # Run CNN inference
        with torch.no_grad():
            probs = model.predict_proba(x_tensor)
            long_prob = float(probs[0, 1])  # P(LONG)
            short_prob = float(probs[0, 0])  # P(SHORT)
        
        # Determine prediction and confidence
        if long_prob > short_prob:
            direction = "LONG"
            confidence = long_prob
        else:
            direction = "SHORT"
            confidence = short_prob
        
        decision_count += 1
        triggered = confidence >= args.threshold
        
        emit_event('DECISION', {
            'decision_id': f'ifvg_cnn_{decision_count:04d}',
            'bar_idx': bar_idx,
            'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
            'direction': direction,
            'confidence': round(confidence, 4),
            'long_prob': round(long_prob, 4),
            'short_prob': round(short_prob, 4),
            'threshold': args.threshold,
            'triggered': triggered,
            'price': float(current_bar['close'])
        })
        
        if triggered:
            trigger_count += 1
            
            # Calculate entry levels using current price and ATR
            atr = (df.iloc[max(0, bar_idx-14):bar_idx]['high'].max() - 
                   df.iloc[max(0, bar_idx-14):bar_idx]['low'].min()) / 3
            
            entry_price = float(current_bar['close'])
            if direction == "LONG":
                stop_price = entry_price - atr
                tp_price = entry_price + (2 * atr)
            else:
                stop_price = entry_price + atr
                tp_price = entry_price - (2 * atr)
            
            emit_event('OCO_OPEN', {
                'decision_id': f'ifvg_cnn_{decision_count:04d}',
                'direction': direction,
                'entry_price': round(entry_price, 2),
                'stop_price': round(stop_price, 2),
                'tp_price': round(tp_price, 2),
                'confidence': round(confidence, 4)
            })
        
        time.sleep(bar_delay)
    
    emit_event('REPLAY_END', {
        'total_bars_processed': decision_count * 5,
        'total_decisions': decision_count,
        'total_triggers': trigger_count,
        'trigger_rate': f'{trigger_count/max(1,decision_count)*100:.1f}%'
    })


if __name__ == "__main__":
    main()

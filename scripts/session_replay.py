"""
Replay Mode Runner

Run trained CNN model on historical data bar-by-bar, emitting events.
Usage: python scripts/session_replay.py --model models/best_model.pth --start-date 2025-03-17 --days 1
"""

import argparse
import json
import sys
import time
import torch
import pandas as pd
from pathlib import Path
from datetime import timedelta

from src.config import RESULTS_DIR, NY_TZ
from src.data.loader import load_continuous_contract
from src.data.resample import resample_all_timeframes
from src.features.indicators import calculate_atr
from src.features.pipeline import compute_features, FeatureConfig
from src.sim.stepper import MarketStepper
from src.models.fusion import FusionModel, SimpleCNN
from src.core.enums import RunMode


def normalize_window(x, method='zscore'):
    """Simple z-score normalization for price windows."""
    import numpy as np
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0) + 1e-8
    return (x - mean) / std



def load_model(model_path: Path):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Determine model type from checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Determine model architecture and num_classes
    num_classes = 2
    # Check for FusionModel classifier (layer 6)
    if 'classifier.6.weight' in state_dict:
        num_classes = state_dict['classifier.6.weight'].shape[0]
    elif 'classifier.6.bias' in state_dict:
        num_classes = state_dict['classifier.6.bias'].shape[0]
    # Check for SimpleCNN classifier (layer 4)
    elif 'classifier.4.weight' in state_dict:
        num_classes = state_dict['classifier.4.weight'].shape[0]
    elif 'classifier.4.bias' in state_dict:
        num_classes = state_dict['classifier.4.bias'].shape[0]
        
    emit_event('STATUS', {'message': f'Detected {num_classes} output classes in model'})

    if any('price_encoder' in k for k in state_dict.keys()):
        model = FusionModel(num_classes=num_classes)
    else:
        model = SimpleCNN(num_classes=num_classes)
    
    model.load_state_dict(state_dict)
    model.eval()
    return model


def emit_event(event_type: str, data: dict):
    """Emit event as JSON line to stdout."""
    event = {
        'type': event_type,
        **data
    }
    print(json.dumps(event), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Run CNN Model Replay")
    parser.add_argument("--model", type=str, default="models/best_model.pth",
                        help="Path to trained model")
    parser.add_argument("--start-date", type=str, default="2025-03-17",
                        help="Start date for replay")
    parser.add_argument("--days", type=int, default=1,
                        help="Number of days to replay")
    parser.add_argument("--speed", type=float, default=10.0,
                        help="Speed multiplier (1.0 = real-time, 10.0 = 10x)")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="Confidence threshold for triggering")
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory for decisions")
    
    args = parser.parse_args()
    
    model_path = Path(args.model)
    if not model_path.exists():
        emit_event('ERROR', {'message': f'Model not found: {model_path}'})
        sys.exit(1)
    
    # Load model
    emit_event('STATUS', {'message': 'Loading model...'})
    model = load_model(model_path)
    
    # Load data
    emit_event('STATUS', {'message': 'Loading market data...'})
    df = load_continuous_contract()
    
    start_date = pd.Timestamp(args.start_date, tz=NY_TZ)
    end_date = start_date + timedelta(days=args.days)
    
    df = df[(df['time'] >= start_date) & (df['time'] < end_date)].reset_index(drop=True)
    if len(df) < 200:
        emit_event('ERROR', {'message': f'Not enough data: {len(df)} bars'})
        sys.exit(1)
    
    emit_event('STATUS', {'message': f'Loaded {len(df)} bars'})
    
    # Resample for higher timeframes
    htf_data = resample_all_timeframes(df)
    df_5m = htf_data.get('5m')
    df_15m = htf_data.get('15m')
    df_5m['atr'] = calculate_atr(df_5m, 14)
    
    # Initialize stepper
    stepper = MarketStepper(df, start_idx=120, end_idx=len(df) - 30)
    feature_config = FeatureConfig(lookback_1m=120)
    
    # Emit replay start
    emit_event('REPLAY_START', {
        'start_date': start_date.isoformat(),
        'end_date': end_date.isoformat(),
        'total_bars': len(df),
        'model': str(model_path)
    })
    
    # Delay between bars based on speed
    bar_delay = 1.0 / args.speed  # seconds per bar
    
    decision_count = 0
    trigger_count = 0
    decisions = []
    
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
        
        # Only run model every 5 bars (reduce noise) 
        if bar_idx % 5 != 0:
            time.sleep(bar_delay)
            continue
        
        # Compute features
        try:
            features = compute_features(stepper, feature_config, df_5m=df_5m, df_15m=df_15m)
        except Exception as e:
            time.sleep(bar_delay)
            continue
        
        # Prepare model inputs
        x_1m = features.x_price_1m
        x_5m = features.x_price_5m
        x_15m = features.x_price_15m
        
        if x_1m is None or len(x_1m) < 60:
            time.sleep(bar_delay)
            continue
        
        # Normalize all timeframes
        x_1m_norm = normalize_window(x_1m, method='zscore')
        x_5m_norm = normalize_window(x_5m, method='zscore') if x_5m is not None and len(x_5m) > 0 else np.zeros((24, 5))
        x_15m_norm = normalize_window(x_15m, method='zscore') if x_15m is not None and len(x_15m) > 0 else np.zeros((8, 5))
        
        # Convert to tensors: (1, channels, length)
        import numpy as np
        x_1m_t = torch.tensor(x_1m_norm.T, dtype=torch.float32).unsqueeze(0)
        x_5m_t = torch.tensor(x_5m_norm.T, dtype=torch.float32).unsqueeze(0)
        x_15m_t = torch.tensor(x_15m_norm.T, dtype=torch.float32).unsqueeze(0)
        
        # Context vector (use indicators if available)
        context_dim = 20
        context = np.zeros(context_dim)
        if features.indicators:
            ind = features.indicators
            context[0] = getattr(ind, 'rsi_1m_14', 50) / 100 if hasattr(ind, 'rsi_1m_14') else 0.5
            context[1] = getattr(ind, 'rsi_5m_14', 50) / 100 if hasattr(ind, 'rsi_5m_14') else 0.5
            context[2] = features.atr / 20 if features.atr else 0
        x_context = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            try:
                if hasattr(model, 'price_encoder'):
                    # FusionModel
                    probs = model.predict_proba(x_1m_t, x_5m_t, x_15m_t, x_context)
                    win_prob = float(probs[0])
                elif hasattr(model, 'features'):
                    # SimpleCNN
                    logits = model(x_1m_t)
                    probs = torch.softmax(logits, dim=-1)
                    win_prob = float(probs[0, 1]) if probs.shape[-1] > 1 else float(probs[0, 0])
                else:
                    win_prob = 0.5
            except Exception as e:
                emit_event('DEBUG', {'error': str(e)})
                win_prob = 0.5
        
        decision_count += 1
        
        # Emit decision event
        triggered = win_prob >= args.threshold
        
        if triggered:
            # Determine direction from class or from heuristic if scalar
            # 4-class: 0=NoSignal, 1=Long, 2=Short, 3=Wait? (Hypothetical)
            # Binary: 0=No, 1=Yes
            
            # For this patch, assume default simple Binary or assume win_prob > thresh means action.
            # Ideally we check model logic. 
            
            # Simple heuristic:
            # If we don't know direction, default to LONG for now or alternating?
            # Actually, IFVG strategy usually implies direction from the pattern.
            # Pure CNN Replay: The model output SHOULD imply direction.
            # If output is single float (win_prob), it usually implies one specific setups (e.g. Long-only model?)
            # Ref: ifvg_4class usually has [Long prob, Short prob, ...]
            
            # Let's inspect probs if available
            direction = 'LONG' 
            if 'probs' in locals():
                if probs.shape[-1] == 3: # [No, Long, Short]
                    if probs[0, 2] > probs[0, 1]: direction = 'SHORT'
                elif probs.shape[-1] == 4: # [No, Long, Short, Other]
                     if probs[0, 2] > probs[0, 1]: direction = 'SHORT'
                # If binary, maybe >0.5 is Long? Or model is Long-Only.
            
            atr = float(features.atr) if features.atr else 0
            if atr == 0: atr = current_bar['close'] * 0.001
            
            entry_price = float(current_bar['close'])
            stop_price = entry_price - (2 * atr) if direction == 'LONG' else entry_price + (2 * atr)
            tp_price = entry_price + (4 * atr) if direction == 'LONG' else entry_price - (4 * atr)

            emit_event('DECISION', {
                'decision_id': f'replay_{decision_count:04d}',
                'bar_idx': bar_idx,
                'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
                'win_probability': round(win_prob, 4),
                'threshold': args.threshold,
                'triggered': True,
                'price': entry_price,
                'atr': atr,
                'direction': direction,
                'stop_price': stop_price,
                'tp_price': tp_price
            })
        else:
            emit_event('DECISION', {
                'decision_id': f'replay_{decision_count:04d}',
                'bar_idx': bar_idx,
                'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
                'win_probability': round(win_prob, 4),
                'threshold': args.threshold,
                'triggered': False,
                'price': float(current_bar['close']),
                'atr': float(features.atr) if features.atr else 0
            })
        
        if triggered:
            trigger_count += 1
            decisions.append({
                'decision_id': f'replay_{decision_count:04d}',
                'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
                'bar_idx': bar_idx,
                'win_probability': win_prob,
                'price': float(current_bar['close'])
            })
        
        # Delay for visualization
        time.sleep(bar_delay)
    
    # Emit replay end
    emit_event('REPLAY_END', {
        'total_bars_processed': decision_count * 5,
        'total_decisions': decision_count,
        'total_triggers': trigger_count,
        'trigger_rate': f'{trigger_count/max(1,decision_count)*100:.1f}%'
    })
    
    # Save decisions if output specified
    if args.out and decisions:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / 'replay_decisions.jsonl', 'w') as f:
            for d in decisions:
                f.write(json.dumps(d) + '\n')
        emit_event('STATUS', {'message': f'Saved {len(decisions)} decisions to {out_dir}'})


if __name__ == "__main__":
    main()

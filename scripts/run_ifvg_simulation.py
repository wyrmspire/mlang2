#!/usr/bin/env python3
"""
IFVG CNN Simulation Runner

Uses the trained 4-class CNN with IFVG Scanner for live simulation.
Pipeline:
1. IFVG Scanner detects setup (FVG inversion + liquidity)
2. CNN predicts class probabilities [LONG_WIN, LONG_LOSS, SHORT_WIN, SHORT_LOSS]
3. If quality threshold met, take the trade with predicted direction
4. Execute with limit order OCO (entry at FVG midpoint, SL at invalidation, TP at 3R)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import timedelta, time as dt_time
from zoneinfo import ZoneInfo

from src.config import MODELS_DIR, NY_TZ
from src.data.loader import load_continuous_contract
from src.data.resample import resample_all_timeframes
from src.sim.stepper import MarketStepper

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = MODELS_DIR / "ifvg_4class_cnn.pth"
MIN_FVG_POINTS = 2.0
INVERSION_WINDOW = 12  # 1 hour on 5m
SL_PADDING = 1.0
RISK_REWARD = 3.0
MIN_WIN_PROB = 0.40  # Minimum P(WIN) to take trade
POINT_VALUE = 50


# ============================================================================
# MODEL
# ============================================================================

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
        return self.classifier(self.features(x))
    
    def predict_proba(self, x):
        return torch.softmax(self.forward(x), dim=-1)


def load_model(path: Path):
    model = IFVG4ClassCNN()
    state_dict = torch.load(path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ============================================================================
# FVG DETECTION (simplified from debug scanner)
# ============================================================================

def find_fvgs(df_5m: pd.DataFrame, min_gap: float = 2.0):
    fvgs = []
    for i in range(1, len(df_5m) - 1):
        prev = df_5m.iloc[i-1]
        curr = df_5m.iloc[i]
        next_ = df_5m.iloc[i+1]
        bar_time = curr['time'] if 'time' in curr else df_5m.index[i]
        
        # Bullish FVG
        bullish_gap = next_['low'] - prev['high']
        if bullish_gap >= min_gap:
            fvgs.append({
                'type': 'BULLISH', 'time': bar_time, 'bar_idx': i,
                'high': next_['low'], 'low': prev['high'],
                'midpoint': (next_['low'] + prev['high']) / 2,
                'gap': bullish_gap
            })
        
        # Bearish FVG
        bearish_gap = prev['low'] - next_['high']
        if bearish_gap >= min_gap:
            fvgs.append({
                'type': 'BEARISH', 'time': bar_time, 'bar_idx': i,
                'high': prev['low'], 'low': next_['high'],
                'midpoint': (prev['low'] + next_['high']) / 2,
                'gap': bearish_gap
            })
    return fvgs


def check_inversion(fvgs, new_idx, window=12):
    """Check if there's an opposite FVG within window."""
    if new_idx >= len(fvgs):
        return None
    new_fvg = fvgs[new_idx]
    opposite = 'BULLISH' if new_fvg['type'] == 'BEARISH' else 'BEARISH'
    
    for i in range(new_idx - 1, -1, -1):
        old_fvg = fvgs[i]
        if old_fvg['type'] == opposite:
            bar_diff = new_fvg['bar_idx'] - old_fvg['bar_idx']
            if bar_diff <= window:
                return old_fvg
            break
    return None


# ============================================================================
# HELPERS
# ============================================================================

def emit_event(event_type: str, data: dict):
    print(json.dumps({'type': event_type, **data}), flush=True)


def normalize_window(ohlcv: np.ndarray) -> np.ndarray:
    ohlcv = ohlcv.copy().astype(np.float32)
    first_close = ohlcv[3, 0]
    if first_close > 0:
        ohlcv[0:4] = (ohlcv[0:4] - first_close) / first_close * 100
    max_vol = ohlcv[4].max() if ohlcv[4].max() > 0 else 1
    ohlcv[4] = ohlcv[4] / max_vol
    return ohlcv


def get_price_window(df_1m: pd.DataFrame, bar_idx: int, lookback: int = 30):
    start_idx = max(0, bar_idx - lookback)
    window = df_1m.iloc[start_idx:bar_idx]
    if len(window) < lookback:
        return None
    return np.array([
        window['open'].values, window['high'].values,
        window['low'].values, window['close'].values,
        window['volume'].values
    ], dtype=np.float32)


def decide_trade(probs):
    """
    Decide whether to trade based on 4-class probabilities.
    
    probs: [P(LONG_WIN), P(LONG_LOSS), P(SHORT_WIN), P(SHORT_LOSS)]
    
    Returns: (take_trade, direction, confidence)
    """
    long_win = probs[0]
    long_loss = probs[1]
    short_win = probs[2]
    short_loss = probs[3]
    
    # Quality = P(WIN | direction)
    long_quality = long_win / (long_win + long_loss + 1e-6)
    short_quality = short_win / (short_win + short_loss + 1e-6)
    
    # Which direction is better?
    if long_win > short_win and long_quality >= MIN_WIN_PROB:
        return True, "LONG", float(long_win), float(long_quality)
    elif short_win > long_win and short_quality >= MIN_WIN_PROB:
        return True, "SHORT", float(short_win), float(short_quality)
    else:
        return False, None, 0, 0


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="IFVG CNN Simulation")
    parser.add_argument("--model", type=str, default=str(MODEL_PATH))
    parser.add_argument("--start-date", type=str, default="2025-03-24")
    parser.add_argument("--days", type=int, default=1)
    parser.add_argument("--speed", type=float, default=10.0)
    parser.add_argument("--min-quality", type=float, default=0.40)
    args = parser.parse_args()
    
    global MIN_WIN_PROB
    MIN_WIN_PROB = args.min_quality
    
    model_path = Path(args.model)
    if not model_path.exists():
        emit_event('ERROR', {'message': f'Model not found: {model_path}'})
        sys.exit(1)
    
    emit_event('STATUS', {'message': 'Loading model...'})
    model = load_model(model_path)
    
    emit_event('STATUS', {'message': 'Loading data...'})
    df_1m = load_continuous_contract()
    
    start_date = pd.Timestamp(args.start_date, tz=NY_TZ)
    end_date = start_date + timedelta(days=args.days)
    
    df_1m = df_1m[(df_1m['time'] >= start_date) & (df_1m['time'] < end_date)].reset_index(drop=True)
    if len(df_1m) < 60:
        emit_event('ERROR', {'message': f'Not enough data: {len(df_1m)} bars'})
        sys.exit(1)
    
    htf = resample_all_timeframes(df_1m)
    df_5m = htf.get('5m')
    if 'time' not in df_5m.columns:
        df_5m = df_5m.reset_index()
    
    emit_event('REPLAY_START', {
        'start_date': start_date.isoformat(),
        'end_date': end_date.isoformat(),
        'total_bars': len(df_1m),
        'model': str(model_path),
        'strategy': 'IFVG 4-Class CNN'
    })
    
    bar_delay = 1.0 / args.speed
    stepper = MarketStepper(df_1m, start_idx=30, end_idx=len(df_1m) - 10)
    
    # Track state
    recent_fvgs = []
    last_fvg_check_5m_idx = -1
    triggers = 0
    cooldown_until = -1
    
    while True:
        step = stepper.step()
        if step.is_done:
            break
        
        bar_idx = step.bar_idx
        current_bar = step.bar
        timestamp = current_bar['time']
        
        emit_event('BAR', {
            'bar_idx': bar_idx,
            'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
            'open': float(current_bar['open']),
            'high': float(current_bar['high']),
            'low': float(current_bar['low']),
            'close': float(current_bar['close']),
            'volume': float(current_bar['volume'])
        })
        
        # Only check on 5m boundaries
        if bar_idx % 5 != 0:
            time.sleep(bar_delay)
            continue
        
        # Cooldown
        if bar_idx < cooldown_until:
            time.sleep(bar_delay)
            continue
        
        # Find 5m index
        current_5m_idx = bar_idx // 5
        if current_5m_idx >= len(df_5m):
            time.sleep(bar_delay)
            continue
        
        # Scan for new FVGs
        df_5m_up_to = df_5m.iloc[:current_5m_idx + 1]
        fvgs = find_fvgs(df_5m_up_to, MIN_FVG_POINTS)
        
        # Check for new FVG
        if len(fvgs) > len(recent_fvgs):
            new_fvg = fvgs[-1]
            recent_fvgs = fvgs[-20:]  # Keep last 20
            
            # Check for inversion
            old_fvg = check_inversion(fvgs, len(fvgs) - 1, INVERSION_WINDOW)
            
            if old_fvg:
                # IFVG detected! Run CNN
                price_window = get_price_window(df_1m, bar_idx, lookback=30)
                
                if price_window is not None:
                    x = normalize_window(price_window)
                    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
                    
                    with torch.no_grad():
                        probs = model.predict_proba(x_tensor)[0].numpy()
                    
                    take_trade, direction, confidence, quality = decide_trade(probs)
                    
                    emit_event('IFVG_DETECTED', {
                        'fvg_type': new_fvg['type'],
                        'fvg_gap': round(new_fvg['gap'], 2),
                        'probs': {
                            'LONG_WIN': round(float(probs[0]), 4),
                            'LONG_LOSS': round(float(probs[1]), 4),
                            'SHORT_WIN': round(float(probs[2]), 4),
                            'SHORT_LOSS': round(float(probs[3]), 4)
                        },
                        'take_trade': take_trade,
                        'direction': direction,
                        'quality': round(quality, 4)
                    })
                    
                    if take_trade:
                        triggers += 1
                        cooldown_until = bar_idx + 30  # 30 min cooldown
                        
                        # Calculate levels
                        entry = new_fvg['midpoint']
                        if direction == "SHORT":
                            stop = new_fvg['high'] + SL_PADDING
                            risk = stop - entry
                            tp = entry - (RISK_REWARD * risk)
                        else:
                            stop = new_fvg['low'] - SL_PADDING
                            risk = entry - stop
                            tp = entry + (RISK_REWARD * risk)
                        
                        # Emit DECISION for UI compatibility (includes OCO levels)
                        emit_event('DECISION', {
                            'decision_id': f'ifvg_sim_{triggers:04d}',
                            'bar_idx': bar_idx,
                            'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
                            'win_probability': quality,
                            'threshold': MIN_WIN_PROB,
                            'triggered': True,
                            'price': entry,
                            'stop_price': round(stop, 2),
                            'tp_price': round(tp, 2),
                            'direction': direction,
                            'atr': risk
                        })
                        
                        emit_event('OCO_OPEN', {
                            'decision_id': f'ifvg_sim_{triggers:04d}',
                            'direction': direction,
                            'entry_price': round(entry, 2),
                            'stop_price': round(stop, 2),
                            'tp_price': round(tp, 2),
                            'confidence': round(confidence, 4),
                            'quality': round(quality, 4),
                            'fvg_gap': round(new_fvg['gap'], 2)
                        })
        
        time.sleep(bar_delay)
    
    emit_event('REPLAY_END', {
        'total_triggers': triggers
    })


if __name__ == "__main__":
    main()

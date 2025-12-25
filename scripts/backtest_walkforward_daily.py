#!/usr/bin/env python3
"""
Walk-Forward Daily Retrain Test

Theory: Hyper-aggressive retraining adapts to regime changes faster.
Method: Retrain model EVERY DAY using a rolling 2-week window.

Compare:
- Static Model: Trained once on first 2 weeks
- Adaptive Model: Retrained daily on rolling 2-week window

Usage:
    python scripts/backtest_walkforward_daily.py --days 7
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple

from src.storage import ExperimentDB


# =============================================================================
# Simple CNN for direction prediction
# =============================================================================

class SimpleCNN(nn.Module):
    """Lightweight CNN for quick retraining."""
    
    def __init__(self, lookback: int = 30, features: int = 5):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(features, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(32 * 4, 16),
            nn.ReLU(),
            nn.Linear(16, 2),  # UP or DOWN
        )
    
    def forward(self, x):
        # x: (batch, seq, features) -> (batch, features, seq)
        x = x.permute(0, 2, 1)
        return self.fc(self.conv(x))


# =============================================================================
# Dataset from raw OHLCV
# =============================================================================

class DirectionDataset(Dataset):
    """Simple next-bar direction prediction dataset."""
    
    def __init__(self, df: pd.DataFrame, lookback: int = 30, lookahead: int = 5):
        self.samples = []
        self.labels = []
        
        for i in range(lookback, len(df) - lookahead):
            # Window
            window = df.iloc[i-lookback:i][['open', 'high', 'low', 'close', 'volume']].values
            
            # Normalize
            window = self._normalize(window)
            
            # Label: price direction over lookahead
            current_close = df.iloc[i]['close']
            future_close = df.iloc[i + lookahead]['close']
            label = 0 if future_close > current_close else 1  # 0 = UP, 1 = DOWN
            
            self.samples.append(window)
            self.labels.append(label)
    
    def _normalize(self, data):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0) + 1e-8
        return (data - mean) / std
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.samples[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


# =============================================================================
# Training helper
# =============================================================================

def quick_train(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 5,
    lr: float = 0.001,
    device: str = 'cuda',
) -> nn.Module:
    """Quick training for daily retrain."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
    
    return model


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = 'cuda',
) -> float:
    """Evaluate accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            _, pred = out.max(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    
    return correct / total if total > 0 else 0


# =============================================================================
# Walk-Forward Engine
# =============================================================================

def run_walkforward_daily(days: int = 7, train_days: int = 5) -> Dict[str, Any]:
    """
    Run walk-forward with daily retraining.
    
    Due to yfinance 1m limit (7 days), we simulate with available data.
    """
    print("=" * 60)
    print("WALK-FORWARD DAILY RETRAIN TEST")
    print("=" * 60)
    print(f"Comparing: Static vs Daily-Retrain models")
    print(f"Training window: {train_days} days (rolling)")
    print("=" * 60)
    
    # Load data
    actual_days = min(days, 7)
    end = datetime.now()
    start = end - timedelta(days=actual_days)
    
    print(f"\n[1] Loading {actual_days} days of ES data...")
    ticker = yf.Ticker("ES=F")
    df = ticker.history(start=start, end=end, interval="1m")
    
    if df is None or len(df) == 0:
        print("ERROR: No data!")
        return {}
    
    df.columns = [c.lower() for c in df.columns]
    df = df.reset_index()
    df['time'] = pd.to_datetime(df['Datetime'] if 'Datetime' in df.columns else df['datetime'])
    
    print(f"    Loaded {len(df)} bars")
    
    # Split by dates
    df['date'] = df['time'].dt.date
    unique_dates = sorted(df['date'].unique())
    
    if len(unique_dates) < train_days + 2:
        print(f"ERROR: Need at least {train_days + 2} days, have {len(unique_dates)}")
        return {}
    
    print(f"    {len(unique_dates)} trading days")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"    Device: {device}")
    
    # =========================================================================
    # Static Model: Train once on first train_days
    # =========================================================================
    print(f"\n[2] Training STATIC model on first {train_days} days...")
    
    train_dates = unique_dates[:train_days]
    train_df = df[df['date'].isin(train_dates)]
    
    train_ds = DirectionDataset(train_df)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    
    static_model = SimpleCNN()
    static_model = quick_train(static_model, train_loader, epochs=10, device=device)
    
    print(f"    Trained on {len(train_ds)} samples")
    
    # =========================================================================
    # Walk-Forward: Test each remaining day
    # =========================================================================
    print(f"\n[3] Walk-forward testing...")
    
    static_results = []
    adaptive_results = []
    
    test_dates = unique_dates[train_days:]
    
    for i, test_date in enumerate(test_dates):
        test_df = df[df['date'] == test_date]
        if len(test_df) < 40:
            continue
        
        test_ds = DirectionDataset(test_df)
        test_loader = DataLoader(test_ds, batch_size=32)
        
        # Static model accuracy
        static_acc = evaluate(static_model, test_loader, device)
        static_results.append(static_acc)
        
        # Adaptive: Retrain on rolling window up to this day
        rolling_end_idx = unique_dates.index(test_date)
        rolling_start_idx = max(0, rolling_end_idx - train_days)
        rolling_dates = unique_dates[rolling_start_idx:rolling_end_idx]
        
        rolling_df = df[df['date'].isin(rolling_dates)]
        if len(rolling_df) > 100:
            rolling_ds = DirectionDataset(rolling_df)
            rolling_loader = DataLoader(rolling_ds, batch_size=32, shuffle=True)
            
            adaptive_model = SimpleCNN()
            adaptive_model = quick_train(adaptive_model, rolling_loader, epochs=5, device=device)
            
            adaptive_acc = evaluate(adaptive_model, test_loader, device)
        else:
            adaptive_acc = static_acc  # Fallback if not enough data
        
        adaptive_results.append(adaptive_acc)
        
        print(f"    Day {i+1} ({test_date}): Static={static_acc:.1%}, Adaptive={adaptive_acc:.1%}")
    
    # Summary
    avg_static = np.mean(static_results) if static_results else 0
    avg_adaptive = np.mean(adaptive_results) if adaptive_results else 0
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Test days: {len(test_dates)}")
    print(f"  Static Model Avg Accuracy:   {avg_static:.1%}")
    print(f"  Adaptive Model Avg Accuracy: {avg_adaptive:.1%}")
    
    diff = (avg_adaptive - avg_static) * 100
    if avg_adaptive > avg_static:
        print(f"\n  ✓ Daily retraining wins by {diff:.1f}pp!")
        print(f"  → Hyper-aggressive adaptation DOES help")
    else:
        print(f"\n  ✗ Static model wins by {-diff:.1f}pp")
        print(f"  → Constant retraining adds noise, doesn't help")
    
    # Store
    db = ExperimentDB()
    run_id = f"walkforward_daily_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    db.store_run(
        run_id=run_id,
        strategy="walkforward_daily",
        config={
            'train_days': train_days,
            'test_days': len(test_dates),
            'retrain_frequency': 'daily',
        },
        metrics={
            'total_trades': len(test_dates),
            'wins': int(len(test_dates) * avg_adaptive),
            'losses': int(len(test_dates) * (1 - avg_adaptive)),
            'win_rate': avg_adaptive,
            'static_win_rate': avg_static,
            'adaptive_win_rate': avg_adaptive,
            'total_pnl': 0,
        }
    )
    print(f"\n[+] Stored: {run_id}")
    
    return {
        'static_acc': avg_static,
        'adaptive_acc': avg_adaptive,
        'improvement': diff,
        'test_days': len(test_dates),
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Walk-Forward Daily Retrain")
    parser.add_argument("--days", type=int, default=7, help="Total days")
    parser.add_argument("--train-days", type=int, default=4, help="Training window")
    
    args = parser.parse_args()
    
    results = run_walkforward_daily(args.days, args.train_days)

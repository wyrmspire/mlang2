#!/usr/bin/env python3
"""
Multi-Timeframe Fusion Model

Theory: Only trade if higher timeframe (1H) agrees with entry direction.
Input: 30 bars of 1m data + 5 bars of 1H data
Filter: If 1H is bullish (close > open over window), only take LONGS.

Usage:
    python scripts/train_fusion_mtf.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
from torch.utils.data import Dataset, DataLoader, random_split
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple

from src.storage import ExperimentDB


# =============================================================================
# Multi-Timeframe Fusion Model
# =============================================================================

class MTFFusionModel(nn.Module):
    """
    Fusion model that processes 1m and 1H data separately, then combines.
    
    Architecture:
    - 1m Branch: CNN for short-term patterns (30 bars × 5 features)
    - 1H Branch: MLP for trend context (5 bars × 5 features)
    - Fusion: Concatenate + FC layers
    """
    
    def __init__(
        self,
        bars_1m: int = 30,
        bars_1h: int = 5,
        num_features: int = 5,  # OHLCV
        num_classes: int = 2,   # LONG or SHORT
    ):
        super().__init__()
        
        self.bars_1m = bars_1m
        self.bars_1h = bars_1h
        
        # 1-Minute Branch (CNN)
        self.cnn_1m = nn.Sequential(
            nn.Conv1d(num_features, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
        )
        cnn_out_size = 64 * 4  # 256
        
        # 1-Hour Branch (MLP)
        self.mlp_1h = nn.Sequential(
            nn.Flatten(),
            nn.Linear(bars_1h * num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        mlp_out_size = 16
        
        # Fusion Head
        self.fusion = nn.Sequential(
            nn.Linear(cnn_out_size + mlp_out_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )
    
    def forward(self, x_1m, x_1h):
        """
        Args:
            x_1m: (batch, bars_1m, features) - 1-minute data
            x_1h: (batch, bars_1h, features) - 1-hour data
        """
        # CNN expects (batch, channels, seq_len)
        x_1m = x_1m.permute(0, 2, 1)
        
        # Process each branch
        feat_1m = self.cnn_1m(x_1m)      # (batch, 256)
        feat_1h = self.mlp_1h(x_1h)       # (batch, 16)
        
        # Fuse
        fused = torch.cat([feat_1m, feat_1h], dim=1)
        
        return self.fusion(fused)
    
    def get_1h_trend(self, x_1h):
        """Determine if 1H trend is bullish (returns True/False per sample)."""
        # Simple: compare first close to last close
        # x_1h: (batch, bars, features) where features = [O, H, L, C, V]
        first_close = x_1h[:, 0, 3]  # First bar close
        last_close = x_1h[:, -1, 3]  # Last bar close
        return last_close > first_close  # Bullish if rising


# =============================================================================
# Dataset with MTF data
# =============================================================================

class MTFDataset(Dataset):
    """Generate multi-timeframe samples from market data."""
    
    def __init__(
        self,
        df_1m: pd.DataFrame,
        df_1h: pd.DataFrame,
        bars_1m: int = 30,
        bars_1h: int = 5,
    ):
        self.samples_1m = []
        self.samples_1h = []
        self.labels = []
        self.bars_1m = bars_1m
        self.bars_1h = bars_1h
        
        # Align 1m and 1h data
        df_1m = df_1m.copy()
        df_1h = df_1h.copy()
        
        df_1m['hour'] = df_1m['time'].dt.floor('h')
        
        # For each valid point, create a sample
        for i in range(bars_1m + 60, len(df_1m) - 10):  # Leave room for outcome
            current_time = df_1m.iloc[i]['time']
            current_hour = current_time.floor('h')
            
            # Get 1m window
            window_1m = df_1m.iloc[i-bars_1m:i][['open', 'high', 'low', 'close', 'volume']].values
            
            # Get 1h window (last 5 hours before current)
            hourly_mask = df_1h['time'] < current_hour
            hourly_data = df_1h[hourly_mask].tail(bars_1h)
            
            if len(hourly_data) < bars_1h:
                continue
            
            window_1h = hourly_data[['open', 'high', 'low', 'close', 'volume']].values
            
            # Normalize each window separately (Z-score)
            window_1m = self._normalize(window_1m)
            window_1h = self._normalize(window_1h)
            
            # Simple label: next 10 bars go up = LONG(0), down = SHORT(1)
            future_close = df_1m.iloc[i + 10]['close']
            current_close = df_1m.iloc[i]['close']
            label = 0 if future_close > current_close else 1
            
            self.samples_1m.append(window_1m)
            self.samples_1h.append(window_1h)
            self.labels.append(label)
        
        print(f"MTF Dataset: {len(self.samples_1m)} samples")
        print(f"  LONG (up): {sum(1 for l in self.labels if l == 0)}")
        print(f"  SHORT (down): {sum(1 for l in self.labels if l == 1)}")
    
    def _normalize(self, data):
        """Z-score normalize."""
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0) + 1e-8
        return (data - mean) / std
    
    def __len__(self):
        return len(self.samples_1m)
    
    def __getitem__(self, idx):
        x_1m = torch.tensor(self.samples_1m[idx], dtype=torch.float32)
        x_1h = torch.tensor(self.samples_1h[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x_1m, x_1h, y


# =============================================================================
# Training with Trend Filter
# =============================================================================

def train_fusion_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 30,
    lr: float = 0.001,
    device: str = 'cuda',
    use_trend_filter: bool = True,
) -> Tuple[dict, float, Dict]:
    """
    Train fusion model with optional 1H trend filter.
    
    If use_trend_filter=True:
    - Only count LONG predictions as correct if 1H is bullish
    - Only count SHORT predictions as correct if 1H is bearish
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    best_state = None
    stats = {'filtered_trades': 0, 'total_trades': 0}
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for x_1m, x_1h, y in train_loader:
            x_1m, x_1h, y = x_1m.to(device), x_1h.to(device), y.to(device)
            
            optimizer.zero_grad()
            out = model(x_1m, x_1h)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validate with trend filter
        model.eval()
        correct_unfiltered = 0
        correct_filtered = 0
        total = 0
        filtered_trades = 0
        
        with torch.no_grad():
            for x_1m, x_1h, y in val_loader:
                x_1m, x_1h, y = x_1m.to(device), x_1h.to(device), y.to(device)
                
                out = model(x_1m, x_1h)
                _, pred = out.max(1)
                
                # Get 1H trend
                trend_bullish = model.get_1h_trend(x_1h)
                
                for i in range(len(pred)):
                    total += 1
                    
                    # Unfiltered accuracy
                    if pred[i] == y[i]:
                        correct_unfiltered += 1
                    
                    # Filtered: only count if direction matches trend
                    if use_trend_filter:
                        is_long = pred[i] == 0
                        should_take = (is_long and trend_bullish[i]) or (not is_long and not trend_bullish[i])
                        
                        if should_take:
                            filtered_trades += 1
                            if pred[i] == y[i]:
                                correct_filtered += 1
                    else:
                        filtered_trades += 1
                        if pred[i] == y[i]:
                            correct_filtered += 1
        
        acc_unfiltered = correct_unfiltered / total if total > 0 else 0
        acc_filtered = correct_filtered / filtered_trades if filtered_trades > 0 else 0
        
        if acc_filtered > best_acc:
            best_acc = acc_filtered
            best_state = model.state_dict().copy()
            stats = {
                'filtered_trades': filtered_trades,
                'total_trades': total,
                'unfiltered_acc': acc_unfiltered,
                'filtered_acc': acc_filtered,
            }
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f} "
                  f"- Val Acc: {acc_unfiltered:.1%} (unfiltered) / {acc_filtered:.1%} (filtered)")
    
    return best_state, best_acc, stats


# =============================================================================
# Main
# =============================================================================

def run_fusion_comparison(days: int = 7) -> Dict[str, Any]:
    """
    Train fusion model and compare filtered vs unfiltered expectancy.
    """
    print("=" * 60)
    print("MULTI-TIMEFRAME FUSION MODEL")
    print("=" * 60)
    print("Input: 30 bars (1m) + 5 bars (1H)")
    print("Filter: Only trade if 1H trend agrees with direction")
    print("=" * 60)
    
    # Load data
    actual_days = min(days, 7)
    end = datetime.now()
    start = end - timedelta(days=actual_days)
    
    print(f"\n[1] Loading {actual_days} days of ES data...")
    ticker = yf.Ticker("ES=F")
    
    df_1m = ticker.history(start=start, end=end, interval="1m")
    df_1h = ticker.history(start=start - timedelta(days=30), end=end, interval="1h")
    
    if df_1m is None or len(df_1m) == 0:
        print("ERROR: No data!")
        return {}
    
    # Standardize
    for df in [df_1m, df_1h]:
        df.columns = [c.lower() for c in df.columns]
    
    df_1m = df_1m.reset_index()
    df_1h = df_1h.reset_index()
    df_1m['time'] = pd.to_datetime(df_1m['Datetime'] if 'Datetime' in df_1m.columns else df_1m['datetime'])
    df_1h['time'] = pd.to_datetime(df_1h['Datetime'] if 'Datetime' in df_1h.columns else df_1h['datetime'])
    
    print(f"    1m: {len(df_1m)} bars")
    print(f"    1h: {len(df_1h)} bars")
    
    # Create dataset
    print(f"\n[2] Creating MTF dataset...")
    dataset = MTFDataset(df_1m, df_1h, bars_1m=30, bars_1h=5)
    
    if len(dataset) < 50:
        print("ERROR: Not enough samples")
        return {}
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    
    # Train model
    print(f"\n[3] Training Fusion Model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"    Device: {device}")
    
    model = MTFFusionModel(bars_1m=30, bars_1h=5, num_classes=2)
    
    best_state, best_acc, stats = train_fusion_model(
        model, train_loader, val_loader,
        epochs=30, lr=0.001, device=device,
        use_trend_filter=True
    )
    
    # Save
    model_path = Path("models/mtf_fusion.pth")
    model_path.parent.mkdir(exist_ok=True)
    torch.save(best_state, model_path)
    
    # Results
    print("\n" + "=" * 60)
    print("RESULTS: TREND FILTER IMPACT")
    print("=" * 60)
    print(f"  Unfiltered Accuracy: {stats['unfiltered_acc']:.1%}")
    print(f"  Filtered Accuracy:   {stats['filtered_acc']:.1%}")
    print(f"  Trades Taken:        {stats['filtered_trades']}/{stats['total_trades']} "
          f"({100*stats['filtered_trades']/stats['total_trades']:.0f}%)")
    
    improvement = (stats['filtered_acc'] - stats['unfiltered_acc']) * 100
    if improvement > 0:
        print(f"\n  ✓ Filter IMPROVED expectancy by {improvement:.1f} percentage points!")
    else:
        print(f"\n  ✗ Filter did not improve expectancy ({improvement:.1f}pp)")
    
    # Store
    db = ExperimentDB()
    run_id = f"mtf_fusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    db.store_run(
        run_id=run_id,
        strategy="mtf_fusion",
        config={
            'bars_1m': 30,
            'bars_1h': 5,
            'use_trend_filter': True,
        },
        metrics={
            'total_trades': stats['total_trades'],
            'filtered_trades': stats['filtered_trades'],
            'wins': int(stats['filtered_trades'] * stats['filtered_acc']),
            'losses': int(stats['filtered_trades'] * (1 - stats['filtered_acc'])),
            'win_rate': stats['filtered_acc'],
            'unfiltered_win_rate': stats['unfiltered_acc'],
            'total_pnl': 0,
        },
        model_path=str(model_path)
    )
    print(f"\n[+] Saved to ExperimentDB: {run_id}")
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MTF Fusion Model")
    parser.add_argument("--days", type=int, default=7, help="Days to train on")
    
    args = parser.parse_args()
    
    results = run_fusion_comparison(args.days)

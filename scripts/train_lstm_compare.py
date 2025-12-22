#!/usr/bin/env python3
"""
LSTM vs CNN Comparison

Engineer theory: CNN focuses on shapes, LSTM captures price flow/sequence.
Test: Train an LSTM on 60-bar close price sequences, compare to baseline CNN.

Usage:
    python scripts/train_lstm_compare.py --input results/ict_ifvg/records.jsonl
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from datetime import datetime
from typing import Dict, Any, List, Tuple

from src.storage import ExperimentDB
from src.features.engine import normalize_ohlcv_window, FeatureConfig


# =============================================================================
# LSTM Model Architecture
# =============================================================================

class PriceLSTM(nn.Module):
    """
    LSTM for price sequence classification.
    
    Input: (batch, seq_len, features)
    Output: (batch, num_classes)
    """
    def __init__(
        self,
        input_size: int = 1,      # Just close prices
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 4,     # LONG_WIN, LONG_LOSS, SHORT_WIN, SHORT_LOSS
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,  # Causal - only look back
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = h_n[-1]  # (batch, hidden_size)
        
        return self.fc(last_hidden)


# =============================================================================
# Dataset
# =============================================================================

class LSTMDataset(Dataset):
    """Dataset for LSTM training using close price sequences."""
    
    LABEL_MAP = {
        'LONG_WIN': 0, 'LONG_LOSS': 1,
        'SHORT_WIN': 2, 'SHORT_LOSS': 3,
    }
    
    def __init__(self, records: List[Dict], lookback: int = 60):
        self.samples = []
        self.labels = []
        self.lookback = lookback
        
        for rec in records:
            # Get label
            direction = rec.get('direction', 'LONG')
            outcome = rec.get('label', rec.get('outcome', 'LOSS'))
            label_str = f"{direction}_{outcome}"
            
            if label_str not in self.LABEL_MAP:
                continue
            
            # Get close prices from window
            window = rec.get('window', {})
            ohlcv = window.get('raw_ohlcv_1m', [])
            
            if len(ohlcv) < lookback:
                continue
            
            # Extract close prices - handle both dict and array format
            closes = []
            for bar in ohlcv[-lookback:]:
                if isinstance(bar, dict):
                    closes.append(bar.get('close', bar.get('Close', 0)))
                else:
                    closes.append(bar[3])  # Index 3 = close in array format
            
            closes = np.array(closes, dtype=np.float32)
            
            # Normalize: Z-score
            mean = np.mean(closes)
            std = np.std(closes) + 1e-8
            closes_norm = (closes - mean) / std
            
            self.samples.append(closes_norm)
            self.labels.append(self.LABEL_MAP[label_str])
        
        print(f"Dataset: {len(self.samples)} samples")
        for label, idx in self.LABEL_MAP.items():
            count = sum(1 for l in self.labels if l == idx)
            print(f"  {label} ({idx}): {count}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx], dtype=torch.float32).unsqueeze(-1)  # (seq, 1)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


# =============================================================================
# Training
# =============================================================================

def train_lstm(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 0.001,
    device: str = 'cuda',
) -> Tuple[dict, float]:
    """Train LSTM and return best state dict and accuracy."""
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    best_state = None
    
    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validate
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                _, pred = out.max(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        acc = correct / total if total > 0 else 0
        
        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f} - Val Acc: {acc:.1%}")
    
    return best_state, best_acc


# =============================================================================
# Main
# =============================================================================

def run_comparison(input_path: str, lookback: int = 60) -> Dict[str, Any]:
    """
    Train LSTM and compare to CNN baseline.
    """
    print("=" * 60)
    print("LSTM vs CNN COMPARISON")
    print("=" * 60)
    print(f"Lookback: {lookback} bars (close prices only)")
    print("=" * 60)
    
    # Load records
    print(f"\n[1] Loading records from {input_path}...")
    records = []
    with open(input_path) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"    Loaded {len(records)} records")
    
    # Create dataset
    print(f"\n[2] Creating LSTM dataset...")
    dataset = LSTMDataset(records, lookback=lookback)
    
    if len(dataset) < 20:
        print("ERROR: Not enough samples for training")
        return {'lstm_acc': 0, 'cnn_acc': 0}
    
    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)
    
    print(f"    Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Train LSTM
    print(f"\n[3] Training LSTM...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"    Device: {device}")
    
    lstm_model = PriceLSTM(
        input_size=1,
        hidden_size=64,
        num_layers=2,
        num_classes=4,
    )
    
    lstm_state, lstm_acc = train_lstm(
        lstm_model, train_loader, val_loader,
        epochs=50, lr=0.001, device=device
    )
    
    # Save LSTM
    lstm_path = Path("models/lstm_price_seq.pth")
    lstm_path.parent.mkdir(exist_ok=True)
    torch.save(lstm_state, lstm_path)
    print(f"\n    LSTM saved to {lstm_path}")
    print(f"    LSTM Val Accuracy: {lstm_acc:.1%}")
    
    # Load CNN baseline if exists
    cnn_acc = 0.0
    cnn_path = Path("models/ifvg_4class_cnn.pth")
    if cnn_path.exists():
        print(f"\n[4] Loading CNN baseline from {cnn_path}...")
        # We'd need to evaluate CNN on same data - for now use stored metrics
        db = ExperimentDB()
        cnn_runs = db.query_best("win_rate", strategy="cnn_training", top_k=1)
        if cnn_runs:
            cnn_acc = cnn_runs[0].get('win_rate', 0)
            print(f"    CNN baseline accuracy: {cnn_acc:.1%}")
    
    # Compare
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    print(f"  LSTM Accuracy: {lstm_acc:.1%}")
    print(f"  CNN Accuracy:  {cnn_acc:.1%}")
    
    if lstm_acc > cnn_acc:
        diff = (lstm_acc - cnn_acc) * 100
        print(f"\n  ✓ LSTM wins by {diff:.1f} percentage points!")
        print("  -> Sequence/flow matters more than shape for this data")
    elif cnn_acc > lstm_acc:
        diff = (cnn_acc - lstm_acc) * 100
        print(f"\n  ✓ CNN wins by {diff:.1f} percentage points!")
        print("  -> Shape patterns matter more than sequence for this data")
    else:
        print(f"\n  = TIE - both models perform similarly")
    
    # Store LSTM result
    db = ExperimentDB()
    run_id = f"lstm_vs_cnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    db.store_run(
        run_id=run_id,
        strategy="lstm_comparison",
        config={
            'lookback': lookback,
            'hidden_size': 64,
            'num_layers': 2,
            'architecture': 'PriceLSTM',
        },
        metrics={
            'total_trades': len(dataset),
            'wins': int(len(dataset) * lstm_acc),
            'losses': int(len(dataset) * (1 - lstm_acc)),
            'win_rate': lstm_acc,
            'total_pnl': 0,
        },
        model_path=str(lstm_path)
    )
    print(f"\n[+] Saved to ExperimentDB: {run_id}")
    
    return {
        'lstm_acc': lstm_acc,
        'cnn_acc': cnn_acc,
        'lstm_path': str(lstm_path),
        'samples': len(dataset),
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LSTM vs CNN Comparison")
    parser.add_argument("--input", type=str, default="results/ict_ifvg/records.jsonl")
    parser.add_argument("--lookback", type=int, default=60, help="Sequence length")
    
    args = parser.parse_args()
    
    results = run_comparison(args.input, args.lookback)

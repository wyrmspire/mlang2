#!/usr/bin/env python3
"""
Train 4-Class IFVG CNN

Train a CNN with 4 classes to predict both direction AND outcome:
- Class 0: LONG_WIN
- Class 1: LONG_LOSS  
- Class 2: SHORT_WIN
- Class 3: SHORT_LOSS

This allows the model to:
1. Skip low-quality setups (neither direction looks like a winner)
2. Pick the direction that has higher win probability
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from typing import List, Dict

from src.config import MODELS_DIR

# ============================================================================
# CONFIGURATION
# ============================================================================

RECORDS_FILE = Path("results/ifvg_debug/records.jsonl")
MODEL_OUT = MODELS_DIR / "ifvg_4class_cnn.pth"
LOOKBACK_BARS = 30
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 0.001


# ============================================================================
# MODEL
# ============================================================================

class IFVG4ClassCNN(nn.Module):
    """
    CNN for 4-class IFVG pattern detection.
    
    Input: (batch, 5, 30) - OHLCV channels
    Output: (batch, 4) - [P(LONG_WIN), P(LONG_LOSS), P(SHORT_WIN), P(SHORT_LOSS)]
    """
    
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
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)


# ============================================================================
# DATASET
# ============================================================================

def get_label(direction: str, outcome: str) -> int:
    """Map direction + outcome to 4-class label."""
    if direction == "LONG" and outcome == "WIN":
        return 0  # LONG_WIN
    elif direction == "LONG" and outcome == "LOSS":
        return 1  # LONG_LOSS
    elif direction == "SHORT" and outcome == "WIN":
        return 2  # SHORT_WIN
    elif direction == "SHORT" and outcome == "LOSS":
        return 3  # SHORT_LOSS
    else:
        return -1  # Skip (timeout, not filled)


class IFVG4ClassDataset(Dataset):
    """Dataset for 4-class IFVG training."""
    
    def __init__(self, records: List[Dict], lookback: int = 30):
        self.samples = []
        self.labels = []
        
        for record in records:
            oco = record.get('oco', {})
            oco_results = record.get('oco_results', {})
            
            direction = oco.get('direction', '')
            outcome = oco_results.get('outcome', '')
            
            # Get label
            label = get_label(direction, outcome)
            if label == -1:
                continue  # Skip timeouts/not filled
            
            # Get price window
            window_data = record.get('window', {})
            raw_ohlcv = window_data.get('raw_ohlcv_1m', [])
            
            if len(raw_ohlcv) < lookback:
                continue
            
            # Take first lookback bars (history before entry)
            pre_trade = raw_ohlcv[:lookback]
            
            # Convert to numpy array: (5, lookback)
            ohlcv = np.array([
                [b['open'] for b in pre_trade],
                [b['high'] for b in pre_trade],
                [b['low'] for b in pre_trade],
                [b['close'] for b in pre_trade],
                [b.get('volume', 0) for b in pre_trade]
            ], dtype=np.float32)
            
            # Normalize prices by first close (percent change)
            first_close = ohlcv[3, 0]
            if first_close > 0:
                ohlcv[0:4] = (ohlcv[0:4] - first_close) / first_close * 100
            
            # Normalize volume
            max_vol = ohlcv[4].max()
            if max_vol > 0:
                ohlcv[4] = ohlcv[4] / max_vol
            
            self.samples.append(ohlcv)
            self.labels.append(label)
        
        # Print class distribution
        from collections import Counter
        counts = Counter(self.labels)
        print(f"Dataset: {len(self.samples)} samples")
        print(f"  LONG_WIN (0): {counts[0]}")
        print(f"  LONG_LOSS (1): {counts[1]}")
        print(f"  SHORT_WIN (2): {counts[2]}")
        print(f"  SHORT_LOSS (3): {counts[3]}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, train_loader, val_loader, epochs, lr, device):
    model = model.to(device)
    
    # Use weighted loss for class imbalance
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_acc = 0
    best_state = None
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = logits.argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        
        val_acc = correct / max(1, total)
        scheduler.step(1 - val_acc)
        
        if epoch % 5 == 0 or val_acc > best_val_acc:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss/len(train_loader):.4f} - Val Acc: {val_acc:.2%}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    return best_state, best_val_acc


def main():
    print("=" * 60)
    print("Train 4-Class IFVG CNN")
    print("=" * 60)
    
    # Load records
    print("\n[1] Loading records...")
    if not RECORDS_FILE.exists():
        print(f"Error: {RECORDS_FILE} not found")
        sys.exit(1)
    
    records = []
    with open(RECORDS_FILE) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} records")
    
    # Create dataset
    print("\n[2] Creating dataset...")
    dataset = IFVG4ClassDataset(records, lookback=LOOKBACK_BARS)
    
    if len(dataset) < 20:
        print("Error: Not enough samples")
        sys.exit(1)
    
    # Split 80/20
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Train
    print("\n[3] Training...")
    model = IFVG4ClassCNN()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    best_state, best_acc = train_model(model, train_loader, val_loader, EPOCHS, LEARNING_RATE, device)
    
    # Save
    print("\n[4] Saving model...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, MODEL_OUT)
    print(f"Saved to: {MODEL_OUT}")
    print(f"Best validation accuracy: {best_acc:.2%}")
    
    # Save model info
    info = {
        "architecture": "IFVG4ClassCNN",
        "input_shape": [5, LOOKBACK_BARS],
        "num_classes": 4,
        "classes": ["LONG_WIN", "LONG_LOSS", "SHORT_WIN", "SHORT_LOSS"],
        "best_val_accuracy": best_acc,
        "training_samples": len(train_ds),
        "epochs": EPOCHS
    }
    with open(MODEL_OUT.with_suffix('.json'), 'w') as f:
        json.dump(info, f, indent=2)


if __name__ == "__main__":
    main()

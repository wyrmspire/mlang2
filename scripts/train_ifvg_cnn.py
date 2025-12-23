#!/usr/bin/env python3
"""
Train IFVG CNN

Train a CNN to recognize pre-trade patterns from successful IFVG trades.
Uses 30 1m candles before each trade, labeled by direction (LONG/SHORT).
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
from typing import List, Dict, Any

from src.data.loader import load_continuous_contract
from src.config import MODELS_DIR, NY_TZ

# ============================================================================
# CONFIGURATION
# ============================================================================

RECORDS_FILE = Path("results/ict_ifvg/records.jsonl")
MODEL_OUT = MODELS_DIR / "ifvg_cnn.pth"
LOOKBACK_BARS = 30  # 30 1m candles before entry
EPOCHS = 30
BATCH_SIZE = 8
LEARNING_RATE = 0.001


# ============================================================================
# MODEL
# ============================================================================

class IFVGPatternCNN(nn.Module):
    """
    Simple CNN for IFVG pattern detection.
    
    Input: (batch, 5, 30) - 5 channels (OHLCV), 30 time steps
    Output: (batch, 2) - probability of LONG vs SHORT
    """
    
    def __init__(self, input_channels: int = 5, seq_length: int = 30, num_classes: int = 2):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 30 -> 15
            
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 15 -> 7
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # 7 -> 1
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, length)
        Returns:
            Logits (batch, num_classes)
        """
        x = self.features(x)
        return self.classifier(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability of each class."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)


# ============================================================================
# DATASET
# ============================================================================

class IFVGTradeDataset(Dataset):
    """Dataset of pre-trade price patterns from IFVG records."""
    
    def __init__(self, records: List[Dict], df_1m, lookback: int = 30):
        self.samples = []
        self.labels = []
        
        for record in records:
            # Only use filled winning trades for training
            oco_results = record.get('oco_results', {})
            if not oco_results.get('filled', False):
                continue
            if oco_results.get('outcome') not in ('WIN', 'LOSS'):
                continue
            
            # Get direction label: LONG=1, SHORT=0
            oco = record.get('oco', {})
            direction = oco.get('direction', 'LONG')
            label = 1 if direction == 'LONG' else 0
            
            # Extract price window from raw_ohlcv_1m
            window_data = record.get('window', {})
            raw_ohlcv = window_data.get('raw_ohlcv_1m', [])
            
            if len(raw_ohlcv) < lookback:
                continue
            
            # Take last `lookback` bars before trade (first bars in window are history)
            # The window should have history before entry
            pre_trade = raw_ohlcv[:lookback]
            
            # Convert to numpy array: (lookback, 5) -> (5, lookback)
            ohlcv = np.array([
                [b['open'] for b in pre_trade],
                [b['high'] for b in pre_trade],
                [b['low'] for b in pre_trade],
                [b['close'] for b in pre_trade],
                [b.get('volume', 0) for b in pre_trade]
            ], dtype=np.float32)
            
            # Normalize price columns (0-3) by first close
            first_close = ohlcv[3, 0]
            if first_close > 0:
                ohlcv[0:4] = (ohlcv[0:4] - first_close) / first_close * 100  # Percent change
            
            # Normalize volume by max
            max_vol = ohlcv[4].max()
            if max_vol > 0:
                ohlcv[4] = ohlcv[4] / max_vol
            
            self.samples.append(ohlcv)
            self.labels.append(label)
        
        print(f"Created dataset with {len(self.samples)} samples")
        print(f"  LONG: {sum(self.labels)}, SHORT: {len(self.labels) - sum(self.labels)}")
    
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
    """Train the model and return best weights."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
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
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss/len(train_loader):.4f} - Val Acc: {val_acc:.2%}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
    
    return best_state, best_val_acc


def main():
    print("=" * 60)
    print("Train IFVG Pattern CNN")
    print("=" * 60)
    
    # Load records
    print("\n[1] Loading IFVG trade records...")
    if not RECORDS_FILE.exists():
        print(f"Error: {RECORDS_FILE} not found. Run backtest_ict_ifvg.py first.")
        sys.exit(1)
    
    records = []
    with open(RECORDS_FILE) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} records")
    
    # Load full 1m data (for potential augmentation)
    print("\n[2] Loading market data...")
    df_1m = load_continuous_contract()
    print(f"Loaded {len(df_1m)} 1m bars")
    
    # Create dataset
    print("\n[3] Creating dataset...")
    dataset = IFVGTradeDataset(records, df_1m, lookback=LOOKBACK_BARS)
    
    if len(dataset) < 4:
        print("Error: Not enough samples to train. Need more trades.")
        sys.exit(1)
    
    # Split
    train_size = max(1, int(0.7 * len(dataset)))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Create model
    print("\n[4] Training...")
    model = IFVGPatternCNN(input_channels=5, seq_length=LOOKBACK_BARS, num_classes=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    best_state, best_acc = train_model(model, train_loader, val_loader, EPOCHS, LEARNING_RATE, device)
    
    # Save
    print("\n[5] Saving model...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, MODEL_OUT)
    print(f"Saved to: {MODEL_OUT}")
    print(f"Best validation accuracy: {best_acc:.2%}")


if __name__ == "__main__":
    main()

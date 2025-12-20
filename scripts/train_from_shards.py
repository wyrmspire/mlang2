"""
Train From Shards
Train a FusionModel on an existing sharded dataset.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader, random_split

from src.datasets.reader import DecisionDataset
from src.models.fusion import FusionModel
from src.models.train import train_model, TrainConfig, ImbalanceStrategy
from src.config import MODELS_DIR

def main():
    parser = argparse.ArgumentParser(description="Train Model from Shards")
    parser.add_argument("--data", type=str, required=True, help="Path to shard directory (e.g. shards/swing_breakout_v1)")
    parser.add_argument("--out", type=str, default="swing_breakout_model.pth", help="Output model filename")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    
    args = parser.parse_args()
    
    shard_dir = Path(args.data)
    model_out = MODELS_DIR / args.out
    
    print("=" * 60)
    print(f"Training on: {shard_dir}")
    print(f"Output to:   {model_out}")
    print("=" * 60)
    
    # 1. Load Data
    print("\n[1] Loading dataset...")
    full_dataset = DecisionDataset(shard_dir)
    print(f"Total records: {len(full_dataset)}")
    
    if len(full_dataset) < 10:
        print("Error: Not enough data to train.")
        sys.exit(1)
    
    # 2. Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch)
    
    print(f"Train size: {len(train_ds)}")
    print(f"Val size:   {len(val_ds)}")
    
    # 3. Configure
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=0.001,
        save_path=model_out,
        imbalance_strategy=ImbalanceStrategy.WEIGHTED_LOSS
    )
    
    # 4. Create Model
    # Note: We need to know the context dim. Default is 20 in schema code.
    # Ideally we read this from schema/manifest.
    model = FusionModel(
        context_dim=20, 
        num_classes=2,
        dropout=0.3
    )
    
    # 5. Train
    print("\n[2] Training...")
    result = train_model(model, train_loader, val_loader, config)
    
    print("\n[3] Results")
    print(f"Best Val Loss: {result.best_val_loss:.4f} (Epoch {result.best_epoch})")
    print(f"Model saved to: {model_out}")

if __name__ == "__main__":
    main()

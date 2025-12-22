# Local Stuff (Not in Git)

This document describes files and directories that exist locally but are **NOT tracked by Git**.

---

## Models (`/models/`)

PyTorch model checkpoints. Ignored because they're large binary files.

| File | Size | Description |
|------|------|-------------|
| `ifvg_4class_cnn.pth` | ~180KB | 4-class IFVG pattern CNN (LONG_WIN, LONG_LOSS, SHORT_WIN, SHORT_LOSS) |
| `ifvg_4class_cnn.json` | ~284B | Training metadata for above model |
| `ifvg_cnn.pth` | ~50KB | Original IFVG CNN (binary) |
| `cnn_filter.pth` | ~59KB | CNN filter model |
| `best_model.pth` | ~1.2MB | Best model from walk-forward testing |
| `swing_breakout_model.pth` | ~1.2MB | Swing breakout strategy model |

**To train new models:**
```bash
python scripts/train_ifvg_4class.py
```

---

## Data (`/data/`)

### `/data/raw/`
Raw market data downloaded from broker/exchange.

| File | Size | Format |
|------|------|--------|
| `continuous_contract.json` | ~32MB | 1-minute OHLCV bars, ~2 years |

**Structure:**
```json
{
  "bars": [
    {"time": "2023-01-03T09:30:00.000Z", "open": 3824.25, "high": 3825.5, "low": 3823.0, "close": 3824.75, "volume": 12345},
    ...
  ]
}
```

### `/data/processed/`
Preprocessed data (normalized, feature-engineered). Regenerated from raw.

---

## Results (`/results/`)

Strategy backtest results, scans, trade records.

```
results/
├── ifvg_debug/
│   └── records.jsonl           # Decision records for training
├── mean_rev_prod/
│   └── trades.json             # Mean reversion strategy results
└── [strategy_name]/
    ├── scan_results.json
    └── trades.json
```

---

## Shards (`/shards/`)

Dataset shards for ML training (chunked decision records).

---

## Cache (`/cache/`)

Runtime caches (feature computations, aggregations).

---

## Logs (`/logs/`, `*.log`)

Application logs, not committed.

---

## How to Get Local Data

1. **Market Data**: Contact your broker API or use the data ingestion script
2. **Models**: Train from records or download pre-trained from shared storage
3. **Results**: Re-run strategies/scans to regenerate

---

## .gitignore Quick Reference

```gitignore
# ML artifacts
*.pth
*.pt
/models/

# Data
data/raw/
data/processed/
results/
shards/
cache/
```

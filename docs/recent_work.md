# Recent Work: Real-Time CNN Simulation

*Last Updated: December 21, 2024*

---

## Overview

Implemented real-time CNN inference during chart playback simulation, allowing a trained model to trigger trades bar-by-bar.

---

## What Was Built

### 1. `/infer` Endpoint (`src/server/infer_routes.py`)
- **POST /infer** - Takes price window (30 bars), runs CNN, returns trade signal
- Auto-detects model architecture (IFVG4ClassCNN vs SimpleCNN)
- Normalization matches training: percent-change from first close for OHLC, max-normalized volume

### 2. Trade Settings Panel (`SimulationView.tsx`)
Configurable trade parameters:
- **CNN Threshold** (0.35 default) - Higher = fewer triggers
- **Stop Loss** (ATR ×) - Default 2.0
- **Take Profit** (ATR ×) - Default 4.0
- **Entry Type** - Market (Limit future)

### 3. Direction-Aware OCO
- LONG: stop below entry, TP above
- SHORT: stop above entry, TP below
- PnL calculated correctly for both directions

---

## Bugs Fixed

| Bug | Cause | Fix |
|-----|-------|-----|
| Trades appear in past | Async closure captured stale `bar` | Use `allBarsRef.current[currentIdx]` when promise resolves |
| Model over-triggering | Threshold 0.2 below random chance for 4-class | Raised to 0.35, made configurable |
| Stale settings | Missing useCallback deps | Added `stopAtr`, `tpAtr`, `threshold` |
| Model 500 error | Wrong classifier layer detection | Fixed to `classifier.7.weight` |
| No SHORT trades | Model training bias | Code correct; model needs more SHORT training data |

---

## Data Flow

```
Chart Playback (setInterval)
    │
    ├─▶ Every 5 bars: POST /infer with last 30 bars
    │       │
    │       └─▶ Backend: Normalize → CNN → Softmax → Threshold
    │               │
    │               └─▶ Response: {triggered, direction, probability}
    │
    └─▶ Frontend: If triggered && no active trade
            │
            └─▶ Create OCO with frontend settings (stopAtr, tpAtr)
                    │
                    └─▶ Draw entry/stop/tp lines on chart
```

---

## Key Files Changed

| File | Changes |
|------|---------|
| `src/server/infer_routes.py` | New endpoint, embedded IFVG4ClassCNN, correct normalization |
| `src/components/SimulationView.tsx` | Real-time inference, trade settings, async timing fix |
| `src/server/main.py` | Registered /infer router |

---

## Model Classes

The 4-class IFVG CNN outputs:
- **Class 0**: LONG_WIN → Trigger LONG
- **Class 1**: LONG_LOSS → No trigger
- **Class 2**: SHORT_WIN → Trigger SHORT
- **Class 3**: SHORT_LOSS → No trigger

Threshold applies to WIN probability (0.35 = 35% confidence required).

---

## Future Work

1. **Model Registry** - Save architecture info in checkpoint for auto-loading
2. **Training Dashboard** - View model performance metrics
3. **Multi-Model** - Run multiple models, vote on signals
4. **Scanners** - Pre-analyze price action to select order type (limit vs market)

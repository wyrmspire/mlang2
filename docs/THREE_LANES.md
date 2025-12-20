# Three Lanes Architecture

## Overview

MLang2 now has a **three lanes** architecture that makes the different execution modes explicit in both the UI and API.

## The Three Lanes

### 🔬 TRAIN Lane
**Purpose**: Training data analysis and model development

**Features**:
- View historical training runs
- Analyze decision distributions  
- Inspect counterfactual labels
- Review model outputs and predictions

**What you can do**:
- Browse decision points
- See what would have happened (counterfactual outcomes)
- Analyze trade-off between different policies
- NO live execution, NO future peeking in model

**Run Mode**: `RunMode.TRAIN`

---

### ▶️ REPLAY Lane
**Purpose**: Real-time simulation and playback

**Features**:
- Simulate model execution bar-by-bar
- Replay existing strategies with controls (play/pause/step)
- View OCO zones as they evolve
- Track simulated account state

**What you can do**:
- Step through historical data as if it were live
- See model decisions in real-time
- Pause and inspect at any moment
- NO training, NO future peeking

**Run Mode**: `RunMode.REPLAY`

---

### 🔍 SCAN Lane
**Purpose**: Read-only analysis and strategy discovery

**Features**:
- Analyze patterns in historical data
- Chat with agent to explore signals
- Run new strategies (creates data, switches to TRAIN mode)
- Pure analysis - no live trading, no learning

**What you can do**:
- Explore scanner signals
- Ask agent to generate strategies
- Analyze market patterns
- NO trading, NO learning - just discovery

**Run Mode**: `RunMode.SCAN`

---

## Key Concepts

### Trigger → Decision → Trade

- **Trigger**: Raw condition (time, candle pattern, indicator threshold)
  - Example: "10:00 AM" or "Hammer candle" or "RSI < 30"
  
- **Decision**: "We would consider a trade here" (scanner output)
  - Includes: scanner context, action (PLACE_ORDER/NO_TRADE), skip reason
  
- **Trade**: Simulated execution result (OCO lifecycle + fills + outcome)
  - Includes: entry/exit prices, PnL, MAE/MFE, bars held

### Contracts & Registries

**RunManifest**: Unified contract for all run outputs
- Tells UI what a run contains
- What mode it was in (TRAIN/REPLAY/SCAN)
- What scanners/models were used
- What artifacts are available
- How to reproduce it

**Plugin Registries**:
- `ScannerRegistry`: All available scanners
- `ModelRegistry`: All available models
- `IndicatorRegistry`: All available indicators

Registries enable:
- UI dropdowns populated dynamically
- Agent discovery of capabilities
- Pluggable architecture

---

## Backend Architecture

### SimulationSession
Backend-owned simulation stepping with events.

**Owns**:
- MarketStepper (bar-by-bar stepping)
- Indicator cache
- Active accounts (multi-account)
- Active policies (scanners/models)

**Emits**: Structured events via SSE
- BAR
- INDICATORS
- DECISION
- ORDER_SUBMIT / FILL
- POSITION_OPEN / POSITION_CLOSE
- ACCOUNT_UPDATE

Frontend subscribes to events and renders them.

### AccountManager
Multi-account simulation tracking.

**Features**:
- Create/delete accounts
- Route orders to specific accounts
- Track PnL per account
- Aggregate statistics

**Use cases**:
- Multiple strategies in one session
- Different risk profiles
- Prop firm rule testing (per-account limits)

---

## API Endpoints

### Manifests
- `GET /runs/{run_id}/manifest` - Get run manifest
  
### Registries  
- `GET /registries/scanners` - List all scanners
- `GET /registries/models` - List all models
- `GET /registries/indicators` - List all indicators

### Indicators
- `GET /runs/{run_id}/indicators?indicator_ids=ema,atr` - Get indicator series

All indicators return first-class series (not hardcoded visuals).
Frontend renders them generically.

---

## Navigation

Top navigation bar shows three lanes:
- Click SCAN, REPLAY, or TRAIN to switch
- Each lane has its own page with appropriate UI
- Clear description of what each mode does

---

## Future Enhancements

1. **Session Control Menu**: File/Settings dropdown for:
   - New/Load/Reset session
   - Speed/Pause/Step controls
   - Add/Remove accounts
   - Add/Remove policies
   - Toggle indicators
   - Export run

2. **Agent Skills**: Enhanced agent capabilities:
   - `list_scanners()` / `list_models()` / `list_indicators()`
   - `create_strategy_config()`
   - `run_scan(config)`
   - `start_simulation_session(manifest_id, policies, accounts)`

3. **Indicator Overlays**: Generic overlay system
   - No hardcoded drawing per indicator
   - Backend returns series with metadata
   - Frontend draws any indicator type

---

## For Developers

### Adding a New Scanner
```python
from src.core.registries import ScannerRegistry

@ScannerRegistry.register(
    scanner_id="my_scanner",
    name="My Scanner",
    description="Description here",
    params_schema={"param1": {"type": "integer", "default": 10}}
)
class MyScanner:
    def __init__(self, param1=10):
        self.param1 = param1
    
    def scan(self, step_result):
        # Return ScannerResult
        pass
```

### Adding a New Model
```python
from src.core.registries import ModelRegistry

@ModelRegistry.register(
    model_id="my_model",
    name="My Model",
    description="Description here"
)
class MyModel:
    def __init__(self, model_path):
        # Load model
        pass
    
    def predict(self, features):
        # Return predictions
        pass
```

### Adding a New Indicator
```python
from src.core.registries import IndicatorRegistry, IndicatorSeries

@IndicatorRegistry.register(
    indicator_id="my_indicator",
    name="My Indicator",
    output_type="line"  # or "histogram", "band", "marker"
)
class MyIndicator:
    def __init__(self, period=20):
        self.period = period
    
    def compute(self, stepper) -> IndicatorSeries:
        # Calculate indicator
        # Return IndicatorSeries with points
        pass
```

---

## Summary

The three lanes architecture provides:
- **Clear separation** of concerns (TRAIN vs REPLAY vs SCAN)
- **Explicit contracts** via RunManifest
- **Plugin system** via registries
- **Backend-owned simulation** with event streaming
- **Multi-account support** for complex testing
- **First-class indicators** for flexible visualization

This architecture scales to multi-model voting, complex policy chains, and diverse testing scenarios while maintaining strict causality boundaries.

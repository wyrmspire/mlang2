# Interactive Simulation - Usage Guide

## Overview

The Interactive Simulation Lab allows you to test trading strategies with full backend-owned order management. Unlike static backtests, this mode lets you:

- **Step through bars** one-by-one or play continuously
- **See order fills in real-time** as they happen
- **Adjust parameters mid-stream** without restarting
- **Watch OCO brackets** trigger and resolve
- **Track live statistics** on performance

## Getting Started

### 1. Start the Backend

```bash
cd /path/to/mlang2
python -m uvicorn src.server.main:app --reload --port 8000
```

The server will start on `http://localhost:8000`

### 2. Start the Frontend

```bash
npm run dev
```

The UI will be available at `http://localhost:5173`

### 3. Navigate to Simulation

- Open the web interface
- Navigate to the Simulation view
- Configure your session

## Usage

### Starting a Simulation

1. **Select Strategy**:
   - `random` - Random signal generator (for testing)
   - `always_long` - Always generates long signals
   - `ifvg_cnn` - CNN-based IFVG detector (future)

2. **Configure Parameters**:
   - **Entry Type**: `MARKET` or `LIMIT`
   - **Stop Loss**: 0.5 to 3.0 ATR
   - **Take Profit**: 1.0 to 3.0x risk multiple
   - **Playback Speed**: 50ms to 500ms per bar

3. **Click "Start Simulation"**:
   - Backend creates a session
   - Loads historical data
   - Initializes the OMS

### Controls

- **▶ Play**: Start continuous playback at configured speed
- **■ Pause**: Pause playback (can resume later)
- **→ Step**: Advance exactly one bar (when paused)
- **Update Parameters**: Apply new settings to active session

### Reading the Display

**Chart Area**:
- Candlestick chart with historical data
- Active OCO brackets shown as lines (entry, stop, TP)
- Completed trades marked on chart

**Event Log** (bottom panel):
- Real-time OMS events
- Order fills, OCO entries/exits
- Position opens/closes

**Status Panel** (right sidebar):
- Progress through data
- Bar count
- Total OCOs submitted
- Active OCOs currently open
- Active positions
- Closed positions

**Active OCOs Panel**:
- Shows details for each active OCO
- Entry price, stop price, TP price
- Current status

## Advanced Features

### Mid-Stream Parameter Updates

You can change execution parameters without restarting:

1. Adjust sliders for Stop Loss or Take Profit
2. Change Entry Type dropdown
3. Click **Update Parameters**
4. New settings apply to future OCOs

This is useful for testing:
- "What if I used wider stops?"
- "How do limit entries vs market entries perform?"
- "Is 2x TP better than 1.5x?"

### API Usage

You can also use the API directly:

```python
import requests

# Start session
response = requests.post('http://localhost:8000/sim/start', json={
    'strategy_name': 'random',
    'config': {
        'entry_type': 'MARKET',
        'stop_atr': 1.0,
        'tp_multiple': 1.5,
        'auto_submit_ocos': True
    },
    'start_idx': 0,
    'end_idx': 1000
})

session_id = response.json()['session_id']

# Step forward
for _ in range(10):
    result = requests.post(
        f'http://localhost:8000/sim/{session_id}/step',
        json={'n_bars': 1}
    ).json()
    
    print(f"Events: {result['events']}")
    print(f"Progress: {result['state']['progress']:.1%}")

# Stop session
requests.post(f'http://localhost:8000/sim/{session_id}/stop')
```

## Architecture

### Backend Components

**DataStream** (`src/sim/engine.py`):
- Yields historical bars sequentially
- Tracks progress through dataset
- Supports date-range filtering

**StrategyRunner** (`src/sim/engine.py`):
- Wraps strategy/model logic
- Takes bar + history → returns signal
- Currently supports: `random`, `always_long`
- Future: CNN model integration

**OMS - Order Management System** (`src/sim/engine.py`):
- Maintains pending orders
- Checks for fills on each bar
- Manages open positions
- Handles OCO bracket logic
- Tracks P&L, MAE, MFE

**SimulationEngine** (`src/sim/engine.py`):
- Coordinates all components
- Processes bars through pipeline
- Emits events for UI display

### API Routes

**`POST /sim/start`**:
- Creates new simulation session
- Returns session_id

**`POST /sim/{session_id}/step`**:
- Advances n bars
- Returns bars, events, state

**`POST /sim/{session_id}/update_params`**:
- Updates execution parameters
- Affects future OCOs only

**`GET /sim/{session_id}/state`**:
- Returns current state snapshot

**`POST /sim/{session_id}/stop`**:
- Stops and cleans up session

**`GET /sim/sessions`**:
- Lists all active sessions

## Testing

Run the test suite:

```bash
python test_sim_engine.py
```

Tests cover:
- DataStream iteration
- StrategyRunner signal generation
- OMS order processing and OCO handling
- Full SimulationEngine pipeline

## Tips

1. **Start Small**: Use 100-500 bars for initial testing
2. **Watch the Event Log**: It shows exactly what the OMS is doing
3. **Use Step Mode**: Great for debugging specific scenarios
4. **Try Different Speeds**: Fast for overview, slow for detail
5. **Update Parameters**: Test sensitivity to execution settings

## Troubleshooting

**"Backend unavailable"**:
- Make sure uvicorn server is running
- Check port 8000 is available
- Check server logs for errors

**"No data loaded"**:
- Ensure continuous contract data exists
- Check data date range matches request
- Look for errors in server console

**"Session not found"**:
- Sessions expire after 24 hours
- Session may have been stopped
- Check active sessions: `GET /sim/sessions`

**OCOs not appearing**:
- Check strategy is generating signals
- Verify `auto_submit_ocos: true` in config
- Look at event log for signal events

## Next Steps

1. **Integrate CNN Models**: Add trained models to StrategyRunner
2. **Save Results**: Export simulation results to files
3. **Compare Strategies**: Run multiple sessions in parallel
4. **Optimize Parameters**: Automate parameter sweeps
5. **Real-Time Mode**: Connect to live data feeds

# Position Tool Fix for Simulation Mode

## Problem
Position tools (SL/TP visualization boxes) were not rendering correctly in simulation mode. The boxes would either not appear or have incorrect timing/width. In scan mode, they worked perfectly.

## Root Cause
The simulation mode uses the Order Management System (OMS) to track trades via OCO brackets. When trades completed:
1. The OMS moved brackets from `active_ocos` to `completed_ocos`
2. But `get_state()` only returned `active_ocos`, not `completed_ocos`
3. Frontend had no way to access completed trade information
4. Position boxes require both VizTrade AND VizDecision with OCO data to render

## Solution

### Backend Changes (`src/sim/engine.py`)
Enhanced `OrderManagementSystem.get_state()` to include:

```python
'completed_ocos': [
    {
        'name': oco.config.name,
        'status': oco.status.value,
        'direction': oco.config.direction,
        'entry_price': oco.entry_price,
        'stop_price': oco.stop_price,
        'tp_price': oco.tp_price,
        'entry_bar': oco.entry_bar,
        'entry_time': oco.entry_fill.fill_time.isoformat(),
        'exit_bar': oco.exit_fill.fill_bar,
        'exit_time': oco.exit_fill.fill_time.isoformat(),
        'exit_price': oco.exit_fill.fill_price,
        'bars_in_trade': oco.bars_in_trade,
        'mae': oco.mae,
        'mfe': oco.mfe,
    }
    for oco in self.completed_ocos
]
```

### Frontend Changes (`src/components/SimulationView.tsx`)
Convert completed OCOs to visualization data:

```typescript
// For each completed OCO, create:
// 1. VizTrade - for trade statistics and timing
const trade: VizTrade = {
    trade_id: oco.name,
    decision_id: oco.name,
    entry_time: oco.entry_time,
    exit_time: oco.exit_time,
    entry_bar: oco.entry_bar,
    exit_bar: oco.exit_bar,
    // ... other fields
};

// 2. VizDecision - for OCO levels (SL/TP prices)
const decision: VizDecision = {
    decision_id: oco.name,
    timestamp: oco.entry_time,
    oco: {
        entry_price: oco.entry_price,
        stop_price: oco.stop_price,
        tp_price: oco.tp_price,
        direction: oco.direction,
        // ... other fields
    }
};
```

## How Position Boxes Work

In `CandleChart.tsx` (lines 229-298):

1. Iterates through `trades` array
2. For each trade, finds matching `decision` by `decision_id`
3. Uses `decision.oco` for price levels (entry, stop, tp)
4. Uses `decision.timestamp` for start time
5. Uses `trade.exit_time` for end time
6. Creates `PositionBox` primitives with correct width

## Data Flow

```
Simulation Mode:
Backend OMS → completed_ocos → Frontend State → VizTrade + VizDecision → Position Boxes

Scan Mode:
Runner → exporter.on_trade_closed() → VizTrade + VizDecision → Position Boxes
```

## Key Files
- `src/sim/engine.py` - OMS state serialization
- `src/components/SimulationView.tsx` - OCO to Viz conversion
- `src/components/CandleChart.tsx` - Position box rendering (unchanged)
- `src/components/PositionBox.ts` - Box primitive implementation (unchanged)

## Testing
1. Start simulation with any strategy
2. Let trades complete (SL/TP/Timeout)
3. Verify green (TP) and red (SL) boxes appear
4. Verify box width matches trade duration exactly
5. Verify scan mode still works (uses different code path)

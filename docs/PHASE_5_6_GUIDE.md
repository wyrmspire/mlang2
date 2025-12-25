# Phase 5/6 Implementation Guide

This document describes the changes made to enforce architectural invariants for Phase 5 (OCO Migration) and Phase 6 (Window Policy).

## Overview

The changes address five critical issues identified in the problem statement:

1. **Enforce 2-hour window in exporter** - Window policy is now enforced at the exporter level
2. **Single source of truth for contracts/risk sizing** - Centralized sizing calculations
3. **Stop counterfactual paths** - Use OCOEngine outputs, not approximate conversions
4. **Chart render from decision windows** - UI reads from decision.window.raw_ohlcv_1m
5. **Add golden run tests** - Tests validate architectural invariants

## New Modules

### `src/sim/sizing.py`

Centralized position sizing module with three main functions:

```python
from src.sim.sizing import calculate_contracts, calculate_pnl_dollars, calculate_reward_dollars

# Calculate contracts based on risk
result = calculate_contracts(
    entry_price=5000.0,
    stop_price=4990.0,
    max_risk_dollars=300.0
)
print(f"Contracts: {result.contracts}")  # 6
print(f"Risk: ${result.risk_dollars}")   # $300

# Calculate PnL (single source of truth)
pnl_points, pnl_dollars = calculate_pnl_dollars(
    entry_price=5000.0,
    exit_price=5010.0,
    direction="LONG",
    contracts=6,
    include_commission=True
)
print(f"PnL: {pnl_points} points = ${pnl_dollars}")  # 10 points = $285
```

**Key Features:**
- Enforces minimum 1 contract
- Returns actual risk with calculated contracts
- Consistent with CostModel for point_value and commission
- Validates invariant: `pnl_dollars == pnl_points * point_value * contracts`

### `src/viz/window_utils.py`

2-hour window enforcement utilities:

```python
from src.viz.window_utils import enforce_2hour_window

# Enforce 2-hour policy for a trade window
raw_ohlcv, warning = enforce_2hour_window(
    df_1m=df_1m,
    entry_time=entry_time,
    exit_time=exit_time  # or use bars_held
)

if warning:
    print(f"Warning: {warning}")
```

**Window Policy:**
- `window_start = entry_time - 2 hours`
- `window_end = exit_time + 2 hours`
- Returns warning if data is missing
- Never silently truncates

## Updated Modules

### `src/viz/export.py`

**Changes:**
1. `on_bracket_created()` now requires `contracts` parameter
2. Tracks window warnings in `_window_warnings`
3. `finalize()` records window bounds in manifest.json
4. `on_trade_closed()` uses contracts from decision's OCO

**Usage:**
```python
# OLD (wrong - defaults to 1 contract)
exporter.on_bracket_created(decision_id, bracket)

# NEW (correct - explicit contracts)
sizing_result = calculate_contracts(entry_price, stop_price, 300.0)
exporter.on_bracket_created(decision_id, bracket, contracts=sizing_result.contracts)
```

### `src/strategy/scan.py`

**Changes:**
1. Uses `calculate_contracts()` for position sizing
2. Uses `calculate_pnl_dollars()` for PnL calculation
3. Uses `enforce_2hour_window()` for raw_ohlcv generation
4. Removed approximate conversion: `pnl_points = cf.pnl_dollars / 50`

**Migration:**
```python
# OLD (approximate/inconsistent)
contracts = max(1, int(max_risk_dollars / (risk_points * point_value)))
pnl_points = cf.pnl_dollars / 50  # Approximate!
pnl_dollars = cf.pnl_dollars * contracts

# NEW (centralized/consistent)
sizing_result = calculate_contracts(entry_price, stop_price, max_risk_dollars)
pnl_points, pnl_dollars = calculate_pnl_dollars(
    entry_price, exit_price, direction, sizing_result.contracts
)
```

### `src/config.py`

**Changes:**
- Added `DEFAULT_MAX_RISK_DOLLARS = 300.0`

This centralizes the default risk parameter used across all strategies.

## Tests

### `tests/test_sizing.py` (16 tests)

Tests for position sizing module:
- Contract calculation for various scenarios
- PnL calculation for LONG/SHORT, WIN/LOSS
- Reward calculation
- Invariant validation: `pnl_dollars == pnl_points * point_value * contracts`

### `tests/test_window_utils.py` (10 tests)

Tests for window utilities:
- 2-hour window enforcement
- Warning generation for missing data
- Window bounds computation from trades
- Output format validation

### `tests/test_golden_runs.py` (4 new tests)

Tests for architectural invariants:
- 2h before/after window bounds in manifest
- Contracts present and non-1 when required
- PnL invariant validation
- OCO results filled status

## Migration Guide for Existing Code

### Step 1: Update Contract Sizing

Replace manual contract calculations with `calculate_contracts()`:

```python
# Before
risk_points = abs(entry_price - stop_price)
risk_per_contract = risk_points * point_value
contracts = max(1, int(max_risk / risk_per_contract))

# After
from src.sim.sizing import calculate_contracts

result = calculate_contracts(entry_price, stop_price, max_risk)
contracts = result.contracts
```

### Step 2: Update PnL Calculations

Replace manual PnL calculations with `calculate_pnl_dollars()`:

```python
# Before
pnl_points = (exit_price - entry_price) if direction == "LONG" else (entry_price - exit_price)
pnl_dollars = pnl_points * point_value * contracts - commission

# After
from src.sim.sizing import calculate_pnl_dollars

pnl_points, pnl_dollars = calculate_pnl_dollars(
    entry_price, exit_price, direction, contracts, include_commission=True
)
```

### Step 3: Update Window Generation

Replace manual window slicing with `enforce_2hour_window()`:

```python
# Before
history_bars = 120  # Approximate
start_idx = max(0, entry_idx - history_bars)
end_idx = min(len(df), entry_idx + lookahead)
raw_ohlcv = df.iloc[start_idx:end_idx].to_dict('records')

# After
from src.viz.window_utils import enforce_2hour_window

raw_ohlcv, warning = enforce_2hour_window(
    df_1m=df_1m,
    entry_time=entry_time,
    bars_held=bars_held
)
```

### Step 4: Update Exporter Calls

Pass contracts to `on_bracket_created()`:

```python
# Before
exporter.on_bracket_created(decision_id, bracket)

# After
exporter.on_bracket_created(decision_id, bracket, contracts=sizing_result.contracts)
```

## Validation

After migration, validate with tests:

```bash
# Run sizing tests
python -m unittest tests.test_sizing

# Run window utility tests
python -m unittest tests.test_window_utils

# Run architectural invariant tests
python -m unittest tests.test_golden_runs.TestArchitecturalInvariants

# Run all tests
python -m unittest discover -s tests -p "test_*.py"
```

## Common Issues

### Issue: "Missing contracts field"
**Solution:** Make sure to pass `contracts` parameter to `on_bracket_created()`

### Issue: "PnL doesn't match"
**Solution:** Use `calculate_pnl_dollars()` consistently, don't mix approximate conversions

### Issue: "Window warning in manifest"
**Solution:** Check that you have 2 hours of data before/after trades, or accept the warning

### Issue: "Position size is always 1"
**Solution:** Use `calculate_contracts()` instead of defaulting to 1

## Next Steps

Remaining work for full Phase 5/6 completion:

1. **MAE/MFE from OCOEngine** - Currently approximated, should come from actual OCO simulation
2. **Migrate remaining scripts** - Update all runner scripts to use centralized sizing
3. **UI validation** - Verify frontend uses `decision.window.raw_ohlcv_1m` time axis

## References

- [ARCHITECTURE_AGREEMENT.md](../ARCHITECTURE_AGREEMENT.md) - Section 3 (Window Policy) and Section 5 (OCO Engine)
- [PHASE_5_10_SUMMARY.md](../PHASE_5_10_SUMMARY.md) - Phase 5/6 status
- Problem statement in PR description

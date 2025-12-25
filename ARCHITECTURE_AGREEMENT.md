# Architecture Agreement

**Version:** 1.0  
**Date:** 2025-12-25  
**Status:** CANONICAL

This document defines the architectural invariants for the MLang2 trading research platform. All code changes, new features, and agent-generated strategies **MUST** comply with these rules to prevent regression.

---

## Core Principle

**One Contract to Rule Them All**: All components producing run artifacts, visualization data, or trading signals must follow a unified contract system. No exceptions.

---

## 1. Run Artifact Requirements

Every strategy run **MUST** produce the following artifacts:

### Required Files

1. **`manifest.json`** - Run metadata and configuration
   - `run_id` (string, unique identifier)
   - `fingerprint` (string, configuration hash)
   - `created_at` (ISO 8601 timestamp with timezone)
   - `config` (dict, complete strategy configuration)
   - `file_inventory` (list of generated files)

2. **`decisions.jsonl`** - All decision points (one JSON object per line)
   - Each decision must include: `decision_id`, `timestamp`, `bar_idx`, `action`, `current_price`, `atr`
   - If `action == "PLACE_ORDER"`, must include `oco` bracket configuration
   - If OCO was executed, must include `oco_results` (flat dictionary, see Section 2)

3. **`trades.jsonl`** - Completed trades (one JSON object per line)
   - Each trade must include: `trade_id`, `decision_id`, `entry_price`, `exit_price`, `pnl_points`, `pnl_dollars`, `outcome`
   - Must include `bars_held`, `entry_time`, `exit_time`, `exit_reason`

### Optional Files

4. **`full_series.json`** - Complete OHLCV timeline (optional, for global view)
5. **`events.jsonl`** - Detailed fill events (optional, for replay)

### Invariant
**All runs must have decisions.jsonl and trades.jsonl, even if empty.**  
**All timestamps must be ISO 8601 format with timezone information.**

---

## 2. Viz Schema - Authoritative Contract

The visualization schema defined in `src/viz/schema.py` is **AUTHORITATIVE**. The backend is the source of truth; the frontend consumes.

### Core Viz Types

1. **`VizDecision`** - Decision point for visualization
   - `decision_id` (string)
   - `timestamp` (ISO 8601 string)
   - `bar_idx` (int)
   - `index` (int, for paging)
   - `scanner_id` (string)
   - `action` (string: "NO_TRADE" | "PLACE_ORDER")
   - `current_price` (float)
   - `atr` (float)
   - `oco` (VizOCO object, if PLACE_ORDER)
   - **`oco_results` (dict, FLAT structure)** - see below
   - `window` (VizWindow object, optional)

2. **`VizTrade`** - Completed trade for visualization
   - `trade_id`, `decision_id`, `index`
   - `direction` (string: "LONG" | "SHORT")
   - `size` (int, number of contracts)
   - `entry_time`, `entry_bar`, `entry_price`
   - `exit_time`, `exit_bar`, `exit_price`, `exit_reason`
   - `outcome` (string: "WIN" | "LOSS" | "BREAKEVEN")
   - `pnl_points`, `pnl_dollars`, `r_multiple`
   - `bars_held` (int)
   - `fills` (list of VizFill objects)

3. **`VizOCO`** - OCO bracket configuration
   - `entry_price`, `stop_price`, `tp_price`
   - `entry_type` (string: "LIMIT" | "MARKET")
   - `direction` (string: "LONG" | "SHORT")
   - **`contracts` (int, REQUIRED)** - Number of contracts for position sizing
   - `reference_type`, `reference_value`
   - `atr_at_creation`, `max_bars`
   - `stop_atr`, `tp_multiple` (for tooltip display)

4. **`VizWindow`** - OHLCV snapshot at decision time
   - Normalized model inputs: `x_price_1m`, `x_price_5m`, `x_price_15m`, `x_price_1h`, `x_price_4h`, `x_context`
   - Raw OHLCV for chart: `raw_ohlcv_1m`
   - Future context: `future_price_1m`
   - Indicators: `indicators` (dict)
   - Normalization metadata: `norm_method`, `norm_params`

### OCO Results Format - CRITICAL INVARIANT

**`oco_results` MUST be a FLAT dictionary at the decision level, NOT nested.**

**CORRECT FORMAT:**
```json
{
  "decision_id": "abc123",
  "oco": {...},
  "oco_results": {
    "filled": true,
    "entry_price": 5000.0,
    "exit_price": 5014.0,
    "exit_type": "TP",
    "bars_held": 23,
    "pnl_points": 14.0,
    "pnl_dollars": 70.0,
    "outcome": "WIN"
  }
}
```

**INCORRECT FORMAT (DO NOT USE):**
```json
{
  "decision_id": "abc123",
  "oco": {
    "results": {...}  // ❌ WRONG - nested
  }
}
```

**Contract Field Requirement:**
- VizOCO must include `contracts` field
- UI uses `contracts` for position box size calculation
- Missing `contracts` breaks UI rendering

---

## 3. Time Window Policy

### 2-Hour Context Rule

**Hard Rule:** The exporter **MUST** guarantee 2 hours of context before the first trade entry and 2 hours after the last exit.

This is enforced at the **exporter level**, not as a UI hack.

### Implementation

1. Exporter computes:
   - `window_start = first_entry_time - 2 hours`
   - `window_end = last_exit_time + 2 hours`

2. `VizWindow.raw_ohlcv_1m` includes complete data for this range

3. UI never "hunts" for missing bars; it renders what the exporter provides

4. Per-trade windows follow the same rule (2h before entry, 2h after exit)

### Exception Handling

- If data doesn't exist for the full 2h window, include what's available and log a warning
- **Never** silently truncate without logging
- Manifest should record actual window bounds

---

## 4. Tool/Plugin/Skill System - Unified Registry

### Single Registry Pattern

All executable components (scanners, models, indicators, strategies) **MUST** be registered in a unified system:

**`ToolRegistry`** (to be implemented in Phase 3):
- Every tool has a `tool_id` (unique string)
- JSON schema for inputs and outputs
- Deterministic artifact outputs (for reproducibility)
- Category labels: "scan", "model", "indicator", "strategy" (not separate registries)

### Current State (Transitional)

Currently, we have:
- `ScannerRegistry`, `ModelRegistry`, `IndicatorRegistry` (in `src/core/registries.py`) ✅ Good pattern
- `SkillRegistry` (in `src/skills/registry.py`) ⚠️ Different contract - to be unified

### End State (Target)

- Single `ToolRegistry` with category labels
- `SkillRegistry` becomes a thin wrapper or disappears
- Agent tool schemas auto-generated from `ToolRegistry`
- No hardcoded tool definitions in server code

---

## 5. OCO Execution - Single Source of Truth

### Unified OCO Engine

**One OCO simulation engine** (to be created/consolidated in Phase 5) defines:

1. **Fill Rules**
   - LIMIT orders: filled if price touches level
   - MARKET orders: filled immediately at next bar's open
   - Slippage model: configurable per order type

2. **Stop/TP Priority**
   - If both stop and TP can be touched in the same bar, which triggers first?
   - Rule: Check bar's high/low extremes first, then check which would be hit first based on direction

3. **Tick Size Rounding**
   - All prices rounded to instrument's tick size
   - Document tick size per instrument (MES = 0.25 points)

4. **Bars Held Definition**
   - `bars_held = exit_bar - entry_bar`
   - Consistent across all strategies

5. **PnL Calculation**
   - `pnl_points = (exit_price - entry_price) * direction_multiplier`
   - `pnl_dollars = pnl_points * point_value * contracts`
   - Direction multiplier: LONG = +1, SHORT = -1

6. **Output Format**
   - Always produces `oco_results` as a **flat dictionary**
   - Always includes `contracts` in the result

### Risk/Position Sizing

- Every OCO bracket must calculate `contracts` based on:
  - Account balance
  - Risk percentage
  - Stop distance
  - Instrument point value

- **Never** default to 1 contract without explicit sizing calculation

---

## 6. Strategy Definition - Declarative Only

### No Ad-Hoc Scripts

Agents **MUST NOT** write arbitrary strategy scripts. Instead, they create a **`StrategySpec`** (to be implemented in Phase 4):

```python
@dataclass
class StrategySpec:
    strategy_id: str
    trigger: TriggerConfig  # Scanner ID + params
    bracket: BracketConfig  # OCO configuration
    sizing: SizingConfig    # Risk/position sizing
    filters: List[FilterConfig]  # Entry filters
    indicators: List[str]   # Indicator IDs to compute
    metadata: Dict[str, Any]
```

### Single Runner

One strategy runner consumes `StrategySpec` and produces compliant artifacts.

Currently: `run_modular_strategy` is moving in this direction but needs full enforcement.

---

## 7. Indicator Overlays - First-Class Contract

### Indicator Series Export

1. Strategy runs **declare** required `indicator_ids` in `StrategySpec`
2. Exporter computes and writes indicator series to run artifacts
3. UI renders overlays generically based on `IndicatorSeries` type:
   - `line`: continuous line plot
   - `histogram`: vertical bars
   - `band`: upper/lower bounds
   - `marker`: discrete points

### No Ad-Hoc Indicator Fields

**Don't** add custom fields to decision records for indicators.  
**Do** use the `IndicatorRegistry` and export `IndicatorSeries` objects.

---

## 8. Frontend Contract

### What the Frontend Expects

The frontend (`src/components/`) expects the backend to provide:

1. **Runs API**: `GET /runs` returns list of run metadata
2. **Decisions API**: `GET /runs/{id}/decisions` returns `VizDecision[]` (JSON array)
3. **Trades API**: `GET /runs/{id}/trades` returns `VizTrade[]` (JSON array)
4. **Series API**: `GET /runs/{id}/series` returns `VizBarSeries` (full OHLCV)

### Frontend Responsibilities

- Render data provided by backend
- **Never** perform data transformations to "fix" backend issues
- **Never** implement workarounds for format drift
- Report format violations as bugs

### Backend Responsibilities

- Produce compliant viz schema
- Validate outputs before writing
- Ensure consistency across all runs

---

## 9. Compatibility Matrix

### Schema Version Compatibility

| Backend Version | Frontend Version | Compatible | Notes |
|----------------|------------------|-----------|-------|
| 1.0            | 1.0              | ✅        | Baseline |
| 1.x            | 1.x              | ✅        | Backwards compatible within major version |
| 2.0            | 1.x              | ⚠️        | May require frontend upgrade |

### Breaking Changes

Changes that **REQUIRE** version bump:
- Removing required fields from viz schema
- Changing field types (e.g., string → int)
- Changing `oco_results` structure
- Changing timestamp format

Changes that **DO NOT** require version bump:
- Adding optional fields
- Adding new indicator types
- Adding new tool categories

---

## 10. Enforcement Mechanisms

### Pre-Commit Checks (Phase 9-10)

1. **Tool Contract Linter**
   - Verify all registered tools have valid JSON schemas
   - Verify tool outputs include required artifact files
   - Verify `VizDecision` shape (flat `oco_results`, `contracts` present)
   - Verify timestamps are ISO 8601 with timezone

2. **Golden File Tests** (Phase 2)
   - Reference run must pass all structural checks
   - New runs compared against golden structure

3. **Property Tests** (Phase 10)
   - OCO engine: verify stop/tp priority, rounding, pnl calculation
   - Viz schema: verify serialization/deserialization
   - Timestamp handling: verify timezone consistency

### CI Pipeline

- Run all linters before merge
- Run golden file tests
- Run property tests
- Fail build on violations

---

## 11. Migration Path

### Current State → Target State

**Phase by Phase:**
1. Document this agreement (✅ current)
2. Establish golden run + tests
3. Unify tool system
4. Declarative strategies
5. Single OCO engine
6. Exporter owns windows
7. First-class indicators
8. Folder cleanup
9. Auto-generated tool catalog + linting
10. Full CI hardening

### Backward Compatibility

During migration:
- Existing runs remain valid
- Old format accepted with deprecation warnings
- New format enforced for new runs
- Hard cutoff after Phase 10 complete

---

## 12. Glossary

- **OCO**: One-Cancels-Other (bracket order with entry, stop, take-profit)
- **VizSchema**: Visualization schema types in `src/viz/schema.py`
- **ToolRegistry**: Unified registry for all executable components
- **StrategySpec**: Declarative strategy configuration
- **Exporter**: Component that collects simulation events and writes viz artifacts
- **Golden Run**: Reference run with known-good structure for regression tests

---

## Document Control

- **Owner**: Architecture Team
- **Review Cycle**: Quarterly or on major changes
- **Approval Required**: Yes (2+ reviewers)
- **Next Review**: 2025-03-25

**Amendments:**
- Any changes to this document require PR review and approval
- Breaking changes require major version bump

---

**END OF ARCHITECTURE AGREEMENT**

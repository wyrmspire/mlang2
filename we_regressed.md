# We Regressed: Atomic Skills Audit Report

> **Date:** 2025-12-28  
> **Status:** Analysis Only - No Changes Made  
> **Purpose:** Document what exists, what's missing, and how to restore atomic composition for agents

---

## Executive Summary

The MLang2 project has **extensive backend components** for strategy composition, but these are **not exposed to agents as atomic tools**. The agents have been reduced to running pre-defined strategies rather than composing new ones from primitives.

**The core regression:** Agents lost the ability to:
- Discover what building blocks exist
- Choose indicators, levels, timeframes, patterns
- Compose custom scans from atomic primitives
- Configure OCO brackets with advanced options
- Run both fast (metrics-only) and full (training data + viz) scans

---

## Part 1: What EXISTS in the Backend

### 1.1 The CompositeScanner (The Heart of Composition)

**File:** `src/policy/composite_scanner.py`

This is a **dynamic strategy engine** that interprets JSON recipes with AND/OR logic:

```python
# Example: Agent could compose this
{
    "name": "My Composed Strategy",
    "entry_trigger": {
        "type": "AND",
        "children": [
            {"type": "ema_cross", "fast": 9, "slow": 21},
            {"type": "rsi_threshold", "threshold": 30, "direction": "lt"}
        ]
    },
    "cooldown_bars": 10
}
```

**Status:** ✅ Exists, ❌ NOT exposed as agent tool

---

### 1.2 Triggers (17+ Composable Entry Signals)

**Location:** `src/policy/triggers/`

| Trigger | File | Description |
|---------|------|-------------|
| `time` | `time_trigger.py` | Time-based entries |
| `candle_pattern` | `candle_patterns.py` | Candlestick patterns |
| `ema_cross` | `indicator_triggers.py` | EMA crossover |
| `rsi_threshold` | `indicator_triggers.py` | RSI extremes |
| `structure_break` | `structure_break.py` | Market structure breaks |
| `fakeout` | `fakeout.py` | False breakout detection |
| `ema200_rejection` | `ema_rejection.py` | EMA 200 rejection |
| `rejection` | `price_action_triggers.py` | Wick rejection |
| `pin_bar` | `price_action_triggers.py` | Pin bar pattern |
| `engulfing` | `price_action_triggers.py` | Engulfing candle |
| `inside_bar` | `price_action_triggers.py` | Inside bar |
| `double_top_bottom` | `price_action_triggers.py` | Double top/bottom |
| `flag_pattern` | `price_action_triggers.py` | Flag pattern |
| `sweep` | `sweep.py` | Liquidity sweep |
| `or_false_break` | `or_false_break.py` | Opening Range false break |
| `vwap_reclaim` | `vwap_reclaim.py` | VWAP reclaim |
| `AND`, `OR`, `NOT` | `logic.py` | Composable logic gates |

**Critical:** There's a `list_triggers()` function in `factory.py` that returns all available triggers - **but no agent tool exposes this!**

---

### 1.3 Scanners (21 Pre-Built Library Scanners)

**Location:** `src/policy/library/`

| Scanner | File | Description |
|---------|------|-------------|
| `VWAPBounceScanner` | `vwap_bounce.py` | VWAP cross and bounce |
| `ICTFVGScanner` | `ict_fvg.py` | Fair Value Gap entry |
| `ICTIFVGScanner` | `ict_ifvg.py` | Inverse FVG |
| `OpeningRangeScanner` | `opening_range.py` | OR breakout/fade |
| `StructureBreakScanner` | `structure_break.py` | BOS/CHoCH |
| `MeanReversionScanner` | `mean_reversion.py` | Mean reversion |
| `SwingBreakoutScanner` | `swing_breakout.py` | Swing breakout |
| `PullerScanner` | `puller.py` | Puller pattern |
| `DelayedBreakoutScanner` | `delayed_breakout.py` | Delayed breakout |
| `SessionBreakScanner` | `session_break.py` | Session level break |
| `VolumeSpikeScanner` | `volume_spike.py` | Volume spike |
| `MomentumDivergenceScanner` | `momentum_divergence.py` | Momentum divergence |
| `FirstPullbackScanner` | `first_pullback.py` | First pullback |
| `MidDayReversalScanner` | `mid_day_reversal.py` | Mid-day reversal |
| *(+ 7 more test scanners)* | `new_test_*.py` | Test scanners |

**Status:** ✅ All exist, ❌ Agent cannot discover or list them

---

### 1.4 Features/Levels (The Context Layer)

**Location:** `src/features/`

| Component | File | What It Provides |
|-----------|------|------------------|
| **Levels** | `levels.py` | PDH/PDL, 1h/4h S/R, current day H/L |
| **Session Levels** | `session_levels.py` | Asian range, London range, Overnight levels |
| **FVG Detection** | `fvg.py` | Fair Value Gap detection, impulse candles |
| **Indicators** | `indicators.py`, `indicators_pro.py` | RSI, EMA, ATR, VWAP, ADR, momentum, etc. |
| **Patterns** | `patterns.py` | Flags, wedges, pullbacks |
| **Swings** | `swings.py` | Swing high/low detection |
| **Time Features** | `time_features.py` | Session, hour, day of week |

**Critical Missing:** No agent tool to:
- `list_levels()` - Discover available level types
- `get_level(type, timestamp)` - Get specific level value
- `find_fvg(direction, timeframe)` - Find Fair Value Gaps

---

### 1.5 Brackets (Exit Strategy Components)

**Location:** `src/policy/brackets.py`

| Bracket Type | Class | Description |
|--------------|-------|-------------|
| ATR-based | `ATRBracket` | Stop/TP as ATR multiples |
| Percent | `PercentBracket` | Stop/TP as % of entry |
| Fixed Points | `FixedBracket` | Fixed point stop/TP |
| ICT Style | `ICTBracket` | PDH/PDL targeting, wick-based stops |
| Level-based | `LevelBracket` | Target specific levels |

**Status:** ✅ Full implementation exists, ❌ Agent cannot configure these

---

### 1.6 Filters (Pre-Trade Conditions)

**Location:** `src/policy/filters.py`

| Filter | Class | Description |
|--------|-------|-------------|
| Session | `SessionFilter` | RTH/Globex only |
| Time | `TimeFilter` | Specific hours, exclude lunch |
| Volatility | `VolatilityFilter` | Min ATR, max ADR usage |
| Filter Chain | `FilterChain` | Combine multiple filters |

---

### 1.7 Skills (Partially Registered Atomic Tools)

**Location:** `src/skills/`

| File | Registered Tools | Status |
|------|------------------|--------|
| `indicator_skills.py` | `get_rsi`, `check_ema_cross`, `get_current_rsi`, `get_atr`, `get_vwap`, `detect_support_resistance`, `get_volume_profile` | ✅ Registered |
| `data_skills.py` | `fetch_ohlcv`, `get_dataset_last_price`, `get_dataset_summary`, `get_market_regime`, `get_time_of_day_stats` | ✅ Registered |
| `pattern_skills.py` | `detect_chart_patterns`, `analyze_pullback` | ✅ Registered |

**The Problem:** These are registered in `ToolRegistry` but **not exposed to agents** because of category filtering (see Part 2).

---

### 1.8 OCO Engine (Order Management)

**Location:** `src/sim/oco_engine.py`

Full-featured OCO engine with:
- Multiple entry types (LIMIT, MARKET, etc.)
- Configurable stop types (smart stops, ATR, level-based)
- Exit priority rules (STOP_FIRST, TP_FIRST, INTRABAR_MODEL)
- Full trade lifecycle tracking

---

## Part 2: THE BUG - Category Filtering

**File:** `src/server/main.py` (lines 600-609)

```python
def get_agent_tools() -> List[Dict[str, Any]]:
    """Get tools for main TradeViz agent (strategy + indicators only)."""
    return ToolRegistry.get_gemini_function_declarations(
        categories=[ToolCategory.STRATEGY, ToolCategory.INDICATOR]  # ❌ MISSING UTILITY!
    )

def get_lab_tools() -> List[Dict[str, Any]]:
    """Get tools for lab agent (all categories)."""
    return ToolRegistry.get_gemini_function_declarations()  # ✅ Gets everything
```

**Impact:**
- All skills in `src/skills/` are registered as `ToolCategory.SCANNER` or `ToolCategory.UTILITY`
- TradeViz agent only sees `STRATEGY` + `INDICATOR` categories
- Navigation tools (`set_index`, `set_mode`, `load_run`) are all `UTILITY` → **HIDDEN**

---

## Part 3: What's MISSING (Never Built)

These atomic composition tools **do not exist**:

### 3.1 Discovery Tools

| Tool | Purpose |
|------|---------|
| `list_triggers()` | Show all available trigger types |
| `get_trigger_info(trigger_id)` | Get trigger parameters and description |
| `list_scanners()` | Show all library scanners |
| `list_levels()` | Show available level types |
| `list_indicators()` | Show available indicators |
| `list_brackets()` | Show bracket types |
| `list_filters()` | Show filter types |

### 3.2 Composition Tools

| Tool | Purpose |
|------|---------|
| `create_trigger(type, params)` | Build a trigger config |
| `create_bracket(type, params)` | Build a bracket config |
| `create_filter(type, params)` | Build a filter config |
| `compose_scan(trigger, bracket, filters)` | Combine into full scan |
| `save_scan_spec(name, spec)` | Save for reuse |
| `load_scan_spec(name)` | Load saved spec |

### 3.3 Execution Tools

| Tool | Purpose |
|------|---------|
| `run_fast_viz(scan_spec, date_range)` | Run metrics-only (no full data) |
| `run_full_viz(scan_spec, date_range)` | Run with training data + visualization |
| `preview_scan(scan_spec, bars=100)` | Quick preview before full run |

### 3.4 Context Tools

| Tool | Purpose |
|------|---------|
| `get_fvg(direction, timeframe)` | Find Fair Value Gaps |
| `get_session_levels(date)` | Get Asian/London/Overnight levels |
| `get_level_at_time(level_type, timestamp)` | Get specific level value |
| `choose_timeframe(tf)` | Set analysis timeframe |
| `set_window(bars)` | Set lookback window |

---

## Part 4: File Organization Issues

### 4.1 Potentially Misplaced Files

| File | Current Location | Should Be |
|------|------------------|-----------|
| Skills with scanner logic | `src/skills/` | OK, but need proper registration |
| Sweep utilities | `scripts/sweep/` | OK for scripts |
| Pattern detection | `src/features/patterns.py` | OK, needs agent tool wrapper |
| FVG detection | `src/features/fvg.py` | OK, needs agent tool wrapper |

### 4.2 Good Organization (Keep As-Is)

```
src/
├── core/           # ToolRegistry, enums - GOOD
├── features/       # Indicators, levels, FVG - GOOD
├── policy/
│   ├── triggers/   # Atomic triggers - GOOD
│   ├── library/    # Pre-built scanners - GOOD
│   ├── brackets.py # Bracket types - GOOD
│   ├── filters.py  # Filter types - GOOD
│   └── composite_scanner.py # Dynamic composition - GOOD
├── sim/            # Execution engine - GOOD
├── skills/         # Agent-callable wrappers - NEEDS EXPANSION
└── tools/          # Agent tools - NEEDS COMPOSITION TOOLS
```

---

## Part 5: How to Restore Atomic Composition

### Step 1: Fix Category Filtering (Quick Win)

In `src/server/main.py`, modify `get_agent_tools()`:

```python
def get_agent_tools() -> List[Dict[str, Any]]:
    """Get tools for TradeViz agent."""
    return ToolRegistry.get_gemini_function_declarations(
        categories=[
            ToolCategory.STRATEGY,
            ToolCategory.INDICATOR,
            ToolCategory.UTILITY,  # ADD THIS
            ToolCategory.SCANNER,  # ADD THIS
        ]
    )
```

**Risk:** Low - just exposes existing registered tools.

### Step 2: Add Discovery Tools (Medium Effort)

Create `src/tools/discovery_tools.py`:

```python
@ToolRegistry.register(
    tool_id="list_triggers",
    category=ToolCategory.UTILITY,
    name="List Available Triggers",
    description="List all trigger types that can be used to compose strategies"
)
class ListTriggersTool:
    def execute(self, **kwargs):
        from src.policy.triggers.factory import list_triggers, TRIGGER_REGISTRY
        return {
            "triggers": [
                {"id": tid, "class": str(tcls.__name__)}
                for tid, tcls in TRIGGER_REGISTRY.items()
            ]
        }
```

Similarly for `list_scanners`, `list_levels`, `list_brackets`, `list_filters`.

### Step 3: Add Composition Tools (Higher Effort)

Create `src/tools/composition_tools.py`:

```python
@ToolRegistry.register(
    tool_id="compose_scan",
    category=ToolCategory.STRATEGY,
    name="Compose Scan",
    description="Compose a custom scan from trigger, bracket, and filters",
    input_schema={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "trigger": {"type": "object"},  # Trigger config
            "bracket": {"type": "object"},  # Bracket config
            "filters": {"type": "array"},   # Filter configs
            "cooldown_bars": {"type": "integer", "default": 15}
        },
        "required": ["name", "trigger"]
    }
)
class ComposeScanTool:
    def execute(self, **inputs):
        # Validate and compose the scan spec
        spec = {
            "name": inputs["name"],
            "entry_trigger": inputs["trigger"],
            "oco": inputs.get("bracket", {"type": "atr", "stop_atr": 2.0, "tp_atr": 3.0}),
            "filters": inputs.get("filters", []),
            "cooldown_bars": inputs.get("cooldown_bars", 15)
        }
        
        # Validate trigger exists
        from src.policy.triggers.factory import trigger_from_dict
        try:
            trigger_from_dict(spec["entry_trigger"])
        except ValueError as e:
            return {"error": str(e)}
        
        return {"success": True, "scan_spec": spec}
```

### Step 4: Add Fast Viz / Full Viz Execution Modes

Modify `run_modular_strategy` or create new tools:

```python
@ToolRegistry.register(
    tool_id="run_fast_viz",
    category=ToolCategory.STRATEGY,
    name="Run Fast Viz",
    description="Run scan in fast mode - metrics only, no full visualization data",
    # ... schema
)
class RunFastVizTool:
    def execute(self, scan_spec, start_date, end_date, **kwargs):
        # Calls fast_forward.py internally
        ...
```

---

## Part 6: Integration Without Breaking Current System

### 6.1 Additive Changes Only

- **DO NOT** modify existing tool schemas unless adding optional parameters
- **DO NOT** remove any existing tools
- **ADD** new tools alongside existing ones
- **KEEP** `run_modular_strategy` working as-is

### 6.2 Backward Compatibility

The current hardcoded trigger_type enum:
```python
"trigger_type": ["ema_cross", "ema_bounce", "rsi_threshold", "ifvg", "orb", "candle_pattern", "time"]
```

**Keep this working** for now. Add a **new** `trigger_config` parameter that accepts the full JSON spec:

```python
"trigger_config": {
    "type": "object",
    "description": "Full trigger config (alternative to trigger_type)"
}
```

The backend can check: if `trigger_config` is provided, use it; otherwise fall back to `trigger_type`.

### 6.3 Phased Rollout

1. **Phase 1:** Fix category filtering → immediately exposes existing skills
2. **Phase 2:** Add discovery tools → agents can explore capabilities
3. **Phase 3:** Add composition tools → agents can build custom scans
4. **Phase 4:** Add fast/full viz modes → agents can iterate quickly

---

## Part 7: The Vision Restored

### Before (Current State)
```
User: "Build a strategy with VWAP and ADR"
Agent: *uses run_modular_strategy with trigger_type="ema_cross"*
       (Falls back to generic because VWAP isn't in the enum)
```

### After (Goal State)
```
User: "Build a strategy with VWAP and ADR"

Agent: list_triggers()
       → Sees "vwap_reclaim" available

Agent: list_levels()
       → Sees "adr_high", "adr_low" available

Agent: compose_scan(
    name="VWAP_ADR_Strategy",
    trigger={
        "type": "AND",
        "children": [
            {"type": "vwap_reclaim"},
            {"type": "comparison", "left": "price", "right": "adr_low", "op": "gt"}
        ]
    },
    bracket={"type": "atr", "stop_atr": 1.5, "tp_atr": 3.0}
)

Agent: run_fast_viz(scan_spec, start_date="2025-05-01", end_date="2025-05-31")
       → Gets quick metrics

Agent: "Win rate: 62%, 23 trades. Want me to run full viz?"
```

---

## Summary of Findings

| Category | Status |
|----------|--------|
| **Triggers** | ✅ 17+ exist, ❌ not discoverable by agent |
| **Scanners** | ✅ 21 exist, ❌ not listable by agent |
| **Levels** | ✅ Full implementation, ❌ no agent access |
| **FVG** | ✅ Detection exists, ❌ no agent tool |
| **Brackets** | ✅ 5 types, ❌ agent can't configure |
| **Filters** | ✅ 4 types, ❌ agent can't compose |
| **CompositeScanner** | ✅ Full composition engine, ❌ not agent-accessible |
| **Skills** | ✅ 14 registered, ❌ hidden by category filter |
| **OCO Engine** | ✅ Full implementation, ✅ used internally |

**Root Cause:** The ToolRegistry has everything, but `get_agent_tools()` filters it down to almost nothing.

**Fix Complexity:** 
- Phase 1 (category fix): 1 line change
- Phase 2 (discovery): ~200 lines new code
- Phase 3 (composition): ~400 lines new code
- Phase 4 (execution modes): ~300 lines new code

---

*Document generated by audit process. No changes made to codebase.*

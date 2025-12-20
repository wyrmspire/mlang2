# Agent Tools Guide

**Complete reference for autonomous agents building and testing trading strategies**

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Tool Catalog](#tool-catalog)
3. [Building Strategies](#building-strategies)
4. [Running Simulations](#running-simulations)
5. [Pattern Discovery](#pattern-discovery)
6. [Advanced Composition](#advanced-composition)
7. [Validation & Testing](#validation--testing)
8. [Common Workflows](#common-workflows)
9. [Best Practices](#best-practices)

---

## Quick Start

```python
from src.tools import ToolCatalog, StrategyBuilder, SimulationRunner

# Discover available tools
catalog = ToolCatalog()
catalog.print_catalog()

# Build a strategy from template
strategy = StrategyBuilder.from_template(
    "mean_reversion",
    start_date="2025-03-01",
    end_date="2025-03-15",
    oversold=25,
    overbought=75
)

# Run simulation with model
result = SimulationRunner.run(
    strategy_config=strategy,
    model_path="runs/exp_001/model.pt",
    start_date="2025-03-01",
    end_date="2025-03-15"
)

print(result.summary())
```

---

## Tool Catalog

### Discovering Tools

The `ToolCatalog` provides centralized discovery of all available tools:

```python
from src.tools import ToolCatalog

catalog = ToolCatalog()

# List all tools
all_tools = catalog.list_all()

# List by category
triggers = catalog.list_by_category('trigger')
scanners = catalog.list_by_category('scanner')
brackets = catalog.list_by_category('bracket')

# Search for tools
time_tools = catalog.search('time')

# Get detailed info
info = catalog.get_info('time_trigger')
print(f"Name: {info.name}")
print(f"Description: {info.description}")
print(f"Parameters: {info.parameters}")
print(f"Examples: {info.examples}")
```

### Tool Categories

- **Triggers**: Atomic entry signals (time, patterns, indicators)
- **Scanners**: Setup detection (when to check for entries)
- **Brackets**: Risk management (stops and targets)
- **Models**: Neural models for decision-making
- **Indicators**: Technical indicators
- **Utilities**: Helper tools (builders, validators, etc.)

---

## Building Strategies

### From Templates

Use pre-built templates for common strategies:

```python
from src.tools import StrategyBuilder

# List available templates
StrategyBuilder.print_all_templates()

# Mean Reversion
strategy = StrategyBuilder.from_template(
    "mean_reversion",
    start_date="2025-03-01",
    end_date="2025-03-15",
    oversold=25,
    overbought=75,
    stop_atr=1.0,
    tp_multiple=1.5
)

# Opening Range Breakout
strategy = StrategyBuilder.from_template(
    "opening_range",
    start_date="2025-03-01",
    end_date="2025-03-15",
    or_minutes=30,
    stop_atr=0.8
)

# Time-Based
strategy = StrategyBuilder.from_template(
    "time_based",
    start_date="2025-03-01",
    end_date="2025-03-15",
    entry_hour=10,
    entry_minute=0
)
```

### Custom Strategies

Build custom strategies from components:

```python
from src.tools import StrategyBuilder

strategy = StrategyBuilder.create(
    name="Custom Level Bounce",
    scanner_id="level_proximity",
    scanner_params={
        "atr_threshold": 0.3,
        "level_types": ["1h", "4h"]
    },
    bracket={
        "type": "atr",
        "stop_atr": 1.0,
        "tp_atr": 2.0
    },
    start_date="2025-03-01",
    end_date="2025-03-15"
)
```

### With Modular Triggers

Use the modular scanner with custom triggers:

```python
strategy = StrategyBuilder.create(
    name="Time and RSI Entry",
    scanner_id="modular",
    scanner_params={
        "trigger_config": {
            "type": "time",
            "hour": 10,
            "minute": 0
        },
        "cooldown_bars": 30
    },
    start_date="2025-03-01",
    end_date="2025-03-15"
)
```

---

## Running Simulations

### Basic Simulation

The `SimulationRunner` executes strategies with model inference:

```python
from src.tools import SimulationRunner

result = SimulationRunner.run(
    strategy_id="opening_range",
    model_path="models/my_model.pt",
    start_date="2025-03-01",
    end_date="2025-03-15"
)

# View results
print(result.summary())
print(f"Win Rate: {result.win_rate:.1%}")
print(f"Total PnL: ${result.total_pnl:.2f}")
print(f"Sharpe: {result.sharpe_ratio:.2f}")
```

### Advanced Configuration

```python
from src.tools import SimulationRunner, StrategyBuilder
from src.experiments.config import ReplayConfig

# Build strategy
strategy = StrategyBuilder.from_template(
    "mean_reversion",
    start_date="2025-03-01",
    end_date="2025-03-31",
    oversold=25,
    overbought=75
)

# Configure replay
replay_config = ReplayConfig(
    speed_multiplier=1.0,
    pause_on_decision=False,
    show_oco_zones=True
)

# Run simulation
result = SimulationRunner.run(
    strategy_config=strategy,
    model_path="models/mean_rev_v2.pt",
    start_date="2025-03-01",
    end_date="2025-03-31",
    replay_config=replay_config
)
```

### Batch Testing

Test multiple configurations:

```python
configs = [
    {"strategy_id": "mean_reversion", "oversold": 20, "overbought": 80},
    {"strategy_id": "mean_reversion", "oversold": 25, "overbought": 75},
    {"strategy_id": "mean_reversion", "oversold": 30, "overbought": 70},
]

results = SimulationRunner.batch_run(
    strategy_configs=configs,
    model_path="models/my_model.pt",
    start_date="2025-03-01",
    end_date="2025-03-15"
)

# Compare results
for result in results:
    print(f"{result.strategy_config}: Win Rate = {result.win_rate:.1%}")
```

### Model Comparison

Compare multiple models on same strategy:

```python
from src.tools import StrategyBuilder, SimulationRunner

strategy = StrategyBuilder.from_template("opening_range")

results = SimulationRunner.compare_models(
    strategy_config=strategy,
    model_paths=[
        "models/model_v1.pt",
        "models/model_v2.pt",
        "models/model_v3.pt"
    ],
    start_date="2025-03-01",
    end_date="2025-03-15"
)

# Find best model
best_model = max(results.items(), key=lambda x: x[1].total_pnl)
print(f"Best model: {best_model[0]}")
print(f"PnL: ${best_model[1].total_pnl:.2f}")
```

---

## Pattern Discovery

### Scan for Patterns

Discover pattern occurrences in historical data:

```python
from src.tools import PatternScanner

# Scan for hammer candles
results = PatternScanner.scan(
    data_path="data/ES_1m_2025-03.parquet",
    pattern_type="candle_pattern",
    pattern_config={"patterns": ["hammer"]},
    include_outcomes=True
)

PatternScanner.print_pattern_report(results)
```

Output:
```
Pattern Analysis: candle_pattern
============================================================
Total Occurrences: 142
Success Rate: 58.5%
Avg Forward PnL: $3.42
Best Time of Day: 10:00
```

### Compare Patterns

```python
patterns = [
    {"type": "candle_pattern", "patterns": ["hammer"], "name": "Hammer"},
    {"type": "candle_pattern", "patterns": ["doji"], "name": "Doji"},
    {"type": "rsi_threshold", "threshold": 30, "direction": "below", "name": "RSI<30"},
]

comparison = PatternScanner.compare_patterns(
    data_path="data/ES_1m_2025-03.parquet",
    pattern_configs=patterns,
    include_outcomes=True
)

PatternScanner.print_comparison_report(comparison)
```

---

## Advanced Composition

### Trigger Composition

Combine multiple triggers with logical operators:

```python
from src.tools import TriggerComposer

# AND: Both conditions must be true
time_and_rsi = TriggerComposer.AND([
    {"type": "time", "hour": 10, "minute": 0},
    {"type": "rsi_threshold", "threshold": 30, "direction": "below"}
])

# OR: Either condition can be true
pattern_or_level = TriggerComposer.OR([
    {"type": "candle_pattern", "patterns": ["hammer"]},
    {"type": "rsi_threshold", "threshold": 25, "direction": "below"}
])

# SEQUENCE: Conditions in order
rsi_then_pattern = TriggerComposer.SEQUENCE([
    {"type": "rsi_threshold", "threshold": 30, "direction": "below"},
    {"type": "candle_pattern", "patterns": ["hammer"]}
], max_bars=3)
```

### Use in Strategies

```python
from src.tools import StrategyBuilder, TriggerComposer

# Create composite trigger
composite = TriggerComposer.AND([
    {"type": "time", "hours": [10, 14], "minute": 0},
    {"type": "rsi_threshold", "threshold": 30, "direction": "below"}
])

# Use in strategy
strategy = StrategyBuilder.create(
    name="Time + RSI Strategy",
    scanner_id="modular",
    scanner_params={
        "trigger_config": composite.to_dict(),
        "cooldown_bars": 30
    },
    start_date="2025-03-01",
    end_date="2025-03-15"
)
```

---

## Validation & Testing

### Validate Strategies

Always validate before running:

```python
from src.tools import StrategyValidator, StrategyBuilder

strategy = StrategyBuilder.from_template("mean_reversion")

# Validate
result = StrategyValidator.validate(strategy)
result.print_report()

if result.has_errors():
    print("Fix errors before running!")
else:
    print("Strategy is valid!")
```

### Quick Validation

```python
# Raises exception if invalid
StrategyValidator.quick_check(strategy)
```

---

## Common Workflows

### Workflow 1: Discover → Build → Test

```python
from src.tools import PatternScanner, StrategyBuilder, SimulationRunner

# 1. Discover best pattern
patterns = [
    {"type": "rsi_threshold", "threshold": 30, "direction": "below", "name": "RSI30"},
    {"type": "rsi_threshold", "threshold": 25, "direction": "below", "name": "RSI25"},
]

comparison = PatternScanner.compare_patterns(
    data_path="data/ES_1m_2025-03.parquet",
    pattern_configs=patterns,
    include_outcomes=True
)

# Find best
best = max(comparison.items(), key=lambda x: x[1].success_rate or 0)
print(f"Best pattern: {best[0]}")

# 2. Build strategy with best pattern
strategy = StrategyBuilder.from_template(
    "mean_reversion",
    oversold=25,  # From best pattern
    start_date="2025-03-01",
    end_date="2025-03-15"
)

# 3. Test with model
result = SimulationRunner.run(
    strategy_config=strategy,
    model_path="models/my_model.pt",
    start_date="2025-03-01",
    end_date="2025-03-15"
)

print(result.summary())
```

### Workflow 2: Parameter Sweep

```python
from src.tools import StrategyBuilder, SimulationRunner

# Test different parameters
results = []
for oversold in [20, 25, 30]:
    for overbought in [70, 75, 80]:
        strategy = StrategyBuilder.from_template(
            "mean_reversion",
            oversold=oversold,
            overbought=overbought,
            start_date="2025-03-01",
            end_date="2025-03-15"
        )
        
        result = SimulationRunner.run(
            strategy_config=strategy,
            model_path="models/my_model.pt",
            start_date="2025-03-01",
            end_date="2025-03-15"
        )
        
        results.append({
            'oversold': oversold,
            'overbought': overbought,
            'win_rate': result.win_rate,
            'pnl': result.total_pnl
        })

# Find best
best = max(results, key=lambda x: x['pnl'])
print(f"Best params: oversold={best['oversold']}, overbought={best['overbought']}")
print(f"Win Rate: {best['win_rate']:.1%}, PnL: ${best['pnl']:.2f}")
```

### Workflow 3: Iterative Refinement

```python
from src.tools import (
    StrategyBuilder, SimulationRunner,
    StrategyValidator, PatternScanner
)

# 1. Start with template
strategy = StrategyBuilder.from_template("opening_range")

# 2. Validate
StrategyValidator.quick_check(strategy)

# 3. Run simulation
result = SimulationRunner.run(
    strategy_config=strategy,
    model_path="models/my_model.pt",
    start_date="2025-03-01",
    end_date="2025-03-15"
)

print(f"Baseline: Win Rate = {result.win_rate:.1%}")

# 4. Refine based on results
if result.win_rate < 0.55:
    # Try tighter stops
    strategy_v2 = StrategyBuilder.from_template(
        "opening_range",
        stop_atr=0.8,  # Tighter
        tp_multiple=2.0,  # Higher reward
        start_date="2025-03-01",
        end_date="2025-03-15"
    )
    
    result_v2 = SimulationRunner.run(
        strategy_config=strategy_v2,
        model_path="models/my_model.pt",
        start_date="2025-03-01",
        end_date="2025-03-15"
    )
    
    print(f"Refined: Win Rate = {result_v2.win_rate:.1%}")
```

---

## Best Practices

### 1. Always Validate

```python
# ✓ Good
strategy = StrategyBuilder.from_template("mean_reversion")
StrategyValidator.quick_check(strategy)
result = SimulationRunner.run(strategy_config=strategy, ...)

# ✗ Bad
strategy = StrategyBuilder.from_template("mean_reversion")
result = SimulationRunner.run(strategy_config=strategy, ...)  # Might fail!
```

### 2. Use Templates First

```python
# ✓ Good - Start with template
strategy = StrategyBuilder.from_template("mean_reversion", oversold=25)

# ✗ Harder - Build from scratch
strategy = StrategyBuilder.create(
    name="custom",
    scanner_id="rsi_extreme",
    scanner_params={"oversold": 25, ...},
    ...
)
```

### 3. Discover Patterns Before Building

```python
# ✓ Good - Data-driven
patterns = PatternScanner.compare_patterns(...)
best_pattern = max(patterns.items(), key=lambda x: x[1].success_rate)
strategy = StrategyBuilder.from_template(..., **best_pattern_params)

# ✗ Bad - Guessing
strategy = StrategyBuilder.from_template(..., oversold=30)  # Why 30?
```

### 4. Save Simulation Results

```python
result = SimulationRunner.run(...)

# Save for later analysis
SimulationRunner.save_results(result, "results/run_001.json")

# Load later
loaded = SimulationRunner.load_results("results/run_001.json")
```

### 5. Use Appropriate Date Ranges

```python
# ✓ Good - Meaningful range
start_date = "2025-03-01"
end_date = "2025-03-31"  # 1 month

# ✗ Bad - Too short
start_date = "2025-03-01"
end_date = "2025-03-03"  # 2 days - not statistically significant

# ✗ Bad - Too long for initial testing
start_date = "2024-01-01"
end_date = "2025-12-31"  # 2 years - too slow for iteration
```

---

## Tips for Agents

1. **Start Simple**: Use templates before building custom strategies
2. **Iterate Quickly**: Short date ranges for testing, longer for validation
3. **Validate Early**: Catch errors before wasting compute time
4. **Save Everything**: Save results for comparison and analysis
5. **Use the Catalog**: Discover tools via `ToolCatalog.print_catalog()`
6. **Read Examples**: Each tool has usage examples in docstrings
7. **Check Outcomes**: Use `PatternScanner` to validate assumptions
8. **Compose Smartly**: Use `TriggerComposer` for complex conditions
9. **Track Models**: Always specify which model was used in simulation
10. **Review Trades**: Inspect `result.trades` and `result.decisions` for insights

---

## Additional Resources

- `/src/tools/catalog.py` - Full tool catalog implementation
- `/src/tools/strategy_builder.py` - Strategy building utilities
- `/src/tools/simulation_runner.py` - Simulation execution
- `/src/tools/pattern_scanner.py` - Pattern discovery
- `/src/tools/trigger_composer.py` - Trigger composition
- `/src/tools/validation.py` - Strategy validation
- `/docs/THREE_LANES.md` - Architecture overview
- `/plan_for_this_software.md` - Development roadmap

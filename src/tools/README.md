# Agent Tools Package

Comprehensive toolkit for autonomous agents to build, test, and analyze trading strategies.

## Overview

This package provides a standardized, modular, and discoverable set of tools that enable agents to:

1. **Discover** available capabilities through the Tool Catalog
2. **Build** strategies from templates or custom components
3. **Compose** complex triggers and conditions
4. **Validate** strategies before execution
5. **Simulate** strategies with model inference
6. **Analyze** pattern occurrences and success rates
7. **Iterate** and refine strategies based on results

## Key Components

### 🗂️ Tool Catalog (`catalog.py`)

Central discovery system for all available tools, triggers, scanners, models, and utilities.

```python
from src.tools import ToolCatalog

catalog = ToolCatalog()
catalog.print_catalog()  # See everything

# Search and discover
triggers = catalog.list_by_category('trigger')
info = catalog.get_info('time_trigger')
```

### 🏗️ Strategy Builder (`strategy_builder.py`)

Build complete strategies from modular components or templates.

```python
from src.tools import StrategyBuilder

# From template
strategy = StrategyBuilder.from_template(
    "mean_reversion",
    start_date="2025-03-01",
    end_date="2025-03-15",
    oversold=25
)

# Custom
strategy = StrategyBuilder.create(
    name="Custom Strategy",
    scanner_id="level_proximity",
    scanner_params={"atr_threshold": 0.3},
    bracket={"type": "atr", "stop_atr": 1.0, "tp_atr": 2.0}
)
```

### 🎮 Simulation Runner (`simulation_runner.py`)

Run strategies with model inference in simulation mode.

**Addresses requirement**: Pulls associated model, runs inference on strategy triggers, generates trade results with playback data.

```python
from src.tools import SimulationRunner

result = SimulationRunner.run(
    strategy_config=strategy,
    model_path="models/my_model.pt",
    start_date="2025-03-01",
    end_date="2025-03-15"
)

print(result.summary())
# Shows: trades, win rate, PnL, model inference count, etc.
```

### 🔍 Pattern Scanner (`pattern_scanner.py`)

Discover pattern occurrences and validate assumptions.

```python
from src.tools import PatternScanner

results = PatternScanner.scan(
    data_path="data/ES_1m_2025-03.parquet",
    pattern_type="candle_pattern",
    pattern_config={"patterns": ["hammer"]},
    include_outcomes=True
)

# Shows: occurrence count, success rate, best times
PatternScanner.print_pattern_report(results)
```

### 🔧 Trigger Composer (`trigger_composer.py`)

Combine triggers with logical operators (AND, OR, SEQUENCE).

```python
from src.tools import TriggerComposer

# Time AND RSI
composite = TriggerComposer.AND([
    {"type": "time", "hour": 10, "minute": 0},
    {"type": "rsi_threshold", "threshold": 30, "direction": "below"}
])

# Use in strategy
strategy = StrategyBuilder.create(
    name="Composite Strategy",
    scanner_id="modular",
    scanner_params={"trigger_config": composite.to_dict()}
)
```

### ✅ Strategy Validator (`validation.py`)

Validate strategies before execution to catch errors early.

```python
from src.tools import StrategyValidator

result = StrategyValidator.validate(strategy)
result.print_report()

# Quick check (raises on error)
StrategyValidator.quick_check(strategy)
```

## Available Templates

- **mean_reversion**: Trade RSI extremes with level confluence
- **opening_range**: First bar breakout strategy
- **time_based**: Fixed time entries with model confirmation
- **pattern_following**: Candle pattern-based entries
- **level_bounce**: Price rejections at key levels

## Tool Categories

All tools are organized into categories for easy discovery:

- **Triggers**: Atomic entry signals (time, candle patterns, indicator thresholds)
- **Scanners**: Setup detection systems (when to check for entries)
- **Brackets**: Risk management (stops and targets)
- **Models**: Neural models for decision-making
- **Indicators**: Technical indicators
- **Utilities**: Helper tools (builders, validators, composers)

## Integration with Skills Registry

All tools are automatically registered in the agent skills registry:

```python
from src.skills.registry import list_available_skills

print(list_available_skills())
```

New skills include:
- `list_all_tools()` - Discover all capabilities
- `run_simulation()` - Execute strategy simulations
- `scan_pattern()` - Analyze pattern occurrences
- `build_strategy_from_template()` - Quick strategy creation
- `validate_strategy()` - Pre-execution validation
- `compose_triggers_and()` / `compose_triggers_or()` - Complex conditions

## Documentation

See `/docs/AGENT_TOOLS_GUIDE.md` for comprehensive guide with:
- Quick start examples
- Common workflows
- Best practices
- Advanced composition techniques
- Tips for autonomous agents

## Design Principles

1. **Discoverable**: All tools cataloged and searchable
2. **Modular**: Components can be mixed and matched
3. **Standardized**: Consistent interfaces across all tools
4. **Validated**: Built-in validation prevents errors
5. **Documented**: Examples and descriptions for everything
6. **Composable**: Simple tools combine into complex strategies
7. **Testable**: Easy to validate before execution

## Usage Example

Complete workflow from discovery to simulation:

```python
from src.tools import (
    ToolCatalog, StrategyBuilder, SimulationRunner,
    PatternScanner, StrategyValidator
)

# 1. Discover available tools
catalog = ToolCatalog()
catalog.print_catalog('trigger')

# 2. Analyze patterns
pattern_results = PatternScanner.scan(
    data_path="data/ES_1m_2025-03.parquet",
    pattern_type="rsi_threshold",
    pattern_config={"threshold": 30, "direction": "below"},
    include_outcomes=True
)
print(f"RSI<30 success rate: {pattern_results.success_rate:.1%}")

# 3. Build strategy
strategy = StrategyBuilder.from_template(
    "mean_reversion",
    start_date="2025-03-01",
    end_date="2025-03-15",
    oversold=30,  # From pattern analysis
    overbought=70
)

# 4. Validate
StrategyValidator.quick_check(strategy)

# 5. Simulate with model
result = SimulationRunner.run(
    strategy_config=strategy,
    model_path="models/my_model.pt",
    start_date="2025-03-01",
    end_date="2025-03-15"
)

# 6. Review results
print(result.summary())
print(f"Model made {result.inference_count} inferences")
print(f"Generated {len(result.trades)} trades")
print(f"Win Rate: {result.win_rate:.1%}")
print(f"Total PnL: ${result.total_pnl:.2f}")
```

## Benefits for Agents

✅ **Organized**: Everything in one place, easy to find
✅ **Standardized**: Consistent patterns across all tools
✅ **Scalable**: Easy to add new patterns and strategies
✅ **Iterative**: Quick feedback loop for refinement
✅ **Safe**: Validation before execution
✅ **Complete**: Full pipeline from discovery to results
✅ **Autonomous**: Agents can work independently
✅ **Model-Aware**: Proper integration with model inference

## Future Enhancements

- [ ] More strategy templates
- [ ] Advanced pattern recognition
- [ ] Multi-model voting systems
- [ ] Automatic parameter optimization
- [ ] Portfolio-level analysis
- [ ] Risk management presets
- [ ] Performance attribution tools

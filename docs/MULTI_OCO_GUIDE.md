# Multi-OCO Grid Trading - Feature Guide

## Overview

The Multi-OCO Grid Trading feature allows agents to **test multiple OCO (One-Cancels-Other) bracket configurations simultaneously** on the same trade trigger, with **limit order entries** for better execution.

## Why Multi-OCO?

### The Problem
- Testing different stop/target combinations requires multiple backtests
- Each backtest is time-consuming
- Hard to compare configurations fairly
- Market orders cause slippage

### The Solution
- Test 3-12 OCO configurations in **one** backtest
- Use limit orders with configurable entry offsets
- Compare performance across all configs
- Automatically identify best configuration
- Statistical confidence from multiple data points

## Quick Start

```python
from src.tools import StrategyBuilder, SimulationRunner
from src.sim.multi_oco import MultiOCOConfig

# 1. Create multi-OCO grid
multi_oco = MultiOCOConfig.create_tight_medium_wide(
    direction="LONG",
    entry_offset=0.25  # Limit order 0.25 ATR from trigger
)

# 2. Build strategy
strategy = StrategyBuilder.from_template("opening_range")

# 3. Run simulation with multi-OCO
result = SimulationRunner.run(
    strategy_config=strategy,
    model_path="models/my_model.pt",
    start_date="2025-03-01",
    end_date="2025-03-15",
    multi_oco_config=multi_oco  # Add multi-OCO config
)

# 4. Check results
print(result.summary())
print(f"Best OCO: {result.best_oco_config}")
```

## Pre-Built Grid Templates

### 1. Tight-Medium-Wide (3 OCOs)
Test 3 target levels: 1R, 1.5R, 2R

```python
multi_oco = MultiOCOConfig.create_tight_medium_wide(
    direction="LONG",
    entry_offset=0.25
)

# Creates:
# - OCO 1: 1.0R target (tight)
# - OCO 2: 1.5R target (medium)
# - OCO 3: 2.0R target (wide)
# All with limit entry 0.25 ATR from trigger
```

### 2. Standard Grid (Customizable)
Test all combinations of targets × entry offsets

```python
multi_oco = MultiOCOConfig.create_standard_grid(
    direction="LONG",
    base_stop_atr=1.0,
    tp_multiples=[1.0, 1.5, 2.0, 2.5],  # 4 targets
    entry_offsets=[0.1, 0.25, 0.5]      # 3 entry levels
)

# Creates: 4 × 3 = 12 OCO brackets
```

### 3. Entry Ladder (Test Entry Distances)
Same target, different entry levels

```python
multi_oco = MultiOCOConfig.create_entry_ladder(
    direction="LONG",
    entry_offsets=[0.1, 0.25, 0.5],  # Close, medium, far
    tp_multiple=1.5,
    stop_atr=1.0
)

# Creates 3 OCOs with different entry points
```

## How It Works

### Trigger → Limit Orders
1. **Trigger fires** (e.g., RSI < 30 at 10:00 AM)
2. **Multiple limit orders placed** at different offsets
3. **Limit orders fill** based on price action
4. **Each filled order** creates OCO bracket (stop + target)
5. **All OCOs tracked** independently
6. **Results compared** to find best config

### Limit Order Entry
```
Trigger Price: 4500
Entry Offset: 0.25 ATR (= 1.25 points)

LONG: Limit Buy @ 4498.75 (trigger - offset)
SHORT: Limit Sell @ 4501.25 (trigger + offset)

Better fill price than market order!
```

### Multiple OCOs on Same Trigger
```
Trigger at Bar 100:

OCO 1: Entry 0.1 ATR, Target 1.0R
OCO 2: Entry 0.1 ATR, Target 1.5R
OCO 3: Entry 0.1 ATR, Target 2.0R
OCO 4: Entry 0.25 ATR, Target 1.0R
OCO 5: Entry 0.25 ATR, Target 1.5R
OCO 6: Entry 0.25 ATR, Target 2.0R

All tested simultaneously!
```

## Results & Analysis

### SimulationResult Enhancements
```python
result = SimulationRunner.run(..., multi_oco_config=multi_oco)

# New fields:
result.multi_oco_results  # List of MultiOCOResult objects
result.best_oco_config     # Name of best performing OCO

# Summary includes multi-OCO stats
print(result.summary())
```

### MultiOCOResult Object
```python
for grid_result in result.multi_oco_results:
    print(grid_result.summary())
    
# Output:
# Multi-OCO Grid Result: tight_medium_wide
# ============================================================
# Total OCOs: 3
# Filled: 3 (100%)
# Winners: 2
# Losers: 1
# Total PnL: $45.00
# Best OCO: medium_1.5R
#
# Individual OCO Results:
# ------------------------------------------------------------
#   ✓ tight_1R          PnL: $10.00   Status: CLOSED_TP
#   ✓ medium_1.5R       PnL: $25.00   Status: CLOSED_TP
#   ✓ wide_2R           PnL: $10.00   Status: CLOSED_SL
```

### Performance Analysis
```python
from src.sim.multi_oco import MultiOCOHelper

# Analyze across multiple runs
analysis = MultiOCOHelper.analyze_grid_performance(result.multi_oco_results)

# Print detailed statistics
MultiOCOHelper.print_analysis(analysis)

# Output:
# Multi-OCO Performance Analysis
# ================================================================
# Total Grids Analyzed: 45
#
# Best by Total PnL: oco_offset0.25_tp1.5
# Best by Win Rate: oco_offset0.1_tp1.0
#
# Detailed OCO Statistics:
# ================================================================
# OCO Name              Trades  Fill%   Win%    Avg PnL   Total PnL
# ----------------------------------------------------------------
# tight_1R              45      93.3%   62.2%   $12.50    $562.50
# medium_1.5R           45      88.9%   55.6%   $18.75    $750.00  ← Best
# wide_2R               45      71.1%   65.6%   $15.00    $480.00
```

## Use Cases

### 1. Find Optimal Target
```python
# Which take-profit multiple works best?
multi_oco = MultiOCOConfig.create_standard_grid(
    tp_multiples=[1.0, 1.5, 2.0, 2.5, 3.0]
)

result = SimulationRunner.run(..., multi_oco_config=multi_oco)
print(f"Best target: {result.best_oco_config}")
# → "oco_offset0.25_tp1.5" means 1.5R is optimal
```

### 2. Find Optimal Entry Offset
```python
# What limit order distance is best?
multi_oco = MultiOCOConfig.create_entry_ladder(
    entry_offsets=[0.05, 0.1, 0.25, 0.5, 1.0],
    tp_multiple=1.5
)

result = SimulationRunner.run(..., multi_oco_config=multi_oco)
# Compare fill rates and profitability
```

### 3. Test Complete Grid
```python
# Test ALL combinations
multi_oco = MultiOCOConfig.create_standard_grid(
    tp_multiples=[1.0, 1.5, 2.0],
    entry_offsets=[0.1, 0.25, 0.5]
)
# Tests 9 configurations in one run!
```

## Configuration Options

### MultiOCOConfig Parameters
```python
MultiOCOConfig(
    direction="LONG",          # or "SHORT"
    use_limit_entry=True,      # Use limit orders
    oco_configs=[...],         # List of OCOConfig objects
    grid_name="my_grid"        # Descriptive name
)
```

### OCOConfig Parameters
```python
OCOConfig(
    direction="LONG",
    entry_type="LIMIT",        # "MARKET" or "LIMIT"
    entry_offset_atr=0.25,     # Limit offset in ATR
    stop_atr=1.0,              # Stop loss distance
    tp_multiple=1.5,           # Target as multiple of risk
    max_bars=200,              # Max time in trade
    name="my_oco"              # Unique identifier
)
```

## Best Practices

### 1. Start Small
```python
# Begin with 3 OCOs
multi_oco = MultiOCOConfig.create_tight_medium_wide("LONG")
```

### 2. Reasonable Ranges
```python
# Don't test too many at once (increases noise)
# Good: 3-9 OCOs
# Avoid: 20+ OCOs

# Good
tp_multiples=[1.0, 1.5, 2.0]  # ✓

# Too many
tp_multiples=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]  # ✗
```

### 3. Entry Offsets
```python
# Common values:
# - 0.1 ATR: Very close to trigger (high fill rate)
# - 0.25 ATR: Standard (balanced)
# - 0.5 ATR: Conservative (lower fill rate, better price)

# Test what works for your market
```

### 4. Analyze Results
```python
# Don't just look at total PnL
# Consider:
# - Fill rates (too low = missed trades)
# - Win rates (varies by target size)
# - Risk/reward balance
# - Consistency across time periods

analysis = MultiOCOHelper.analyze_grid_performance(results)
MultiOCOHelper.print_analysis(analysis)
```

## Cookbook Recipe

See the complete recipe:
```python
from src.tools.cookbook import AgentCookbook

print(AgentCookbook.get_recipe('multi_oco_testing'))
```

## Benefits Summary

✅ **Efficiency**: Test 9 configs in time of 1  
✅ **Better Execution**: Limit orders vs market orders  
✅ **Data-Driven**: Statistical comparison of configs  
✅ **Fill Rate Insights**: Understand entry level impact  
✅ **Automatic Optimization**: Best config identified  
✅ **Flexible**: Custom grids or pre-built templates  
✅ **Scalable**: Easy to add new configurations  

## Integration Points

### With StrategyBuilder
```python
strategy = StrategyBuilder.from_template("mean_reversion")
multi_oco = MultiOCOConfig.create_tight_medium_wide("LONG")

result = SimulationRunner.run(
    strategy_config=strategy,
    multi_oco_config=multi_oco,
    ...
)
```

### With Pattern Scanner
```python
# Find best pattern first
patterns = PatternScanner.compare_patterns(...)

# Then test multi-OCO on best pattern
strategy = StrategyBuilder.from_template(best_pattern)
multi_oco = MultiOCOConfig.create_standard_grid(...)

result = SimulationRunner.run(...)
```

### With Model Comparison
```python
# Test multiple models with multi-OCO
multi_oco = MultiOCOConfig.create_tight_medium_wide("LONG")

for model_path in ["model_v1.pt", "model_v2.pt"]:
    result = SimulationRunner.run(
        model_path=model_path,
        multi_oco_config=multi_oco,
        ...
    )
    # Compare which model + OCO combo works best
```

## Examples

See complete working examples in:
- `src/tools/cookbook.py` - Recipe #7
- `examples/agent_tools_demo.py` - Multi-OCO demo
- `docs/AGENT_TOOLS_GUIDE.md` - Advanced patterns

## Summary

Multi-OCO Grid Trading enables **efficient, data-driven optimization** of bracket configurations with **limit order execution**. Test multiple combinations simultaneously to find the optimal risk/reward setup for your strategies.

**Ready to optimize your brackets? Start with `create_tight_medium_wide()`!** 🎯

# MLang2 - Modular Strategy Development for Autonomous Agents

**Organized, Standardized, Scalable Trading Strategy Framework**

---

## 🎯 For Autonomous Agents

This repository provides a complete toolkit for building, testing, and refining trading strategies autonomously. Everything is **discoverable**, **modular**, and **standardized** to enable independent agent operation.

### Quick Start

```python
from src.tools import StrategyBuilder, SimulationRunner

# Build a strategy
strategy = StrategyBuilder.from_template("mean_reversion")

# Run simulation with model
result = SimulationRunner.run(
    strategy_config=strategy,
    model_path="path/to/model.pt",
    start_date="2025-03-01",
    end_date="2025-03-15"
)

print(result.summary())
```

---

## 📚 Documentation

### Primary Resources
- **[Agent Tools Guide](docs/AGENT_TOOLS_GUIDE.md)** - Comprehensive guide for using all tools
- **[Tools Package README](src/tools/README.md)** - Overview of the tools package
- **[Three Lanes Architecture](docs/THREE_LANES.md)** - System architecture (TRAIN/REPLAY/SCAN)
- **[Software Plan](plan_for_this_software.md)** - Development roadmap

### Quick Access
- Run `./agent_tools.py` for interactive menu
- Run `./examples/agent_tools_demo.py` for complete examples
- Import `from src.tools.cookbook import AgentCookbook` for recipes

---

## 🛠️ What Can Agents Do?

### 1. **Discover** Capabilities
```python
from src.tools import ToolCatalog

catalog = ToolCatalog()
catalog.print_catalog()  # See everything available
```

Available tool categories:
- **Triggers**: Entry signals (time, candle patterns, indicators)
- **Scanners**: Setup detection systems
- **Brackets**: Risk management (stops/targets)
- **Models**: Neural networks for decisions
- **Indicators**: Technical indicators
- **Utilities**: Builders, validators, composers

### 2. **Build** Strategies

#### From Templates
```python
from src.tools import StrategyBuilder

strategy = StrategyBuilder.from_template(
    "opening_range",  # or: mean_reversion, time_based, pattern_following, level_bounce
    start_date="2025-03-01",
    end_date="2025-03-15"
)
```

#### Custom
```python
strategy = StrategyBuilder.create(
    name="My Strategy",
    scanner_id="level_proximity",
    scanner_params={"atr_threshold": 0.3},
    bracket={"type": "atr", "stop_atr": 1.0, "tp_atr": 2.0}
)
```

### 3. **Validate** Before Running
```python
from src.tools import StrategyValidator

result = StrategyValidator.validate(strategy)
result.print_report()

# Or quick check (raises on error)
StrategyValidator.quick_check(strategy)
```

### 4. **Simulate** with Models

**Key Feature**: Simulation mode pulls associated models, runs inference on triggers, and generates complete trade results with playback data.

```python
from src.tools import SimulationRunner

result = SimulationRunner.run(
    strategy_config=strategy,
    model_path="models/my_model.pt",
    start_date="2025-03-01",
    end_date="2025-03-15"
)

# Results include:
# - Complete trade history
# - Model inference count
# - Performance metrics
# - Win rate, PnL, Sharpe, drawdown
```

### 5. **Analyze** Patterns
```python
from src.tools import PatternScanner

results = PatternScanner.scan(
    data_path="data/ES_1m_2025-03.parquet",
    pattern_type="rsi_threshold",
    pattern_config={"threshold": 30, "direction": "below"},
    include_outcomes=True
)

PatternScanner.print_pattern_report(results)
```

### 6. **Compose** Complex Triggers
```python
from src.tools import TriggerComposer

# AND logic
composite = TriggerComposer.AND([
    {"type": "time", "hour": 10, "minute": 0},
    {"type": "rsi_threshold", "threshold": 30, "direction": "below"}
])

# OR logic
alternative = TriggerComposer.OR([
    {"type": "candle_pattern", "patterns": ["hammer"]},
    {"type": "rsi_threshold", "threshold": 25, "direction": "below"}
])

# SEQUENCE logic
sequence = TriggerComposer.SEQUENCE([
    {"type": "rsi_threshold", "threshold": 30, "direction": "below"},
    {"type": "candle_pattern", "patterns": ["hammer"]}
], max_bars=3)
```

### 7. **Iterate** and Refine

```python
# Test multiple parameters
for oversold in [25, 30, 35]:
    strategy = StrategyBuilder.from_template(
        "mean_reversion",
        oversold=oversold
    )
    result = SimulationRunner.run(...)
    # Compare results
```

---

## 🔄 Common Workflows

### Workflow 1: Discover → Build → Test
1. Use `PatternScanner` to find best patterns
2. Use `StrategyBuilder` with best pattern
3. Use `SimulationRunner` to test with model
4. Review results and iterate

### Workflow 2: Parameter Optimization
1. Define parameter grid
2. Test all combinations with `SimulationRunner`
3. Select best parameters
4. Validate on different data

### Workflow 3: Model Comparison
1. Build strategy
2. Use `SimulationRunner.compare_models()` with multiple models
3. Select best-performing model
4. Deploy with confidence

See [Cookbook](src/tools/cookbook.py) for detailed recipes!

---

## 📦 Package Structure

```
mlang2/
├── src/
│   ├── tools/           # 🔥 Agent toolkit (NEW)
│   │   ├── catalog.py          # Tool discovery
│   │   ├── strategy_builder.py # Strategy construction
│   │   ├── simulation_runner.py # Model inference simulation
│   │   ├── trigger_composer.py  # Complex conditions
│   │   ├── pattern_scanner.py   # Pattern analysis
│   │   ├── validation.py        # Pre-execution checks
│   │   └── cookbook.py          # Ready-made recipes
│   │
│   ├── policy/          # Strategy components
│   │   ├── triggers/    # Entry signals
│   │   ├── scanners.py  # Setup detection
│   │   └── brackets.py  # Risk management
│   │
│   ├── models/          # Neural models
│   ├── sim/             # Simulation engine
│   ├── features/        # Feature engineering
│   ├── labels/          # Outcome labeling
│   └── skills/          # Agent skills registry
│
├── docs/
│   ├── AGENT_TOOLS_GUIDE.md  # Comprehensive guide
│   ├── THREE_LANES.md         # Architecture
│   └── IMPLEMENTATION_SUMMARY.md
│
├── examples/
│   └── agent_tools_demo.py    # Complete examples
│
├── agent_tools.py      # Interactive menu
└── plan_for_this_software.md
```

---

## 🎓 Learning Path

1. **Start Here**: Run `./agent_tools.py` for interactive tour
2. **Examples**: Run `./examples/agent_tools_demo.py` to see all features
3. **Documentation**: Read [docs/AGENT_TOOLS_GUIDE.md](docs/AGENT_TOOLS_GUIDE.md)
4. **Cookbook**: Use recipes from `AgentCookbook` for common tasks
5. **Build**: Create your first strategy with `StrategyBuilder`
6. **Test**: Run simulations with your models
7. **Iterate**: Refine based on results

---

## ✨ Key Features for Agents

### ✅ Organized
- Everything cataloged and discoverable
- Consistent structure across all tools
- Clear categorization

### ✅ Standardized
- Uniform interfaces
- Predictable behavior
- JSON-serializable configs

### ✅ Scalable
- Easy to add new triggers, scanners, patterns
- Plugin architecture via registries
- No hard-coded dependencies

### ✅ Autonomous-Ready
- Agents can discover capabilities independently
- No manual intervention needed
- Complete feedback loop from discovery to results

### ✅ Model-Aware
- Proper model integration in simulation
- Track model inference count
- Link strategies to models
- Complete trade results with decisions

### ✅ Validated
- Pre-execution validation
- Error checking before compute time
- Detailed error/warning reports

---

## 🚀 Next Steps

### For Agents
1. Discover tools: `ToolCatalog().print_catalog()`
2. Choose a template: `StrategyBuilder.list_templates()`
3. Build strategy: `StrategyBuilder.from_template(...)`
4. Validate: `StrategyValidator.quick_check(...)`
5. Simulate: `SimulationRunner.run(...)`
6. Analyze results and iterate!

### For Developers
- Add new triggers in `src/policy/triggers/`
- Add new scanners in `src/policy/library/`
- Add new templates in `StrategyBuilder.TEMPLATES`
- Everything auto-registers via catalogs

---

## 💡 Pro Tips

1. **Always validate** before running simulations
2. **Use templates** as starting points
3. **Discover patterns** before building strategies
4. **Save results** for comparison
5. **Iterate quickly** with short date ranges
6. **Use cookbook** for proven workflows
7. **Check catalog** for available tools

---

## 🔗 Integration Points

### Skills Registry
All tools are registered in `src/skills/registry.py`:
- `list_all_tools()` - Discover everything
- `run_simulation()` - Execute simulations
- `build_strategy_from_template()` - Quick creation
- `validate_strategy()` - Pre-check
- More...

### Run Modes
System supports three execution modes:
- **TRAIN**: Build and label training data
- **REPLAY**: Simulate with model inference
- **SCAN**: Analyze patterns (read-only)

### Model Roles
Models have explicit roles:
- **TRAINING_ONLY**: For labeling
- **REPLAY_ONLY**: For simulation
- **FROZEN_EVAL**: For validation
- **SCAN_ASSIST**: For discovery

---

## 📞 Support

- Documentation: `/docs/` directory
- Examples: `/examples/` directory
- Interactive: `./agent_tools.py`
- Cookbook: `from src.tools.cookbook import AgentCookbook`

---

## 🎯 Design Goals Achieved

✅ **Modular**: Components mix and match freely  
✅ **Correct**: Validation prevents errors  
✅ **Scalable**: Easy to add new patterns  
✅ **Organized**: Everything in its place  
✅ **Standardized**: Consistent interfaces  
✅ **Autonomous**: Agents work independently  
✅ **Documented**: Comprehensive guides  
✅ **Tested**: Validation before execution  

---

**Ready to build strategies? Start with `./agent_tools.py`!** 🚀

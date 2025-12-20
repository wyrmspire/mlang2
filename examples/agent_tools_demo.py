#!/usr/bin/env python3
"""
Complete Example: Building and Testing a Strategy

This example demonstrates a full workflow from discovery to deployment.
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools import (
    ToolCatalog,
    StrategyBuilder,
    SimulationRunner,
    PatternScanner,
    TriggerComposer,
    StrategyValidator,
    AgentCookbook
)


def example_1_quick_start():
    """Example 1: Quick start - build and test a strategy."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Quick Start")
    print("="*70)
    
    # Build strategy from template
    strategy = StrategyBuilder.from_template(
        "mean_reversion",
        start_date="2025-03-01",
        end_date="2025-03-15",
        oversold=30,
        overbought=70
    )
    
    print("\n✓ Strategy built from template")
    
    # Validate
    result = StrategyValidator.validate(strategy)
    if result.valid:
        print("✓ Strategy validated successfully")
    else:
        print("⚠️  Validation issues found:")
        result.print_report()
    
    print("\nStrategy would be ready to run with:")
    print("  SimulationRunner.run(strategy_config=strategy, model_path='...', ...)")


def example_2_discover_patterns():
    """Example 2: Discover which patterns work best."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Pattern Discovery")
    print("="*70)
    
    # Note: This would require actual data file
    print("\nThis example would scan for patterns:")
    print("""
    patterns = [
        {"type": "rsi_threshold", "threshold": 30, "direction": "below", "name": "RSI<30"},
        {"type": "candle_pattern", "patterns": ["hammer"], "name": "Hammer"},
    ]
    
    results = PatternScanner.compare_patterns(
        data_path="data/ES_1m_2025-03.parquet",
        pattern_configs=patterns,
        include_outcomes=True
    )
    
    PatternScanner.print_comparison_report(results)
    """)
    
    print("\n✓ Pattern discovery helps identify best entry signals")


def example_3_composite_trigger():
    """Example 3: Build complex entry conditions."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Composite Triggers")
    print("="*70)
    
    # Create composite: Time AND RSI
    composite = TriggerComposer.AND([
        {"type": "time", "hour": 10, "minute": 0},
        {"type": "rsi_threshold", "threshold": 30, "direction": "below"}
    ])
    
    print("\n✓ Created composite trigger: (Time=10:00 AND RSI<30)")
    print(f"  Trigger config: {composite.to_dict()}")
    
    # Use in strategy
    strategy = StrategyBuilder.create(
        name="Time + RSI Strategy",
        scanner_id="modular",
        scanner_params={
            "trigger_config": composite.to_dict(),
            "cooldown_bars": 30
        },
        bracket={"type": "atr", "stop_atr": 1.0, "tp_atr": 2.0},
        start_date="2025-03-01",
        end_date="2025-03-15"
    )
    
    print("✓ Strategy created with composite trigger")


def example_4_parameter_optimization():
    """Example 4: Find optimal parameters."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Parameter Optimization")
    print("="*70)
    
    print("\nThis example would test multiple parameter combinations:")
    print("""
    results = []
    
    for oversold in [25, 30, 35]:
        for overbought in [65, 70, 75]:
            strategy = StrategyBuilder.from_template(
                "mean_reversion",
                oversold=oversold,
                overbought=overbought,
                start_date="2025-03-01",
                end_date="2025-03-15"
            )
            
            result = SimulationRunner.run(
                strategy_config=strategy,
                model_path="path/to/model.pt",
                start_date="2025-03-01",
                end_date="2025-03-15"
            )
            
            results.append({
                'oversold': oversold,
                'overbought': overbought,
                'pnl': result.total_pnl
            })
    
    best = max(results, key=lambda x: x['pnl'])
    print(f"Best: oversold={best['oversold']}, overbought={best['overbought']}")
    """)
    
    print("\n✓ Parameter sweeps find optimal values")


def example_5_validation():
    """Example 5: Validate before running."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Strategy Validation")
    print("="*70)
    
    # Create intentionally problematic strategy
    strategy = StrategyBuilder.from_template(
        "mean_reversion",
        start_date="2025-03-15",  # After end_date!
        end_date="2025-03-01",
        stop_atr=-1.0,  # Invalid!
        oversold=30,
        overbought=70
    )
    
    print("\nValidating intentionally bad strategy...")
    result = StrategyValidator.validate(strategy)
    
    print(f"\nValid: {result.valid}")
    print(f"Errors: {len([i for i in result.issues if i.severity.value == 'ERROR'])}")
    print(f"Warnings: {len([i for i in result.issues if i.severity.value == 'WARNING'])}")
    
    print("\nValidation report:")
    result.print_report()
    
    print("\n✓ Validation catches errors before wasting compute time")


def example_6_tool_discovery():
    """Example 6: Discover available tools."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Tool Discovery")
    print("="*70)
    
    catalog = ToolCatalog()
    
    # List all triggers
    triggers = catalog.list_by_category('trigger')
    print(f"\n{len(triggers)} triggers available:")
    for t in triggers[:3]:
        print(f"  • {t.name}")
    print("  ...")
    
    # Search for tools
    time_tools = catalog.search('time')
    print(f"\n{len(time_tools)} tools related to 'time':")
    for t in time_tools:
        print(f"  • {t.name} ({t.category})")
    
    # Get detailed info
    info = catalog.get_info('time_trigger')
    print(f"\nDetailed info for 'time_trigger':")
    print(f"  Description: {info.description}")
    if info.examples:
        print(f"  Example: {info.examples[0]}")
    
    print("\n✓ Tool Catalog makes everything discoverable")


def example_7_cookbook():
    """Example 7: Use cookbook recipes."""
    print("\n" + "="*70)
    print("EXAMPLE 7: Cookbook Recipes")
    print("="*70)
    
    print("\nCookbook provides ready-to-use recipes:")
    
    recipes = AgentCookbook.list_all_recipes()
    print(f"\n{len(recipes)} recipes available:")
    for recipe in recipes:
        print(f"  • {recipe}")
    
    print("\nGet a recipe with:")
    print("  AgentCookbook.get_recipe('quick_start')")
    
    print("\n✓ Cookbook accelerates development with proven patterns")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("MLang2 Agent Tools - Complete Examples")
    print("="*70)
    print("\nThis script demonstrates all major capabilities.")
    print("\nNote: Some examples show code snippets rather than executing,")
    print("      since they require data files or trained models.")
    
    examples = [
        ("Quick Start", example_1_quick_start),
        ("Pattern Discovery", example_2_discover_patterns),
        ("Composite Triggers", example_3_composite_trigger),
        ("Parameter Optimization", example_4_parameter_optimization),
        ("Validation", example_5_validation),
        ("Tool Discovery", example_6_tool_discovery),
        ("Cookbook Recipes", example_7_cookbook),
    ]
    
    for i, (name, func) in enumerate(examples, 1):
        print(f"\n\n{'='*70}")
        print(f"Running Example {i}/{len(examples)}: {name}")
        print("="*70)
        try:
            func()
            print(f"\n✓ Example {i} completed successfully")
        except Exception as e:
            print(f"\n✗ Example {i} failed: {e}")
            import traceback
            traceback.print_exc()
        
        if i < len(examples):
            input("\nPress Enter to continue to next example...")
    
    print("\n\n" + "="*70)
    print("All Examples Complete!")
    print("="*70)
    print("\nNext Steps:")
    print("  1. Review the docs/AGENT_TOOLS_GUIDE.md for detailed documentation")
    print("  2. Run ./agent_tools.py for interactive menu")
    print("  3. Start building your own strategies!")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

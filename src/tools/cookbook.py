"""
Agent Cookbook - Practical examples and recipes for common tasks.

Quick reference for autonomous agents to accomplish specific goals.
"""

from typing import Dict, Any, List


class AgentCookbook:
    """
    Collection of practical recipes for agents.
    
    Each recipe is a complete, runnable example for a specific task.
    """
    
    @staticmethod
    def recipe_quick_start() -> str:
        """Recipe: Get started in 5 minutes."""
        return """
RECIPE: Quick Start - Your First Strategy Simulation
=====================================================

Goal: Build and test a mean reversion strategy in under 5 minutes.

Steps:
------

1. Import tools:
   ```python
   from src.tools import StrategyBuilder, SimulationRunner, StrategyValidator
   ```

2. Build strategy from template:
   ```python
   strategy = StrategyBuilder.from_template(
       "mean_reversion",
       start_date="2025-03-01",
       end_date="2025-03-15",
       oversold=30,
       overbought=70
   )
   ```

3. Validate (optional but recommended):
   ```python
   StrategyValidator.quick_check(strategy)
   ```

4. Run simulation with your trained model:
   ```python
   result = SimulationRunner.run(
       strategy_config=strategy,
       model_path="path/to/your/model.pt",
       start_date="2025-03-01",
       end_date="2025-03-15"
   )
   ```

5. Review results:
   ```python
   print(result.summary())
   ```

Done! You now have:
- Complete trade history
- Performance metrics
- Model inference statistics

Next: Try different templates or tweak parameters.
"""
    
    @staticmethod
    def recipe_find_best_pattern() -> str:
        """Recipe: Discover which pattern works best."""
        return """
RECIPE: Find Best Pattern
=========================

Goal: Discover which entry pattern has the highest success rate.

Steps:
------

1. Import tools:
   ```python
   from src.tools import PatternScanner
   ```

2. Define patterns to test:
   ```python
   patterns = [
       {"type": "rsi_threshold", "threshold": 30, "direction": "below", "name": "RSI<30"},
       {"type": "rsi_threshold", "threshold": 25, "direction": "below", "name": "RSI<25"},
       {"type": "candle_pattern", "patterns": ["hammer"], "name": "Hammer"},
       {"type": "candle_pattern", "patterns": ["doji"], "name": "Doji"},
   ]
   ```

3. Compare patterns:
   ```python
   results = PatternScanner.compare_patterns(
       data_path="data/ES_1m_2025-03.parquet",
       pattern_configs=patterns,
       include_outcomes=True
   )
   ```

4. Print comparison report:
   ```python
   PatternScanner.print_comparison_report(results)
   ```

5. Find best pattern:
   ```python
   best = max(results.items(), key=lambda x: x[1].success_rate or 0)
   print(f"\\nBest pattern: {best[0]}")
   print(f"Success rate: {best[1].success_rate:.1%}")
   print(f"Avg PnL: ${best[1].avg_forward_pnl:.2f}")
   print(f"Best time: {best[1].best_time_of_day}")
   ```

Use the insights to build your strategy!
"""
    
    @staticmethod
    def recipe_parameter_sweep() -> str:
        """Recipe: Find optimal parameters."""
        return """
RECIPE: Parameter Sweep
=======================

Goal: Find the best stop/target combination for a strategy.

Steps:
------

1. Import tools:
   ```python
   from src.tools import StrategyBuilder, SimulationRunner
   ```

2. Define parameter grid:
   ```python
   stop_values = [0.8, 1.0, 1.2]
   tp_multiples = [1.5, 2.0, 2.5]
   ```

3. Test all combinations:
   ```python
   results = []
   
   for stop_atr in stop_values:
       for tp_multiple in tp_multiples:
           strategy = StrategyBuilder.from_template(
               "opening_range",
               start_date="2025-03-01",
               end_date="2025-03-15",
               stop_atr=stop_atr,
               tp_multiple=tp_multiple
           )
           
           result = SimulationRunner.run(
               strategy_config=strategy,
               model_path="path/to/model.pt",
               start_date="2025-03-01",
               end_date="2025-03-15"
           )
           
           results.append({
               'stop_atr': stop_atr,
               'tp_multiple': tp_multiple,
               'win_rate': result.win_rate,
               'total_pnl': result.total_pnl,
               'sharpe': result.sharpe_ratio
           })
   ```

4. Find best parameters:
   ```python
   # Sort by total PnL
   best_pnl = max(results, key=lambda x: x['total_pnl'])
   
   # Or by Sharpe ratio
   best_sharpe = max(results, key=lambda x: x['sharpe'])
   
   # Or by win rate
   best_wr = max(results, key=lambda x: x['win_rate'])
   
   print(f"Best by PnL: stop={best_pnl['stop_atr']}, tp={best_pnl['tp_multiple']}")
   print(f"  PnL: ${best_pnl['total_pnl']:.2f}, Sharpe: {best_pnl['sharpe']:.2f}")
   ```

5. Use best parameters in production strategy!
"""
    
    @staticmethod
    def recipe_composite_trigger() -> str:
        """Recipe: Build complex entry conditions."""
        return """
RECIPE: Composite Trigger
=========================

Goal: Create strategy with multiple entry conditions.

Example: Enter only at 10am OR 2pm AND when RSI < 30

Steps:
------

1. Import tools:
   ```python
   from src.tools import TriggerComposer, StrategyBuilder, SimulationRunner
   ```

2. Create composite trigger:
   ```python
   # Time OR Time
   time_trigger = TriggerComposer.OR([
       {"type": "time", "hour": 10, "minute": 0},
       {"type": "time", "hour": 14, "minute": 0}
   ])
   
   # Time AND RSI
   composite = TriggerComposer.AND([
       time_trigger.to_dict(),
       {"type": "rsi_threshold", "threshold": 30, "direction": "below"}
   ])
   ```

3. Use in strategy:
   ```python
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
   ```

4. Test it:
   ```python
   result = SimulationRunner.run(
       strategy_config=strategy,
       model_path="path/to/model.pt",
       start_date="2025-03-01",
       end_date="2025-03-15"
   )
   print(result.summary())
   ```

Complex conditions without complex code!
"""
    
    @staticmethod
    def recipe_compare_models() -> str:
        """Recipe: Find which model performs best."""
        return """
RECIPE: Compare Models
=====================

Goal: Test multiple models on the same strategy.

Steps:
------

1. Import tools:
   ```python
   from src.tools import StrategyBuilder, SimulationRunner
   ```

2. Build strategy:
   ```python
   strategy = StrategyBuilder.from_template(
       "opening_range",
       start_date="2025-03-01",
       end_date="2025-03-15"
   )
   ```

3. List models to compare:
   ```python
   model_paths = [
       "models/model_v1.pt",
       "models/model_v2.pt",
       "models/model_v3.pt"
   ]
   ```

4. Compare:
   ```python
   results = SimulationRunner.compare_models(
       strategy_config=strategy,
       model_paths=model_paths,
       start_date="2025-03-01",
       end_date="2025-03-15"
   )
   ```

5. Find best:
   ```python
   print("\\nModel Comparison:")
   print("-" * 80)
   
   for model_path, result in results.items():
       print(f"\\n{model_path}:")
       print(f"  Win Rate: {result.win_rate:.1%}")
       print(f"  Total PnL: ${result.total_pnl:.2f}")
       print(f"  Sharpe: {result.sharpe_ratio:.2f}")
       print(f"  Trades: {result.total_trades}")
   
   best = max(results.items(), key=lambda x: x[1].total_pnl)
   print(f"\\nBest Model: {best[0]}")
   print(f"PnL: ${best[1].total_pnl:.2f}")
   ```

Use the winning model!
"""
    
    @staticmethod
    def recipe_iterative_refinement() -> str:
        """Recipe: Iteratively improve a strategy."""
        return """
RECIPE: Iterative Refinement
============================

Goal: Systematically improve strategy performance.

Steps:
------

1. Start with baseline:
   ```python
   from src.tools import StrategyBuilder, SimulationRunner, StrategyValidator
   
   # Baseline strategy
   strategy_v1 = StrategyBuilder.from_template("opening_range")
   StrategyValidator.quick_check(strategy_v1)
   
   result_v1 = SimulationRunner.run(
       strategy_config=strategy_v1,
       model_path="path/to/model.pt",
       start_date="2025-03-01",
       end_date="2025-03-15"
   )
   
   print(f"Baseline - Win Rate: {result_v1.win_rate:.1%}, PnL: ${result_v1.total_pnl:.2f}")
   ```

2. Analyze results and identify issue:
   ```python
   # Check avg win vs avg loss
   if result_v1.avg_loss > result_v1.avg_win:
       print("Issue: Losing more than winning - try wider targets")
   ```

3. Iterate:
   ```python
   # Try v2 with wider targets
   strategy_v2 = StrategyBuilder.from_template(
       "opening_range",
       tp_multiple=2.0,  # Wider target
       start_date="2025-03-01",
       end_date="2025-03-15"
   )
   
   result_v2 = SimulationRunner.run(
       strategy_config=strategy_v2,
       model_path="path/to/model.pt",
       start_date="2025-03-01",
       end_date="2025-03-15"
   )
   
   print(f"V2 - Win Rate: {result_v2.win_rate:.1%}, PnL: ${result_v2.total_pnl:.2f}")
   ```

4. Compare and decide:
   ```python
   if result_v2.total_pnl > result_v1.total_pnl:
       print("V2 is better! Use wider targets.")
       best_strategy = strategy_v2
   else:
       print("V1 is better. Keep original.")
       best_strategy = strategy_v1
   ```

5. Continue iterating on other parameters:
   - Stop loss size
   - Entry timing
   - Pattern filters
   - Risk per trade

Keep what works, discard what doesn't!
"""
    
    @staticmethod
    def recipe_multi_oco_testing() -> str:
        """Recipe: Test multiple OCO brackets simultaneously."""
        return """
RECIPE: Multi-OCO Testing
=========================

Goal: Test multiple risk/reward combinations on same trade trigger with limit entries.

Why: Compare different targets/stops to find optimal bracket configuration.

Steps:
------

1. Import tools:
   ```python
   from src.tools import StrategyBuilder, SimulationRunner
   from src.sim.multi_oco import MultiOCOConfig
   ```

2. Create multi-OCO grid (Option A - Pre-built):
   ```python
   # Test 3 targets: 1R, 1.5R, 2R on same trigger
   multi_oco = MultiOCOConfig.create_tight_medium_wide(
       direction="LONG",
       entry_offset=0.25  # Limit order 0.25 ATR from trigger
   )
   ```

3. Or create custom grid (Option B):
   ```python
   # Test different entry offsets AND targets
   multi_oco = MultiOCOConfig.create_standard_grid(
       direction="LONG",
       base_stop_atr=1.0,
       tp_multiples=[1.0, 1.5, 2.0, 2.5],  # 4 different targets
       entry_offsets=[0.1, 0.25, 0.5]      # 3 different entry levels
   )
   # This creates 4×3 = 12 OCO brackets!
   ```

4. Build strategy with multi-OCO:
   ```python
   strategy = StrategyBuilder.from_template(
       "opening_range",
       start_date="2025-03-01",
       end_date="2025-03-15"
   )
   ```

5. Run simulation:
   ```python
   result = SimulationRunner.run(
       strategy_config=strategy,
       model_path="path/to/model.pt",
       start_date="2025-03-01",
       end_date="2025-03-15",
       multi_oco_config=multi_oco  # Add multi-OCO config
   )
   ```

6. Analyze results:
   ```python
   print(result.summary())
   
   # Shows:
   # - Total trades per OCO config
   # - Fill rates (which entries get filled)
   # - Win rates per config
   # - Best performing OCO
   
   print(f"\\nBest OCO Configuration: {result.best_oco_config}")
   
   # Detailed analysis
   from src.sim.multi_oco import MultiOCOHelper
   
   if result.multi_oco_results:
       analysis = MultiOCOHelper.analyze_grid_performance(result.multi_oco_results)
       MultiOCOHelper.print_analysis(analysis)
   ```

Benefits:
---------
- Test multiple configurations in ONE backtest
- See which targets/stops work best
- Compare entry offsets (limit order distances)
- Find optimal risk/reward
- Understand fill rates for different entry levels

Example Output:
---------------
Multi-OCO Grid Result: tight_medium_wide
============================================================
Total OCOs: 3
Filled: 3 (100%)
Winners: 2
Losers: 1
Total PnL: $45.00
Best OCO: medium_1.5R

Individual OCO Results:
------------------------------------------------------------
  ✓ tight_1R               PnL: $10.00   Status: CLOSED_TP
  ✓ medium_1.5R            PnL: $25.00   Status: CLOSED_TP
  ✓ wide_2R                PnL: $10.00   Status: CLOSED_SL

Finding: Medium target (1.5R) is optimal!
"""
    
    @staticmethod
    def recipe_validate_before_deploy() -> str:
        """Recipe: Validate strategy before deploying."""
        return """
RECIPE: Pre-Deployment Validation
=================================

Goal: Ensure strategy is production-ready.

Checklist:
----------

1. Validate configuration:
   ```python
   from src.tools import StrategyValidator
   
   result = StrategyValidator.validate(strategy)
   result.print_report()
   
   if result.has_errors():
       print("❌ Fix errors before deploying!")
       exit(1)
   
   if result.has_warnings():
       print("⚠️  Review warnings carefully")
   ```

2. Test on multiple periods:
   ```python
   from src.tools import SimulationRunner
   
   periods = [
       ("2025-03-01", "2025-03-15"),  # Period 1
       ("2025-03-16", "2025-03-31"),  # Period 2
       ("2025-04-01", "2025-04-15"),  # Period 3
   ]
   
   results = []
   for start, end in periods:
       result = SimulationRunner.run(
           strategy_config=strategy,
           model_path="path/to/model.pt",
           start_date=start,
           end_date=end
       )
       results.append(result)
       print(f"{start} to {end}: PnL=${result.total_pnl:.2f}")
   
   # Check consistency
   avg_pnl = sum(r.total_pnl for r in results) / len(results)
   print(f"\\nAverage PnL: ${avg_pnl:.2f}")
   ```

3. Verify model is correct:
   ```python
   # Check model path
   import os
   assert os.path.exists("path/to/model.pt"), "Model file not found!"
   
   # Check model was used
   assert result.inference_count > 0, "Model didn't run inference!"
   print(f"✓ Model made {result.inference_count} inferences")
   ```

4. Review trade details:
   ```python
   # Check for anomalies
   for i, trade in enumerate(result.trades[:10], 1):
       print(f"Trade {i}: Entry={trade.get('entry_price')}, PnL=${trade.get('pnl'):.2f}")
   ```

5. Final check:
   ```python
   if all(r.win_rate >= 0.50 for r in results):
       print("✓ Strategy ready for deployment!")
   else:
       print("⚠️  Consider more refinement")
   ```

Better safe than sorry!
"""
    
    @staticmethod
    def list_all_recipes() -> List[str]:
        """List all available recipes."""
        return [
            "quick_start",
            "find_best_pattern",
            "parameter_sweep",
            "composite_trigger",
            "compare_models",
            "iterative_refinement",
            "multi_oco_testing",
            "validate_before_deploy",
        ]
    
    @staticmethod
    def get_recipe(recipe_name: str) -> str:
        """Get a specific recipe."""
        recipes = {
            "quick_start": AgentCookbook.recipe_quick_start,
            "find_best_pattern": AgentCookbook.recipe_find_best_pattern,
            "parameter_sweep": AgentCookbook.recipe_parameter_sweep,
            "composite_trigger": AgentCookbook.recipe_composite_trigger,
            "compare_models": AgentCookbook.recipe_compare_models,
            "iterative_refinement": AgentCookbook.recipe_iterative_refinement,
            "multi_oco_testing": AgentCookbook.recipe_multi_oco_testing,
            "validate_before_deploy": AgentCookbook.recipe_validate_before_deploy,
        }
        
        if recipe_name not in recipes:
            return f"Recipe '{recipe_name}' not found. Available: {list(recipes.keys())}"
        
        return recipes[recipe_name]()
    
    @staticmethod
    def print_menu():
        """Print cookbook menu."""
        print("\n" + "=" * 60)
        print("Agent Cookbook - Practical Recipes")
        print("=" * 60)
        print("\nAvailable Recipes:")
        print("  1. quick_start - Get started in 5 minutes")
        print("  2. find_best_pattern - Discover best entry pattern")
        print("  3. parameter_sweep - Find optimal parameters")
        print("  4. composite_trigger - Build complex conditions")
        print("  5. compare_models - Test multiple models")
        print("  6. iterative_refinement - Systematically improve")
        print("  7. multi_oco_testing - Test multiple OCO brackets with limits")
        print("  8. validate_before_deploy - Pre-deployment checklist")
        print("\nUsage:")
        print("  from src.tools.cookbook import AgentCookbook")
        print("  print(AgentCookbook.get_recipe('multi_oco_testing'))")
        print("=" * 60 + "\n")


# Convenience function
def show_recipe(recipe_name: str):
    """Show a recipe."""
    print(AgentCookbook.get_recipe(recipe_name))


if __name__ == "__main__":
    AgentCookbook.print_menu()

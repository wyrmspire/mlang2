"""
Explore Strategy (Safe Exploration Mode)

Run parameter sweeps WITHOUT generating TradeViz artifacts.
All output goes to results/exploration/ only.

Usage:
    python -m scripts.explore_strategy --recipe my_strat.json --grid '{"oco.tp_multiple": [2, 3]}' --out sweep_name
"""

import argparse
import json
import itertools
import copy
from pathlib import Path
from datetime import datetime, timedelta
import sys

from src.experiments.config import ExperimentConfig
from src.experiments.runner import run_experiment
from src.sim.oco_engine import OCOConfig
from src.features.pipeline import FeatureConfig
from src.policy.composite_scanner import CompositeScanner
from src.config import RESULTS_DIR, NY_TZ
import src.policy.scanners
import src.experiments.runner


EXPLORATION_DIR = RESULTS_DIR / "exploration"


def set_nested(obj: dict, path: str, value):
    """Set nested dict value using dot notation."""
    keys = path.split(".")
    for key in keys[:-1]:
        obj = obj.setdefault(key, {})
    obj[keys[-1]] = value


def get_nested(obj: dict, path: str, default=None):
    """Get nested dict value using dot notation."""
    keys = path.split(".")
    for key in keys:
        if isinstance(obj, dict):
            obj = obj.get(key, default)
        else:
            return default
    return obj


def generate_configs(base_recipe: dict, param_grid: dict) -> list[dict]:
    """Generate all combinations of recipe configs from a parameter grid."""
    if not param_grid:
        return [base_recipe]
    
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    configs = []
    for values in itertools.product(*param_values):
        recipe = copy.deepcopy(base_recipe)
        for name, value in zip(param_names, values):
            set_nested(recipe, name, value)
        configs.append(recipe)
    
    return configs


def run_single_config(recipe: dict, start_date: str, end_date: str) -> dict:
    """Run a single config and return metrics (no viz output)."""
    scanner = CompositeScanner(recipe)
    
    oco_config = recipe.get("oco", {})
    oco_settings = OCOConfig(
        tp_multiple=get_nested(oco_config, "take_profit.multiple", 2.0),
        stop_atr=get_nested(oco_config, "stop_loss.multiple", 1.0)
    )
    
    feature_settings = FeatureConfig(
        include_ohlcv=True,
        include_indicators=True,
        include_levels=False  # Lightweight
    )
    
    config = ExperimentConfig(
        name=f"explore_{scanner.scanner_id}",
        scanner_id=scanner.scanner_id,
        start_date=start_date,
        end_date=end_date,
        timeframe="1m",
        oco_config=oco_settings,
        feature_config=feature_settings,
        compute_cf=False  # No counterfactuals for speed
    )
    
    # Monkey-patch scanner factory
    original_factory = src.policy.scanners.get_scanner
    
    def mock_factory(scanner_id, **kwargs):
        if scanner_id == scanner.scanner_id:
            return scanner
        return original_factory(scanner_id, **kwargs)
    
    src.policy.scanners.get_scanner = mock_factory
    src.experiments.runner.get_scanner = mock_factory
    
    try:
        # Run with NO exporter (light mode)
        result = run_experiment(config, exporter=None)
        
        total_trades = getattr(result, 'total_trades', result.win_records + result.loss_records)
        wins = getattr(result, 'trade_wins', result.win_records)
        losses = getattr(result, 'trade_losses', result.loss_records)
        
        return {
            "recipe": recipe,
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / total_trades if total_trades > 0 else 0.0,
            "total_pnl": getattr(result, 'total_pnl', 0.0),
            "avg_pnl": getattr(result, 'avg_pnl', 0.0)
        }
    finally:
        src.policy.scanners.get_scanner = original_factory
        src.experiments.runner.get_scanner = original_factory


def main():
    parser = argparse.ArgumentParser(description="Run exploration sweep (no TradeViz output)")
    parser.add_argument("--recipe", required=True, help="Path to base JSON recipe")
    parser.add_argument("--grid", required=True, help="JSON param grid, e.g. '{\"oco.tp_multiple\": [2, 3]}'")
    parser.add_argument("--out", required=True, help="Output name (in results/exploration/)")
    parser.add_argument("--start-date", default="2025-04-01", help="YYYY-MM-DD")
    parser.add_argument("--end-date", default="2025-04-30", help="YYYY-MM-DD")
    
    args = parser.parse_args()
    
    # Load recipe
    recipe_path = Path(args.recipe)
    if not recipe_path.exists():
        print(f"Error: Recipe not found: {args.recipe}")
        sys.exit(1)
    
    with open(recipe_path) as f:
        base_recipe = json.load(f)
    
    # Parse grid
    try:
        param_grid = json.loads(args.grid)
    except json.JSONDecodeError as e:
        print(f"Error parsing grid JSON: {e}")
        sys.exit(1)
    
    # Generate configs
    configs = generate_configs(base_recipe, param_grid)
    print(f"[EXPLORE] Running {len(configs)} configurations...")
    
    # Run all
    results = []
    for i, recipe in enumerate(configs):
        print(f"[EXPLORE] Config {i+1}/{len(configs)}")
        try:
            metrics = run_single_config(recipe, args.start_date, args.end_date)
            results.append(metrics)
        except Exception as e:
            print(f"  Error: {e}")
            results.append({"recipe": recipe, "error": str(e)})
    
    # Rank by win_rate, then total_pnl
    valid_results = [r for r in results if "error" not in r]
    valid_results.sort(key=lambda r: (r["win_rate"], r["total_pnl"]), reverse=True)
    
    # Output
    EXPLORATION_DIR.mkdir(parents=True, exist_ok=True)
    output_path = EXPLORATION_DIR / f"{args.out}.json"
    
    summary = {
        "exploration_id": args.out,
        "timestamp": datetime.now().isoformat(),
        "base_recipe": str(recipe_path),
        "param_grid": param_grid,
        "total_configs": len(configs),
        "successful_configs": len(valid_results),
        "best_config": valid_results[0] if valid_results else None,
        "all_results": results
    }
    
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n[EXPLORE] Done! Results saved to: {output_path}")
    if valid_results:
        best = valid_results[0]
        print(f"[EXPLORE] Best config: WinRate={best['win_rate']:.1%}, PnL=${best['total_pnl']:.2f}")


if __name__ == "__main__":
    main()

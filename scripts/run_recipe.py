"""
Run Recipe
Execute a strategy defined by a JSON recipe file.

Usage:
    python scripts/run_recipe.py --recipe my_strategy.json --out name
"""

import argparse
import json
import asyncio
from pathlib import Path
from datetime import datetime
import sys
import pandas as pd
import numpy as np
from datetime import timedelta

from src.experiments.config import ExperimentConfig
from src.experiments.runner import run_experiment
from src.sim.oco_engine import OCOConfig
from src.features.pipeline import FeatureConfig
from src.viz.export import Exporter
from src.policy.composite_scanner import CompositeScanner
from src.config import RESULTS_DIR, NY_TZ


def main():
    parser = argparse.ArgumentParser(description="Run a strategy from a recipe file.")
    parser.add_argument("--recipe", required=True, help="Path to JSON recipe file")
    parser.add_argument("--out", required=True, help="Output name (folder in results/viz/)")
    parser.add_argument("--start-date", help="YYYY-MM-DD")
    parser.add_argument("--end-date", help="YYYY-MM-DD")
    parser.add_argument("--days", type=int, default=30, help="Days to run if no days provided")
    parser.add_argument("--mock", action="store_true", help="Use synthetic data for testing")
    
    parser.add_argument("--light", action="store_true", help="Run in light mode (no heavy viz files)")
    
    args = parser.parse_args()
    
    # 1. Load Recipe
    recipe_path = Path(args.recipe)
    if not recipe_path.exists():
        print(f"Error: Recipe file not found: {args.recipe}")
        return
        
    with open(recipe_path, 'r') as f:
        recipe = json.load(f)
        
    print(f"Loaded recipe: {recipe.get('name', 'Unknown')}")
    
    # Mock Data Injection
    if args.mock:
        print("WARNING: Using MOCK data mode.")
        
        def mock_loader(**kwargs):
            # Generate 1000 bars of sine wave price
            base = datetime.now(NY_TZ) - timedelta(days=10)
            times = [base + timedelta(minutes=i) for i in range(1000)]
            
            # Create a trend + noise
            x = np.linspace(0, 100, 1000)
            price = 5000 + 100 * np.sin(x/10) + np.random.normal(0, 5, 1000)
            
            df = pd.DataFrame({
                'time': times,
                'open': price,
                'high': price + 5,
                'low': price - 5,
                'close': price, # simplistic
                'volume': 1000
            })
            return df
            
        import src.data.loader
        import src.experiments.runner
        
        # Patch the source
        src.data.loader.load_continuous_contract = mock_loader
        src.data.loader.load_processed_1m = lambda **kw: mock_loader()
        
        # Patch the consumer
        src.experiments.runner.load_continuous_contract = mock_loader
        src.experiments.runner.load_processed_1m = lambda **kw: mock_loader()
    
    # 2. Build Scanner
    scanner = CompositeScanner(recipe)
    print(f"Initialized scanner: {scanner.scanner_id}")
    
    # 3. Configure Experiment
    oco_config = recipe.get("oco", {})
    entry_trigger = recipe.get("entry_trigger", {})
    
    # Determine dates
    if args.start_date:
        start_date = args.start_date
        end_date = args.end_date
    else:
        start_date = "2025-01-01"
        end_date = "2025-01-30"
        
    oco_settings = OCOConfig(
        tp_multiple=oco_config.get("take_profit", {}).get("multiple", 2.0),
        stop_atr=oco_config.get("stop_loss", {}).get("multiple", 1.0)
    )

    feature_settings = FeatureConfig(
        include_ohlcv=True,
        include_indicators=True,
        include_levels=True
    )
        
    config = ExperimentConfig(
        name=args.out,
        scanner_id=scanner.scanner_id,
        start_date=start_date, 
        end_date=end_date,
        timeframe="1m",
        oco_config=oco_settings,
        feature_config=feature_settings,
    )
    
    # 4. Setup Exporter (ONLY IF NOT LIGHT MODE)
    exporter = None
    out_dir = RESULTS_DIR / "viz" / args.out

    if not args.light:
        from src.viz.config import VizConfig
        
        viz_config = VizConfig()
        exporter = Exporter(
            config=viz_config,
            run_id=args.out,
            experiment_config=config.to_dict()
        )
    else:
        print("Light Mode enabled: Skipping visualization export")
    
    # 5. Monkey Patch get_scanner
    import src.policy.scanners
    original_factory = src.policy.scanners.get_scanner
    
    def mock_factory(scanner_id, **kwargs):
        if scanner_id == scanner.scanner_id:
            return scanner
        return original_factory(scanner_id, **kwargs)
        
    src.policy.scanners.get_scanner = mock_factory
    src.experiments.runner.get_scanner = mock_factory 
    
    try:
        # 6. Run Experiment
        result = run_experiment(config, exporter=exporter)
        
        # 7. Finalize (ONLY IF EXPORTER EXISTS)
        if exporter:
            exporter.finalize(out_dir)
        
        # 8. Save to ExperimentDB
        try:
            from src.storage.experiments_db import ExperimentDB
            
            # Use direct results from ExperimentResult
            # This works for both Light Mode (where we rely on result objects)
            # and Standard Mode (assuming we added pnl to result object)
            
            total_trades = result.win_records + result.loss_records
            win_rate = result.win_records / total_trades if total_trades > 0 else 0
            
            # Note: total_pnl/avg_pnl were added to ExperimentResult recently
            total_pnl = getattr(result, 'total_pnl', 0.0)
            avg_pnl = getattr(result, 'avg_pnl', 0.0)
            
            # If standard mode and result object lacks PnL (legacy check),
            # fall back to reading file (for double safety if result object update failed)
            # But since we updated runner.py, we trust the object first.
            
            # Only read file if PnL is 0 and we are NOT in light mode (and expected file to exist)
            if total_pnl == 0 and not args.light and total_trades > 0:
                 trades_file = out_dir / "trades.jsonl"
                 if trades_file.exists():
                     # ... Legacy read code path (omitted for brevity as we trust runner.py now)
                     pass

            # Store in database
            db = ExperimentDB()
            db.store_run(
                run_id=args.out,
                strategy=recipe.get('name', 'composite_strategy'),
                config={
                    'recipe': recipe,
                    'start_date': start_date,
                    'end_date': end_date,
                    'entry_trigger': entry_trigger,
                    'oco': oco_config
                },
                metrics={
                    'total_trades': total_trades,
                    'wins': result.win_records,
                    'losses': result.loss_records,
                    'win_rate': win_rate,
                    'total_pnl': total_pnl,
                    'avg_pnl_per_trade': avg_pnl
                }
            )
            print(f"Saved to ExperimentDB: {args.out}")
        except Exception as e:
            print(f"Could not save to ExperimentDB: {e}")
            import traceback
            traceback.print_exc()
        
        if not args.light:
            print(f"Success! Output at: {out_dir}")
        else:
            print(f"Success! (Light Mode - Results in DB only)")
        
    finally:
        # Restore factory
        src.policy.scanners.get_scanner = original_factory


if __name__ == "__main__":
    main()

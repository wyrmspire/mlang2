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

from src.experiments.config import ExperimentConfig
from src.experiments.runner import run_experiment
from src.sim.oco_engine import OCOConfig
from src.features.pipeline import FeatureConfig
from src.viz.export import Exporter
from src.policy.composite_scanner import CompositeScanner
from src.config import RESULTS_DIR


def main():
    parser = argparse.ArgumentParser(description="Run a strategy from a recipe file.")
    parser.add_argument("--recipe", required=True, help="Path to JSON recipe file")
    parser.add_argument("--out", required=True, help="Output name (folder in results/viz/)")
    parser.add_argument("--start-date", help="YYYY-MM-DD")
    parser.add_argument("--end-date", help="YYYY-MM-DD")
    parser.add_argument("--days", type=int, default=30, help="Days to run if no days provided")
    parser.add_argument("--mock", action="store_true", help="Use synthetic data for testing")
    
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
        import pandas as pd
        import numpy as np
        from datetime import timedelta
        from src.config import NY_TZ
        
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
            
            return df
            
        import src.data.loader
        import src.experiments.runner
        
        # Patch the source (good practice)
        src.data.loader.load_continuous_contract = mock_loader
        src.data.loader.load_processed_1m = lambda **kw: mock_loader()
        
        # Patch the consumer (CRITICAL because of 'from X import Y')
        src.experiments.runner.load_continuous_contract = mock_loader
        src.experiments.runner.load_processed_1m = lambda **kw: mock_loader()

    
    # 2. Build Scanner
    scanner = CompositeScanner(recipe)
    print(f"Initialized scanner: {scanner.scanner_id}")
    
    # 3. Configure Experiment
    # Use OCO settings from recipe if available, else defaults
    oco_config = recipe.get("oco", {})
    entry_trigger = recipe.get("entry_trigger", {})
    
    # Determine defaults based on recipe or args
    # Note: ExperimentConfig usually takes start/end dates
    
    from src.config import CONTINUOUS_CONTRACT_PATH
    
    # Determine dates
    if args.start_date:
        start_date = args.start_date
        end_date = args.end_date
    else:
        # Default to last N days
        # (Simplified date logic for now)
        start_date = "2025-01-01" # Placeholder, clearer logic in real app
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
    
    # 4. Setup Exporter
    from src.viz.config import VizConfig
    out_dir = RESULTS_DIR / "viz" / args.out
    
    viz_config = VizConfig()
    exporter = Exporter(
        config=viz_config,
        run_id=args.out,
        experiment_config=config.to_dict()
    )
    
    # 5. Monkey Patch get_scanner
    import src.policy.scanners
    original_factory = src.policy.scanners.get_scanner
    
    def mock_factory(scanner_id, **kwargs):
        if scanner_id == scanner.scanner_id:
            return scanner
        return original_factory(scanner_id, **kwargs)
        
    src.policy.scanners.get_scanner = mock_factory
    src.experiments.runner.get_scanner = mock_factory # Patch consumer!
    
    try:
        # 6. Run Experiment
        result = run_experiment(config, exporter=exporter)
        
        # 7. Finalize
        exporter.finalize(out_dir)
        
        # 8. Save to ExperimentDB
        try:
            from src.storage import ExperimentDB
            
            # Load trades to calculate metrics
            trades_file = out_dir / "trades.jsonl"
            total_pnl = 0.0
            wins = 0
            losses = 0
            total_trades = 0
            
            if trades_file.exists():
                with open(trades_file) as f:
                    for line in f:
                        if line.strip():
                            t = json.loads(line)
                            total_trades += 1
                            pnl = t.get('pnl_dollars', 0)
                            total_pnl += pnl
                            if pnl > 0:
                                wins += 1
                            else:
                                losses += 1
            
            win_rate = wins / total_trades if total_trades > 0 else 0
            avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
            
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
                    'wins': wins,
                    'losses': losses,
                    'win_rate': win_rate,
                    'total_pnl': total_pnl,
                    'avg_pnl_per_trade': avg_pnl
                }
            )
            print(f"✅ Saved to ExperimentDB: {args.out}")
        except Exception as e:
            print(f"⚠️  Could not save to ExperimentDB: {e}")
        
        print(f"Success! Output at: {out_dir}")
        print("Don't forget to validate: python golden/validate_run.py " + str(out_dir))
        
    finally:
        # Restore factory
        src.policy.scanners.get_scanner = original_factory


if __name__ == "__main__":
    main()

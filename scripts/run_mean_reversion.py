"""
Run Mean Reversion Experiment
Runs a 3-week scan using Mean Reversion strategy.
"""

import sys
import os
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.append(os.getcwd())

from src.experiments.config import ExperimentConfig, FeatureConfig, LabelConfig
from src.experiments.runner import run_experiment
from src.viz.export import Exporter
from src.viz.config import VizConfig

def main():
    # Define time range (approx 3 weeks)
    # Using a recent slice of data, assuming data exists
    # If not sure, we can load data first, but let's try a fixed range or last 3 weeks
    end_date = pd.Timestamp("2025-06-21", tz="America/New_York") # Adjust based on available data
    start_date = end_date - pd.Timedelta(weeks=3)
    
    print(f"Running Mean Reversion Scan from {start_date.date()} to {end_date.date()}")
    
    # Configure Experiment
    config = ExperimentConfig(
        name="mean_reversion_3w",
        start_date=start_date,
        end_date=end_date,
        
        # Scanner: Mean Reversion
        scanner_id="mean_reversion_20_3.0_5m_30_70",
        scanner_params={
            "ema_period": 20,
            "atr_multiple": 3.0,
            "rsi_min": 30.0,
            "rsi_max": 70.0,
            "timeframe": "5m"
        },
        
        # Standard configs
        feature_config=FeatureConfig(),
        label_config=LabelConfig(), # Defaults
    )
    
    # Configure Exporter
    # We want to export windows for visualization
    viz_config = VizConfig(
        include_windows=True,
        include_model_outputs=False
    )
    
    # Run
    # Using a temporary output directory for this run
    out_dir = Path("results/mean_reversion_3w")
    exporter = Exporter(viz_config, experiment_config=config.to_dict())
    
    result = run_experiment(config, exporter=exporter)
    
    # Save results
    exporter.finalize(out_dir)
    
    print("\n" + "="*50)
    print(f"Experiment Complete!")
    print(f"Total Decisions: {result.total_records}")
    print(f"Results saved to: {out_dir.absolute()}")
    print("="*50)

if __name__ == "__main__":
    main()

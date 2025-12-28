# MLang2 Project Code Dump
Generated: Sun, Dec 28, 2025  4:29:23 AM

## Project Structure
```
.agent/workflows/lab-agent-prompts.md
.github/workflows/contract-lint.yml
.github/workflows/golden-validation.yml
.github/workflows/test.yml
.gitignore
.pytest_cache/.gitignore
.pytest_cache/CACHEDIR.TAG
.pytest_cache/README.md
_backup/ingest_scan_records.py
_backup/run_replay.py
_backup/SimulationView.tsx.bak
_backup/train_from_shards.py
agents.md
docs/CAUSAL_PRINCIPLES.md
docs/IMPLEMENTATION_SUMMARY.md
docs/QUICK_START.md
docs/REPLAY_MODE.md
docs/SIMULATION_MODE.md
docs/success_study.md
docs/YFINANCE_MODE.md
gitr.sh
gitrdif.sh
gitrdiff.md
golden/.gitkeep
index.html
index.tsx
jules.md
metadata.json
models/ifvg_4class_cnn.json
models/puller_xgb_4class.json
package.json
printcode.sh
README.md
requirements.txt
scripts/agent_chat.py
scripts/backtest_combined_strategy.py
scripts/backtest_delayed_breakout.py
scripts/backtest_ema.py
scripts/backtest_ict_fvg.py
scripts/backtest_ict_ifvg.py
scripts/backtest_inverse_test.py
scripts/backtest_lowvol_breakout.py
scripts/backtest_lunch_fade.py
scripts/backtest_mean_reversion.py
scripts/backtest_modular_strategy.py
scripts/backtest_or_multi_oco.py
scripts/backtest_puller.py
scripts/backtest_rvap.py
scripts/backtest_simple_time.py
scripts/backtest_structure_break.py
scripts/backtest_swing_breakout.py
scripts/backtest_walkforward_daily.py
scripts/backtest_walkforward_viz.py
scripts/create_strategy.py
scripts/debug_ifvg.py
scripts/demo_phase_5_6.py
scripts/explore_strategy.py
scripts/ingest_scan_records.py
scripts/inverse_strategy.py
scripts/optimize_orb_gridsearch.py
scripts/run_recipe.py
scripts/scan_ema_rejection.py
scripts/scan_ema200_rejection.py
scripts/scan_fakeout_fade.py
scripts/scan_or_false_break.py
scripts/scan_pdh_sweep.py
scripts/scan_power_hour.py
scripts/scan_puller_variations.py
scripts/scan_test_simple.py
scripts/session_ifvg_replay.py
scripts/session_ifvg_simulation.py
scripts/session_live.py
scripts/session_replay.py
scripts/smart_cnn.py
scripts/stress_test_tools.py
scripts/sweep/__init__.py
scripts/sweep/config.py
scripts/sweep/oco_tester.py
scripts/sweep/param_grid.py
scripts/sweep/pattern_miner_v2.py
scripts/sweep/run_sweep_integrated.py
scripts/sweep/supersweep.py
scripts/sweep/train_sweep.py
scripts/train_from_shards.py
scripts/train_fusion_mtf.py
scripts/train_ifvg_4class.py
scripts/train_ifvg_cnn.py
scripts/train_lstm_compare.py
scripts/verify_fixes.py
scripts/verify_position_boxes.py
scripts/verify_replay_inference.py
src/.env.local
src/.env.local.example
src/__init__.py
src/api/client.ts
src/App.tsx
src/components/CandleChart.tsx
src/components/ChatAgent.tsx
src/components/DetailsPanel.tsx
src/components/ExperimentsView.tsx
src/components/IndicatorSettings.tsx
src/components/LabPage.tsx
src/components/LiveSessionView.tsx
src/components/LiveSessionView.tsx.backup
src/components/Navigator.tsx
src/components/PositionBox.ts
src/components/ReplayControls.tsx
src/components/RunPicker.tsx
src/components/SimulationChart.tsx
src/components/StatsPanel.tsx
src/components/TimeframeBar.tsx
src/config.py
src/core/__init__.py
src/core/enums.py
src/core/manifest.py
src/core/registries.py
src/core/strategy_tool.py
src/core/tool_registry.py
src/datasets/__init__.py
src/datasets/decision_record.py
src/datasets/reader.py
src/datasets/schema.py
src/datasets/trade_record.py
src/datasets/writer.py
src/eval/__init__.py
src/eval/breakdown.py
src/eval/mae_mfe.py
src/eval/metrics.py
src/experiments/__init__.py
src/experiments/config.py
src/experiments/fast_forward.py
src/experiments/fingerprint.py
src/experiments/report.py
src/experiments/runner.py
src/experiments/splits.py
src/experiments/strategy_config.py
src/experiments/sweep.py
src/features/__init__.py
src/features/chart_indicators.ts
src/features/context.py
src/features/engine.py
src/features/fvg.py
src/features/index.ts
src/features/indicator_registry_init.py
src/features/indicators.py
src/features/indicators_pro.py
src/features/levels.py
src/features/patterns.py
src/features/pipeline.py
src/features/session_levels.py
src/features/state.py
src/features/swings.py
src/features/time_features.py
src/hooks/useIndicators.ts
src/index.css
src/labels/__init__.py
src/labels/counterfactual.py
src/labels/future_window.py
src/labels/labeler.py
src/labels/trade_outcome.py
src/models/__init__.py
src/models/context_mlp.py
src/models/encoders.py
src/models/fusion.py
src/models/heads.py
src/models/model_registry_init.py
src/models/train.py
src/policy/__init__.py
src/policy/actions.py
src/policy/brackets.py
src/policy/composite_scanner.py
src/policy/cooldown.py
src/policy/entry_scans.py
src/policy/filters.py
src/policy/library/__init__.py
src/policy/library/delayed_breakout.py
src/policy/library/first_pullback.py
src/policy/library/ict_fvg.py
src/policy/library/ict_ifvg.py
src/policy/library/mean_reversion.py
src/policy/library/mid_day_reversal.py
src/policy/library/momentum_divergence.py
src/policy/library/new_test_ema_cross.py
src/policy/library/new_test_gap_fill.py
src/policy/library/new_test_rsi_bounce.py
src/policy/library/new_test_three_bar_play.py
src/policy/library/new_test_vol_breakout.py
src/policy/library/opening_range.py
src/policy/library/puller.py
src/policy/library/session_break.py
src/policy/library/simple_time.py
src/policy/library/structure_break.py
src/policy/library/swing_breakout.py
src/policy/library/volume_spike.py
src/policy/library/vwap_bounce.py
src/policy/modular_scanner.py
src/policy/oco_grid.py
src/policy/scanner_registry_init.py
src/policy/scanners.py
src/policy/triggers/__init__.py
src/policy/triggers/base.py
src/policy/triggers/candle_patterns.py
src/policy/triggers/ema_rejection.py
src/policy/triggers/factory.py
src/policy/triggers/fakeout.py
src/policy/triggers/indicator_triggers.py
src/policy/triggers/logic.py
src/policy/triggers/or_false_break.py
src/policy/triggers/parametric.py
src/policy/triggers/price_action_triggers.py
src/policy/triggers/structure_break.py
src/policy/triggers/sweep.py
src/policy/triggers/time_trigger.py
src/policy/triggers/vwap_reclaim.py
src/server/__init__.py
src/server/db_routes.py
src/server/infer_routes.py
src/server/main.py
src/server/replay_routes.py
src/sim/__init__.py
src/sim/account.py
src/sim/account_manager.py
src/sim/bar_fill_model.py
src/sim/causal_runner.py
src/sim/costs.py
src/sim/entry_strategies.py
src/sim/execution.py
src/sim/market_session.py
src/sim/oco.py
src/sim/oco_engine.py
src/sim/sizing.py
src/sim/stepper.py
src/sim/stop_calculator.py
src/sim/validation.py
src/sim/yfinance_stepper.py
src/skills/data_skills.py
src/skills/indicator_skills.py
src/skills/pattern_skills.py
src/storage/__init__.py
src/storage/experiments_db.py
src/strategy/__init__.py
src/strategy/scan.py
src/strategy/spec.py
src/tools/__init__.py
src/tools/agent_tools.py
src/tools/analysis_tools.py
src/tools/contract_linter.py
src/tools/exploration_tools.py
src/tools/price_analysis_tools.py
src/types/viz.ts
src/viz/__init__.py
src/viz/config.py
src/viz/export.py
src/viz/schema.py
src/viz/window_utils.py
start.sh
stress_test_1.json
stress_test_2.json
test_box.json
tests/test_agent_integration.py
tests/test_backend_simulation.py
tests/test_causal_runner.py
tests/test_golden_runs.py
tests/test_indicators_pro.py
tests/test_oco_engine.py
tests/test_oco_properties.py
tests/test_replay_connection.py
tests/test_sizing.py
tests/test_strategy_spec.py
tests/test_timezone_and_scanner.py
tests/test_tool_registry.py
tests/test_triggers_brackets.py
tests/test_viz_export.py
tests/test_window_utils.py
trade_logic_analysis.md
tsconfig.json
verification/error.png
verification/frontend_viz.png
verification/verify_script.py
verify_log.txt
vite.config.ts
```

## Source Files

### .agent/workflows/lab-agent-prompts.md

```markdown
# Lab Agent Roleplay Prompts

These are the prompts for demonstrating Lab Agent capabilities.
Say "prompt X" to run that scenario.

---

## Prompt 1 (Complete)
"Show me the cleanest long opportunities from last week where price moved fast and never really looked back. I don't care about frequency, just quality."

## Prompt 2
"Find trades that look obvious in hindsight — the kind where you'd say 'how did I miss that?' and tell me what they had in common before they happened."

## Prompt 3
"Run a scan that only fires a few times a week but has strong follow-through. I want fewer signals, not more."

## Prompt 4
"Compare trades that worked in the morning vs trades that worked later in the day and explain the difference in price behavior."

## Prompt 5
"Take one solid setup from the past month and show me three different ways it could have been traded better."

## Prompt 6
"I want to trade momentum, not chop. Find patterns that clearly did not work when price went sideways and tell me how to avoid them."

## Prompt 7
"Build a scan that catches moves right as they start, not after they're obvious. Then show me where it still fails."

## Prompt 8
"Look at the biggest winners in the data and tell me what price usually did in the 30 minutes before entry."

## Prompt 9
"Find setups that work great sometimes and terribly other times. What's different when they fail?"

## Prompt 10
"Create two versions of the same idea: one aggressive and one conservative. Show me the tradeoffs."

## Prompt 11
"If I only traded during the best part of the day, what would that be and why?"

## Prompt 12
"Show me trades where holding longer would have paid off, and trades where holding longer would have made things worse."

## Prompt 13
"Find a setup that looks great on paper but is probably a trap. Explain why."

## Prompt 14
"I want something simple: one or two conditions, no fancy logic. See if simple actually works better."

## Prompt 15
"Take a strategy that performs okay and try to remove rules until it breaks. Tell me which rule actually matters."

## Prompt 16
"Look for patterns that repeat across different weeks, not just one lucky stretch."

## Prompt 17
"If I were only allowed to take 1–2 trades a day, what kind of setups should I be waiting for?"

## Prompt 18
"Show me a strategy that loses money overall but has one specific situation where it's very strong."

## Prompt 19
"Run something experimental that you wouldn't normally recommend, just to see what happens. Then explain the risk."

## Prompt 20
"Take your best idea so far and put it on the chart so I can visually inspect whether it actually makes sense."

```

### .github/workflows/contract-lint.yml

```yaml
name: Contract Linter

on:
  push:
    branches: [ main, develop, copilot/** ]
    paths:
      - 'results/**'
      - 'src/tools/contract_linter.py'
      - 'src/viz/**'
  pull_request:
    branches: [ main, develop ]

jobs:
  lint-contracts:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.12'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Lint run artifacts
      run: |
        echo "🔍 Linting run artifacts for contract compliance..."
        if [ -d "results/" ]; then
          FAILED=0
          for run_dir in results/*/; do
            if [ -d "$run_dir" ] && [ -f "$run_dir/manifest.json" ]; then
              echo "Linting: $run_dir"
              if ! python -m src.tools.contract_linter "$run_dir"; then
                FAILED=1
              fi
            fi
          done
          
          if [ $FAILED -eq 1 ]; then
            echo "❌ Contract linter found violations"
            exit 1
          else
            echo "✅ All run artifacts comply with contract"
          fi
        else
          echo "⚠️  No results directory found, skipping"
        fi
    
    - name: Verify linter is working
      run: |
        # Test that linter catches violations
        echo "Testing contract linter functionality..."
        python -c "from src.tools.contract_linter import ContractLinter; print('✅ ContractLinter available')"

```

### .github/workflows/golden-validation.yml

```yaml
name: Golden File Validation

on:
  push:
    branches: [ main, develop, copilot/** ]
    paths:
      - 'golden/**'
      - 'src/viz/**'
      - 'src/sim/**'
      - 'src/strategy/**'
  pull_request:
    branches: [ main, develop ]

jobs:
  validate-golden:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.12'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Validate golden run artifacts
      run: |
        echo "🔍 Validating golden run artifacts..."
        if [ -d "golden/" ]; then
          for run_dir in golden/*/; do
            if [ -d "$run_dir" ]; then
              echo "Validating: $run_dir"
              python golden/validator.py "$run_dir" || exit 1
            fi
          done
          echo "✅ All golden runs validated successfully"
        else
          echo "⚠️  No golden directory found, skipping"
        fi
    
    - name: Run golden file tests
      run: |
        python -m pytest tests/test_golden_runs.py -v --tb=short

```

### .github/workflows/test.yml

```yaml
name: Tests

on:
  push:
    branches: [ main, develop, copilot/** ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.12'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest
    
    - name: Run core tests
      run: |
        python -m pytest tests/test_tool_registry.py -v
        python -m pytest tests/test_golden_runs.py -v
        python -m pytest tests/test_strategy_spec.py -v
    
    - name: Run OCO engine tests
      run: |
        python -m pytest tests/test_oco_engine.py -v
        python -m pytest tests/test_oco_properties.py -v
    
    - name: Test summary
      if: always()
      run: |
        echo "✅ Core architecture tests complete"
        echo "✅ OCO engine property tests complete"

```

### .pytest_cache/README.md

```markdown
# pytest cache directory #

This directory contains data from the pytest's cache plugin,
which provides the `--lf` and `--ff` options, as well as the `cache` fixture.

**Do not** commit this to version control.

See [the docs](https://docs.pytest.org/en/stable/how-to/cache.html) for more information.

```

### _backup/ingest_scan_records.py

```python
"""
Ingest Scan Records
Converts JSONL records from a Scan into a Sharded Dataset for Training.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.datasets.decision_record import DecisionRecord
from src.datasets.writer import ShardWriter


def parse_record(data: dict) -> DecisionRecord:
    """
    Parse a dictionary (from JSONL) into a DecisionRecord object.
    Reconstructs numpy arrays from lists.
    """
    # Extract window data
    window = data.get('window', {})
    
    x_price_1m = np.array(window.get('x_price_1m', []))
    x_context = np.array(window.get('x_context', []))
    
    # Extract outcomes
    oco_res = data.get('oco_results', {}).get('swing_breakout', {})
    outcome = oco_res.get('outcome', 'TIMEOUT')
    pnl = oco_res.get('pnl_dollars', 0.0)
    
    # Parse timestamp
    ts_str = data.get('timestamp')
    timestamp = pd.Timestamp(ts_str) if ts_str else None
    
    return DecisionRecord(
        timestamp=timestamp,
        bar_idx=data.get('bar_idx', 0),
        decision_id=data.get('decision_id'),
        scanner_id=data.get('scanner_id'),
        
        # Features
        x_price_1m=x_price_1m,
        x_price_5m=None, # Not explicitly in simple scan output yet
        x_price_15m=None,
        x_context=x_context,
        
        # Labels
        cf_outcome=outcome,
        cf_pnl=pnl,
        cf_mae=0.0, # Populated if available
        cf_mfe=0.0,
        cf_bars_held=0,
        
        # Metadata
        current_price=data.get('current_price', 0.0),
        atr=data.get('atr', 0.0)
    )

def main():
    parser = argparse.ArgumentParser(description="Ingest Scan Records to Shards")
    parser.add_argument("--input", type=str, required=True, help="Path to records.jsonl")
    parser.add_argument("--out", type=str, required=True, help="Output shard directory")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of records")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_dir = Path(args.out)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Ingesting from: {input_path}")
    print(f"Writing to:     {output_dir}")
    
    count = 0
    with ShardWriter(output_dir, records_per_shard=1000) as writer:
        with open(input_path, 'r') as f:
            for line in tqdm(f):
                if not line.strip():
                    continue
                    
                try:
                    data = json.loads(line)
                    record = parse_record(data)
                    writer.write(record)
                    count += 1
                    
                    if args.limit > 0 and count >= args.limit:
                        break
                        
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line")
                except Exception as e:
                    print(f"Error parsing record: {e}")
                    # raise e # Uncomment to debug
                    
    print(f"Done. Ingested {count} records.")

if __name__ == "__main__":
    main()

```

### _backup/run_replay.py

```python
"""
Replay Mode Runner

Run trained CNN model on historical data bar-by-bar, emitting events.
Usage: python scripts/run_replay.py --model models/best_model.pth --start-date 2025-03-17 --days 1
"""

import argparse
import json
import sys
import time
import torch
import pandas as pd
from pathlib import Path
from datetime import timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RESULTS_DIR, NY_TZ
from src.data.loader import load_continuous_contract
from src.data.resample import resample_all_timeframes
from src.features.indicators import calculate_atr
from src.features.pipeline import compute_features, FeatureConfig
from src.sim.stepper import MarketStepper
from src.models.fusion import FusionModel, SimpleCNN
from src.core.enums import RunMode


def normalize_window(x, method='zscore'):
    """Simple z-score normalization for price windows."""
    import numpy as np
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0) + 1e-8
    return (x - mean) / std



def load_model(model_path: Path):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Determine model type from checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Check if FusionModel or SimpleCNN based on keys
    if any('price_encoder' in k for k in state_dict.keys()):
        model = FusionModel()
    else:
        model = SimpleCNN()
    
    model.load_state_dict(state_dict)
    model.eval()
    return model


def emit_event(event_type: str, data: dict):
    """Emit event as JSON line to stdout."""
    event = {
        'type': event_type,
        **data
    }
    print(json.dumps(event), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Run CNN Model Replay")
    parser.add_argument("--model", type=str, default="models/best_model.pth",
                        help="Path to trained model")
    parser.add_argument("--start-date", type=str, default="2025-03-17",
                        help="Start date for replay")
    parser.add_argument("--days", type=int, default=1,
                        help="Number of days to replay")
    parser.add_argument("--speed", type=float, default=10.0,
                        help="Speed multiplier (1.0 = real-time, 10.0 = 10x)")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="Confidence threshold for triggering")
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory for decisions")
    
    args = parser.parse_args()
    
    model_path = Path(args.model)
    if not model_path.exists():
        emit_event('ERROR', {'message': f'Model not found: {model_path}'})
        sys.exit(1)
    
    # Load model
    emit_event('STATUS', {'message': 'Loading model...'})
    model = load_model(model_path)
    
    # Load data
    emit_event('STATUS', {'message': 'Loading market data...'})
    df = load_continuous_contract()
    
    start_date = pd.Timestamp(args.start_date, tz=NY_TZ)
    end_date = start_date + timedelta(days=args.days)
    
    df = df[(df['time'] >= start_date) & (df['time'] < end_date)].reset_index(drop=True)
    if len(df) < 200:
        emit_event('ERROR', {'message': f'Not enough data: {len(df)} bars'})
        sys.exit(1)
    
    emit_event('STATUS', {'message': f'Loaded {len(df)} bars'})
    
    # Resample for higher timeframes
    htf_data = resample_all_timeframes(df)
    df_5m = htf_data.get('5m')
    df_15m = htf_data.get('15m')
    df_5m['atr'] = calculate_atr(df_5m, 14)
    
    # Initialize stepper
    stepper = MarketStepper(df, start_idx=120, end_idx=len(df) - 30)
    feature_config = FeatureConfig(lookback_1m=120)
    
    # Emit replay start
    emit_event('REPLAY_START', {
        'start_date': start_date.isoformat(),
        'end_date': end_date.isoformat(),
        'total_bars': len(df),
        'model': str(model_path)
    })
    
    # Delay between bars based on speed
    bar_delay = 1.0 / args.speed  # seconds per bar
    
    decision_count = 0
    trigger_count = 0
    decisions = []
    
    while True:
        step = stepper.step()
        if step.is_done:
            break
        
        bar_idx = step.bar_idx
        current_bar = step.bar
        timestamp = current_bar['time']
        
        # Emit bar update
        emit_event('BAR', {
            'bar_idx': bar_idx,
            'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
            'open': float(current_bar['open']),
            'high': float(current_bar['high']),
            'low': float(current_bar['low']),
            'close': float(current_bar['close']),
            'volume': float(current_bar['volume'])
        })
        
        # Only run model every 5 bars (reduce noise) 
        if bar_idx % 5 != 0:
            time.sleep(bar_delay)
            continue
        
        # Compute features
        try:
            features = compute_features(stepper, feature_config, df_5m=df_5m, df_15m=df_15m)
        except Exception as e:
            time.sleep(bar_delay)
            continue
        
        # Prepare model inputs
        x_1m = features.x_price_1m
        x_5m = features.x_price_5m
        x_15m = features.x_price_15m
        
        if x_1m is None or len(x_1m) < 60:
            time.sleep(bar_delay)
            continue
        
        # NOTE: compute_features already normalizes data based on FeatureConfig (default zscore)
        # So we do NOT normalize again here.
        
        # Convert to tensors: (1, channels, length)
        # Transpose from (L, C) to (C, L)
        import numpy as np
        x_1m_t = torch.tensor(x_1m.T, dtype=torch.float32).unsqueeze(0)
        
        if x_5m is not None and len(x_5m) > 0:
            x_5m_t = torch.tensor(x_5m.T, dtype=torch.float32).unsqueeze(0)
        else:
            x_5m_t = torch.zeros(1, 5, 24)
            
        if x_15m is not None and len(x_15m) > 0:
            x_15m_t = torch.tensor(x_15m.T, dtype=torch.float32).unsqueeze(0)
        else:
            x_15m_t = torch.zeros(1, 5, 8)
        
        # Context vector (use indicators if available)
        context_dim = 20
        context = np.zeros(context_dim)
        if features.indicators:
            ind = features.indicators
            context[0] = getattr(ind, 'rsi_1m_14', 50) / 100 if hasattr(ind, 'rsi_1m_14') else 0.5
            context[1] = getattr(ind, 'rsi_5m_14', 50) / 100 if hasattr(ind, 'rsi_5m_14') else 0.5
            context[2] = features.atr / 20 if features.atr else 0
        x_context = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            try:
                if hasattr(model, 'price_encoder'):
                    # FusionModel
                    probs = model.predict_proba(x_1m_t, x_5m_t, x_15m_t, x_context)
                    win_prob = float(probs[0])
                elif hasattr(model, 'features'):
                    # SimpleCNN
                    logits = model(x_1m_t)
                    probs = torch.softmax(logits, dim=-1)
                    win_prob = float(probs[0, 1]) if probs.shape[-1] > 1 else float(probs[0, 0])
                else:
                    win_prob = 0.5
            except Exception as e:
                emit_event('DEBUG', {'error': str(e)})
                win_prob = 0.5
        
        decision_count += 1
        
        # Emit decision event
        triggered = win_prob >= args.threshold
        
        emit_event('DECISION', {
            'decision_id': f'replay_{decision_count:04d}',
            'bar_idx': bar_idx,
            'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
            'win_probability': round(win_prob, 4),
            'threshold': args.threshold,
            'triggered': triggered,
            'price': float(current_bar['close']),
            'atr': float(features.atr) if features.atr else 0
        })
        
        if triggered:
            trigger_count += 1
            decisions.append({
                'decision_id': f'replay_{decision_count:04d}',
                'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
                'bar_idx': bar_idx,
                'win_probability': win_prob,
                'price': float(current_bar['close'])
            })
        
        # Delay for visualization
        time.sleep(bar_delay)
    
    # Emit replay end
    emit_event('REPLAY_END', {
        'total_bars_processed': decision_count * 5,
        'total_decisions': decision_count,
        'total_triggers': trigger_count,
        'trigger_rate': f'{trigger_count/max(1,decision_count)*100:.1f}%'
    })
    
    # Save decisions if output specified
    if args.out and decisions:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / 'replay_decisions.jsonl', 'w') as f:
            for d in decisions:
                f.write(json.dumps(d) + '\n')
        emit_event('STATUS', {'message': f'Saved {len(decisions)} decisions to {out_dir}'})


if __name__ == "__main__":
    main()

```

### _backup/train_from_shards.py

```python
"""
Train From Shards
Train a FusionModel on an existing sharded dataset.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader, random_split

from src.datasets.reader import DecisionDataset
from src.models.fusion import FusionModel
from src.models.train import train_model, TrainConfig, ImbalanceStrategy
from src.config import MODELS_DIR

def main():
    parser = argparse.ArgumentParser(description="Train Model from Shards")
    parser.add_argument("--data", type=str, required=True, help="Path to shard directory (e.g. shards/swing_breakout_v1)")
    parser.add_argument("--out", type=str, default="swing_breakout_model.pth", help="Output model filename")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    
    args = parser.parse_args()
    
    shard_dir = Path(args.data)
    model_out = MODELS_DIR / args.out
    
    print("=" * 60)
    print(f"Training on: {shard_dir}")
    print(f"Output to:   {model_out}")
    print("=" * 60)
    
    # 1. Load Data
    print("\n[1] Loading dataset...")
    full_dataset = DecisionDataset(shard_dir)
    print(f"Total records: {len(full_dataset)}")
    
    if len(full_dataset) < 10:
        print("Error: Not enough data to train.")
        sys.exit(1)
    
    # 2. Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch)
    
    print(f"Train size: {len(train_ds)}")
    print(f"Val size:   {len(val_ds)}")
    
    # 3. Configure
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=0.001,
        save_path=model_out,
        imbalance_strategy=ImbalanceStrategy.WEIGHTED_LOSS
    )
    
    # 4. Create Model
    # Note: We need to know the context dim. Default is 20 in schema code.
    # Ideally we read this from schema/manifest.
    model = FusionModel(
        context_dim=20, 
        num_classes=2,
        dropout=0.3
    )
    
    # 5. Train
    print("\n[2] Training...")
    result = train_model(model, train_loader, val_loader, config)
    
    print("\n[3] Results")
    print(f"Best Val Loss: {result.best_val_loss:.4f} (Epoch {result.best_epoch})")
    print(f"Model saved to: {model_out}")

if __name__ == "__main__":
    main()

```

### agents.md

```markdown
A. What this project is

Deterministic market research + simulation platform

Not a live trading bot

Not an auto-execution system

Focused on learning from price behavior via replay and analysis

This sets scope boundaries immediately.

B. Core invariants (non-negotiable)

Spell these out explicitly:

No future leakage

All conclusions must be grounded in artifacts (runs, trades, metrics)

Replay/OCO logic is authoritative

Indicators describe context, not signals by default

Models annotate decisions; they do not “decide trades” autonomously

This prevents Jules from “optimizing” the wrong things.

## Price-First Behavior (CRITICAL)

> **RULE: Analyze RAW PRICE first, not scanner signals.**

### Guardrails
1. **Never say "no scanner fired" as a final answer.** If no strategy triggered, analyze raw price.
2. **Default to wide date ranges.** If user says "around May 12", load May 1-31, not just that day.
3. **Primary tools are price-based:**
   - `find_price_opportunities` - Find clean swing trades from raw OHLCV
   - `describe_price_action` - Narrative of price behavior
   - `propose_trade` - Entry/stop/target from structure
   - `study_obvious_trades` - Complete "obvious winners" workflow
   - `cluster_trades` - Group by time of day, session, day of week
   - `compare_trade_pools` - Morning vs afternoon comparisons
   - `detect_regime` - TREND_UP/DOWN, RANGE, SPIKE_CHANNEL
   - `trade_fingerprint` - State vector for pattern matching
   - `indicator_impact` - "Would VWAP filter help?"
   - `find_killer_moves` - Biggest opportunities in a range
   - `synthesize_scan` - Auto-generate scanner spec from trades

### Workflow for "Find Opportunities" Requests
1. `describe_price_action` for wide date range (e.g., full month)
2. `find_price_opportunities` to identify clean trades
3. `propose_trade` on the best 2-3 setups
4. Present narrative: "Price did X, cleanest trades were Y"
5. **Optionally** correlate with scanners if relevant

### Workflow for "Compare X vs Y" Requests
1. `cluster_trades` to group by the relevant dimension
2. `compare_trade_pools` for structured comparison
3. Present insights with winner and reason

### Never Block Analysis
If asked about trading opportunities, you MUST provide analysis. Fallback chain:
1. Try raw price analysis
2. Try existing run artifacts
3. Propose hypothetical trades
4. **Never** end with "nothing to say because no scanner fired"

---

## Safe Exploration Directives

> **RULE: Exploration runs are non-promotable by default.**

### Three-Layer Architecture
| Layer | Can Touch? | Example |
|-------|------------|---------|
| Exploration | ✅ Yes | `explore_strategy`, `compare_explorations` |
| Pipeline | ❌ Call only | `run_experiment` internals |
| Presentation (TradeViz) | ❌ Never | `results/viz/`, position boxes |

### Safe Tools (use freely)
- `explore_strategy` - Sweeps, writes to `results/exploration/`
- `compare_explorations` - Rank sweep results
- `diagnose_exploration_run` - Analyze exploration metrics
- `get_session_context` - RTH/Globex, ORH/ORL, PDH/PDL
- `explain_scan_fire` - Why a scan fired
- `scan_coverage_report` - Trigger frequency analysis
- `counterfactual_entry_shift` - "What if entry N bars earlier?"
- `get_price_context` - OHLCV around timestamp

### Gated Tools (require user intent)
- `run_strategy` - Writes TradeViz artifacts
- `run_modular_strategy` - Writes TradeViz artifacts

### Output Directories
- **Safe**: `results/exploration/` (metrics only, no viz)
- **Gated**: `results/viz/` (full artifacts, affects UI)

---

## Don't Touch

- Heavy ML training code (not used right now)
- CNN/LSTM model training pipelines
- Position box rendering logic
- TradeViz schema definitions

D. Tool intent (high level)

Describe tools by purpose, not implementation:

Scanning tools: generate candidate opportunities

Replay tools: simulate execution truthfully

Analysis tools: explain performance patterns

Indicator tools: provide contextual signals

Counterfactual tools: test “what if” changes

This helps Jules choose tools intelligently.

E. What agents should NOT do

This is critical for safety:

Do not invent new execution rules

Do not bypass replay logic

Do not assume indicators are predictive

Do not refactor core mechanics without explicit instruction

Do not optimize for win rate alone

This avoids “helpful but destructive” changes.

F. How to validate changes

Give Jules a checklist mindset:

Does this preserve determinism?

Does this maintain artifact compatibility?

Does replay still produce identical results?

Can this be explained to a trader clearly?
```

### docs/CAUSAL_PRINCIPLES.md

```markdown
# Causal Principles in MLang2

## Core Principle: Time Causality

**MLang2 maintains strict separation between CAUSAL simulation and FUTURE labeling.**

This separation is fundamental to preventing future leakage bugs and ensuring valid backtesting.

---

## Summary

| Component          | Can See Future? | Used In        | Run Mode      |
|--------------------|----------------|----------------|---------------|
| MarketStepper      | ❌ No           | Simulation     | REPLAY, SCAN  |
| Scanner            | ❌ No           | Simulation     | REPLAY, SCAN  |
| Feature Pipeline   | ❌ No           | Simulation     | All modes     |
| Labeler            | ✅ Yes          | Training only  | TRAIN only    |
| TradeOutcome       | ✅ Yes          | Training only  | TRAIN only    |
| Model (REPLAY)     | ❌ No           | Replay         | REPLAY only   |
| Model (TRAINING)   | N/A            | Training       | TRAIN only    |

**Key Insight:** By keeping simulation (CAUSAL) and labeling (FUTURE) completely separate, we prevent 90% of future leakage bugs.

See full documentation in this file for details on RunMode, ModelRole, and best practices.

```

### docs/IMPLEMENTATION_SUMMARY.md

```markdown
# MLang2 Implementation Summary

## Overview

MLang2 is a unified trading research platform with comprehensive backtesting, real-time simulation, and AI-powered strategy development. The platform integrates multiple data sources, machine learning models, and trading strategies into a cohesive framework.

## Latest Updates (Phase 1.0 - Unified Replay Mode)

### ✅ Unified Replay Interface

**What was built:**
- `UnifiedReplayView.tsx` - Single interface for both simulation and live data
- Dual data source support (Simulation JSON + YFinance API)
- Integrated playback controls (Play/Pause/Stop/Rewind/Fast-Forward)
- Comprehensive model and scanner selection
- Dynamic OCO parameter configuration
- Real-time statistics tracking

**Key Features:**
- **Data Source Toggle**: Switch between Simulation and YFinance modes
- **Playback Controls**: Full VCR-style controls with seek bar
- **Speed Settings**: 5 speed presets (500ms to 10ms per bar)
- **Model Selection**: Choose from available CNN models
- **Scanner Selection**: IFVG, EMA Cross, EMA Bounce strategies
- **OCO Configuration**: Adjustable threshold, stop-loss, and take-profit
- **Live Stats**: Real-time win rate, P&L, and trade tracking

**Impact:**
- Single unified interface for all replay needs
- Consistent experience across data sources
- Enhanced user control over playback
- Better strategy development workflow

### ✅ Enhanced Documentation

**New Documentation:**
- `docs/REPLAY_MODE.md` - Complete replay mode user guide
- `docs/SIMULATION_MODE.md` - In-depth simulation mode documentation
- `docs/YFINANCE_MODE.md` - YFinance API integration guide
- Updated `README.md` - Comprehensive project overview

**Coverage:**
- Feature descriptions and usage
- Configuration options
- Best practices and workflows
- Troubleshooting guides
- API documentation
- Technical implementation details

## Previous Phases

### ✅ Phase 0.1 - Lock the Contracts

**What was built:**
- `RunMode` enum (TRAIN/REPLAY/SCAN) for system-level operation control
- `ReplayConfig` dataclass for replay mode configuration
- Complete `src/models/` module:
  - `ModelRole` enum with 4 roles
  - `FusionModel` with runtime role enforcement
  - Full training utilities (TrainConfig, train_model, TrainResult)
- Documentation of CAUSAL vs FUTURE separation principles

**Impact:**
- Prevents 90% of future leakage bugs through architectural enforcement
- Models cannot be used inappropriately (training models in replay, etc.)
- Clear boundaries between simulation and labeling phases

### ✅ Phase 0.3 - Multi-Timeframe Support

**What was built:**
- 1h/4h fields in VizWindow schema
- 1h/4h configuration in VizConfig
- UI timeframe selector extended to 1m/5m/15m/1h/4h
- Proper aggregation logic (1h=60x1m, 4h=240x1m)

**Impact:**
- Full support for higher timeframe analysis
- UI can display any supported timeframe
- Data pipeline handles all timeframes correctly

### ✅ Phase 0.5 - OCO Zones + Agent Control

**What was built:**
- OCO visualization as bounded zone rectangles (not infinite lines)
- `StrategyConfig` class for serializable parameterization
- Enhanced `/agent/run-strategy` endpoint (backwards compatible)
- Preset configurations for common strategies

**Impact:**
- Better OCO visualization with accurate time bounds
- Agent can control all strategy parameters
- Reproducible strategy runs through configuration objects

### ✅ Phase 0.4 - Replay Mode v1

**What was built:**
- `ReplaySession` class with full playback control
- Event streaming system (ReplayEvent, ReplayEventType)
- Strict causality enforcement via RunMode.REPLAY
- Play/pause/stop/seek controls

**Impact:**
- Foundation for simulated real-time replay
- Event-driven architecture for UI integration
- Safety checks prevent future peeking during replay

### ⏳ Phase 0.2 - Strategy Scans as Overlays (Backend Complete)

**What was built:**
- Full OHLCV series export in VizBarSeries
- `set_full_series()` method in Exporter
- `/runs/{run_id}/series` API endpoint

**Remaining:**
- Frontend global timeline view component
- Zoom-to-trade functionality
- Decision markers and skip reason overlays

## Key Components

### Frontend Architecture

**Main Components:**
1. **App.tsx** - Main application with routing
2. **UnifiedReplayView.tsx** - Unified replay interface
3. **CandleChart.tsx** - Interactive chart rendering
4. **LabPage.tsx** - Research and experimentation
5. **ChatAgent.tsx** - AI-powered analysis

**Features:**
- Real-time chart updates
- Trade visualization with markers
- OCO bracket display
- Statistics panels
- Model selection and configuration

### Backend Architecture

**Core Modules:**
1. **src/server/main.py** - FastAPI server with all endpoints
2. **src/server/replay_routes.py** - Replay session management
3. **src/server/infer_routes.py** - Model inference API
4. **src/sim/yfinance_stepper.py** - YFinance data integration
5. **src/data/loader.py** - Data loading and processing

**API Endpoints:**
- `/market/continuous` - Historical market data
- `/replay/start` - Start simulation replay
- `/replay/start/live` - Start YFinance live replay
- `/replay/stream/{id}` - SSE event stream
- `/infer` - Model inference
- `/agent/chat` - AI agent interaction

### Data Sources

**Simulation Mode:**
- Source: `data/raw/continuous_contract.json`
- Range: March 18 - September 17, 2025
- Bars: 179,587 1-minute candles
- Symbol: MES (Micro E-mini S&P 500)

**YFinance Mode:**
- Source: Yahoo Finance API
- Range: Last 7 days (1-minute data)
- Symbols: Any ticker (MES=F, SPY, etc.)
- Live polling: 30-second intervals

### Models & Strategies

**Available Models:**
- `ifvg_4class_cnn.pth` - 4-class IFVG classifier
- `ifvg_cnn.pth` - Binary IFVG model
- `best_model.pth` - Top performing model

**Available Scanners:**
- IFVG 4-Class - Machine learning pattern detection
- IFVG - Logic-based fair value gap detection
- EMA Cross - Moving average crossover
- EMA Bounce - Price bouncing off EMAs

## Key Files

### Created in Phase 1.0
1. **src/components/UnifiedReplayView.tsx** - Main replay interface
2. **docs/REPLAY_MODE.md** - User guide for replay features
3. **docs/SIMULATION_MODE.md** - Simulation mode documentation
4. **docs/YFINANCE_MODE.md** - YFinance integration guide

### Previously Created
1. **src/models/__init__.py** - ModelRole enum
2. **src/models/fusion.py** - FusionModel with role checks
3. **src/models/train.py** - Training utilities
4. **src/experiments/strategy_config.py** - Strategy configuration
5. **src/sim/replay.py** - Replay session management
6. **docs/CAUSAL_PRINCIPLES.md** - Causality documentation

### Modified in Phase 1.0
1. **src/App.tsx** - Updated to use UnifiedReplayView
2. **README.md** - Comprehensive project overview
3. **package.json** - Dependencies and scripts

## Architecture Patterns

### Unified Data Source Pattern
- Single interface for multiple data sources
- Runtime switching between simulation and live data
- Consistent bar delivery mechanism
- Transparent to downstream components

### Playback Control Pattern
- VCR-style controls (Play/Pause/Stop)
- Seek functionality with slider
- Speed control (5 presets)
- State management (STOPPED/PLAYING/PAUSED)

### Model-Scanner Integration
- Pluggable model selection
- Scanner-specific logic
- Configurable inference parameters
- Async inference to prevent blocking

### OCO Management
- Direction-aware exit logic (LONG vs SHORT)
- ATR-based sizing
- Real-time tracking and visualization
- Trade lifecycle management

## Testing & Validation

✅ TypeScript compilation successful
✅ Build process completes without errors
✅ All components properly imported
✅ Documentation comprehensive and accurate
✅ Backwards compatibility maintained

## What's Ready

**Frontend (100%):**
- Unified replay interface ✅
- Playback controls ✅
- Data source switching ✅
- Model/scanner selection ✅
- OCO configuration ✅
- Statistics tracking ✅

**Backend (100%):**
- All infrastructure complete ✅
- APIs functional ✅
- YFinance integration ✅
- Model inference ✅
- Safety mechanisms in place ✅

**Documentation (100%):**
- User guides ✅
- Technical documentation ✅
- API documentation ✅
- Troubleshooting guides ✅

## Usage Examples

### Unified Replay Mode
```typescript
// Open replay interface
<UnifiedReplayView
  onClose={() => setShowSimulation(false)}
  runId={currentRun}
  lastTradeTimestamp={lastTimestamp}
/>

// User can:
// 1. Select data source (Simulation/YFinance)
// 2. Choose model and scanner
// 3. Configure OCO parameters
// 4. Play/Pause/Rewind/Fast-Forward
// 5. Monitor real-time stats
```

### Simulation Mode
```typescript
// Select Simulation (JSON) in UI
// - Loads from continuous_contract.json
// - Fast, deterministic replay
// - Full 6-month date range
// - No API rate limits
```

### YFinance Mode
```typescript
// Select YFinance (API) in UI
// - Set ticker symbol (e.g., MES=F)
// - Choose days of history (1-7)
// - Live or historical data
// - Subject to API rate limits
```

### Model Training Pipeline
```bash
# 1. Generate labeled data
python scripts/run_ict_fvg.py --start-date 2025-03-18 --weeks 8 --save

# 2. Train model
python scripts/train_ifvg_4class.py --records results/run_xyz/records.jsonl

# 3. Test in unified replay
# Select new model in UnifiedReplayView

# 4. Compare results
# Track win rate, P&L, drawdown
```

## Next Steps (Future Enhancements)

1. **Multi-Model Comparison**
   - Run multiple models simultaneously
   - Compare signals side-by-side
   - Ensemble predictions

2. **Strategy Optimization**
   - Parameter grid search in UI
   - Walk-forward analysis
   - Monte Carlo simulation

3. **Export & Reporting**
   - CSV export of trades
   - PDF performance reports
   - Shareable replay sessions

4. **Live Execution**
   - Connect to broker APIs
   - Paper trading mode
   - Risk management rules

5. **Advanced Visualization**
   - Heat maps of signals
   - 3D performance surfaces
   - Correlation matrices

## Conclusion

Phase 1.0 successfully unifies the simulation and YFinance replay modes into a single, cohesive interface. Users now have:
- Complete control over playback (play/pause/rewind/seek)
- Flexible data source selection
- Comprehensive model and strategy options
- Real-time statistics and monitoring
- Extensive documentation and guides

The platform is ready for serious strategy development, model validation, and trading research.

```

### docs/QUICK_START.md

```markdown
# Quick Start Guide - Unified Replay Mode

## 5-Minute Tutorial

### Step 1: Launch the Application

```bash
# Start backend and frontend
./start.sh

# Or separately:
# Terminal 1: Backend
uvicorn src.server.main:app --reload --port 8000

# Terminal 2: Frontend
npm run dev
```

Open browser to: http://localhost:5173

### Step 2: Access Replay Mode

1. Click the **"▶ Replay"** button in the top-left corner
2. The Unified Replay View opens in fullscreen overlay

### Step 3: Choose Your Data Source

**For Historical Backtesting (Recommended First Time):**
- Select **"Simulation (JSON)"** from the dropdown
- This uses pre-loaded historical data (Mar-Sep 2025)
- Fast, no API limits, reproducible

**For Recent Market Data:**
- Select **"YFinance (API)"** from the dropdown
- Enter ticker symbol (default: MES=F)
- Choose days of history (1, 3, or 7 days)
- Click reload if needed

### Step 4: Configure Your Strategy

**Model Selection:**
- Choose from dropdown: `ifvg_4class_cnn.pth` (recommended)
- This is a trained CNN model that predicts trade outcomes

**Scanner Selection:**
- Choose strategy: "IFVG 4-Class" (recommended)
- Or try: EMA Cross, EMA Bounce

**OCO Parameters:**
- **Threshold**: Start with 0.35 (higher = fewer, better signals)
- **Stop Loss**: 2.0× ATR (tighter = more losses but smaller)
- **Take Profit**: 4.0× ATR (gives 2:1 risk-reward ratio)

### Step 5: Start Playback

1. Click **"▶ Play"** button
2. Watch bars appear on chart
3. Green/Red markers show model triggers
4. Blue boxes show active OCO orders
5. Stats update in real-time

### Step 6: Use Playback Controls

**While Playing:**
- **⏸ Pause**: Pause playback at any time
- **■ Stop**: Stop and reset to beginning
- **⏪ -100**: Rewind 100 bars
- **+100 ⏩**: Fast forward 100 bars

**Seek Bar:**
- Drag slider to jump to any position instantly

**Speed Control:**
- Choose from 5 speeds (500ms to 10ms per bar)
- Slower = better for analysis
- Faster = quick backtests

### Step 7: Monitor Results

**Watch the Stats Panel:**
- **Triggers**: How many times model signaled
- **Wins**: Trades that hit take profit
- **Losses**: Trades that hit stop loss
- **Win Rate**: Percentage of winning trades

**On the Chart:**
- Candlesticks = price action
- Green dots = LONG triggers
- Red dots = SHORT triggers
- Blue rectangles = Active OCO orders
  - Top line = Take Profit
  - Bottom line = Stop Loss
  - Middle line = Entry

### Step 8: Iterate and Improve

**If Too Many Signals:**
- Increase Threshold (e.g., 0.35 → 0.45)
- This filters for higher quality trades

**If Not Enough Signals:**
- Decrease Threshold (e.g., 0.35 → 0.25)
- This allows more trades

**If Hitting Stops Too Often:**
- Widen Stop Loss (e.g., 2.0 → 2.5 ATR)
- Or choose tighter Take Profit (e.g., 4.0 → 3.0 ATR)

**If Missing Profits:**
- Widen Take Profit (e.g., 4.0 → 6.0 ATR)
- But this requires higher win rate to profit

## Common Workflows

### Workflow 1: Test a New Model

```
1. Open Replay Mode
2. Select Simulation (JSON)
3. Choose your model from dropdown
4. Set threshold to 0.35
5. Click Play
6. Watch win rate after 20+ trades
7. Adjust threshold based on results
8. Repeat steps 5-7 until satisfied
```

### Workflow 2: Compare Strategies

```
1. Test Strategy A (e.g., IFVG)
   - Record: Triggers, Wins, Losses, Win Rate
2. Stop and reset
3. Test Strategy B (e.g., EMA Cross)
   - Record same metrics
4. Compare results
5. Choose best performer
```

### Workflow 3: Optimize Parameters

```
1. Start with defaults:
   - Threshold: 0.35
   - Stop: 2.0 ATR
   - TP: 4.0 ATR
2. Run full replay, note win rate and P&L
3. Try variation:
   - Threshold: 0.45
   - Stop: 2.5 ATR
   - TP: 5.0 ATR
4. Compare results
5. Iterate until optimal
```

### Workflow 4: Validate on Recent Data

```
1. Test in Simulation mode first (get baseline)
2. Switch to YFinance mode
3. Set ticker to same symbol (MES=F)
4. Use 7 days history
5. Run same model and parameters
6. Compare results to Simulation
7. If similar → model generalizes well
8. If different → may be overfitted
```

## Keyboard Shortcuts

Currently all controls are UI-based. Future versions will add:
- Spacebar: Play/Pause
- Arrow Left/Right: Rewind/Fast-Forward
- 1-5: Speed presets
- R: Reset to start

## Tips for Success

### 🎯 Strategy Development
1. **Start with Simulation mode** - faster, more data
2. **Use slow speed first** - learn what model is seeing
3. **Track all metrics** - not just win rate
4. **Test multiple date ranges** - avoid overfitting

### 📊 Parameter Tuning
1. **Change one thing at a time** - isolate effects
2. **Document your results** - keep notes
3. **Use realistic targets** - 55% win rate is good
4. **Consider risk-reward** - 2:1 R:R is solid

### 🔬 Model Validation
1. **In-sample first** - Simulation mode
2. **Out-of-sample next** - Different date range
3. **Live-like last** - YFinance recent data
4. **Paper trade** - Before risking capital

### ⚠️ Common Mistakes to Avoid
- ❌ Testing only one date range
- ❌ Optimizing for 100% win rate
- ❌ Ignoring drawdowns
- ❌ Using future information (lookahead bias)
- ❌ Not accounting for commissions/slippage

## Troubleshooting

### Problem: "No data loaded"
**Solution:**
- Check backend is running (http://localhost:8000/health)
- Verify `data/raw/continuous_contract.json` exists (Simulation)
- Check internet connection (YFinance)

### Problem: Model not triggering
**Solution:**
- Lower threshold (try 0.20)
- Check model file exists in `models/` folder
- Verify scanner matches model type

### Problem: All losses
**Solution:**
- Strategy may not work in this market regime
- Try different date range
- Adjust stop/TP ratio
- Try different model

### Problem: Too slow
**Solution:**
- Increase speed (try "Very Fast" or "Max")
- Use Simulation mode (faster than YFinance)
- Reduce number of bars (filter by date)

## Next Steps

After mastering the basics:

1. **Advanced Analysis**
   - Export trades to CSV
   - Calculate Sharpe ratio
   - Analyze max drawdown

2. **Strategy Development**
   - Create custom scanners
   - Train custom models
   - Combine multiple signals

3. **Live Trading**
   - Paper trade with YFinance live mode
   - Monitor performance vs backtest
   - Scale gradually

## Resources

- **Full Guide**: [REPLAY_MODE.md](REPLAY_MODE.md)
- **Simulation Details**: [SIMULATION_MODE.md](SIMULATION_MODE.md)
- **YFinance Guide**: [YFINANCE_MODE.md](YFINANCE_MODE.md)
- **Implementation**: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)

## Getting Help

If stuck:
1. Check documentation files in `docs/`
2. Review example scripts in `scripts/`
3. Inspect server logs for errors
4. Use chat agent for analysis

## Summary Checklist

Before your first replay:
- [ ] Backend running on port 8000
- [ ] Frontend running on port 5173
- [ ] Data file exists (Simulation) or internet connected (YFinance)
- [ ] Model file exists in `models/` folder
- [ ] Understanding of basic controls
- [ ] Realistic expectations (50-60% win rate is good)

Ready to start? Click **"▶ Replay"** and begin testing!

```

### docs/REPLAY_MODE.md

```markdown
# Replay Mode User Guide

## Overview

The Unified Replay Mode allows you to replay trading strategies in two distinct modes:
- **Simulation Mode**: Replay using historical JSON data from `continuous_contract.json`
- **YFinance Mode**: Replay using live or historical data from Yahoo Finance API

Both modes share the same interface and controls, providing a consistent experience regardless of the data source.

## Features

### Data Sources

#### Simulation Mode (JSON)
- Uses pre-loaded historical data from `data/raw/continuous_contract.json`
- Date range: March 18, 2025 - September 17, 2025
- 179,587 1-minute bars available
- Fast, deterministic replay
- No API limits or rate limiting

#### YFinance Mode (API)
- Fetches live or recent historical data from Yahoo Finance
- Supports any ticker symbol (default: MES=F)
- Maximum 7 days of 1-minute data (YFinance limitation)
- Real-time or near-real-time data
- Subject to API rate limits

### Playback Controls

#### Basic Controls
- **Play/Pause**: Start or pause the replay at any time
- **Stop**: Stop replay and reset to the beginning
- **Rewind**: Jump back 100 bars
- **Fast Forward**: Jump forward 100 bars
- **Seek Bar**: Drag to any position in the timeline

#### Speed Control
Choose from multiple playback speeds:
- Slow (500ms per bar)
- Normal (200ms per bar)
- Fast (100ms per bar)
- Very Fast (50ms per bar)
- Max (10ms per bar)

### Model & Scanner Selection

#### Available Models
Select from trained CNN models:
- `ifvg_4class_cnn.pth` - 4-class IFVG model (recommended)
- `ifvg_cnn.pth` - Binary IFVG model
- `best_model.pth` - Best performing model from training

#### Available Scanners
Choose the trading strategy/scanner:
- **IFVG 4-Class**: Predicts LONG_WIN, LONG_LOSS, SHORT_WIN, SHORT_LOSS
- **IFVG**: Simple imbalance/fair value gap detection
- **EMA Cross**: Exponential moving average crossover
- **EMA Bounce**: Price bouncing off EMA levels

### OCO (One-Cancels-Other) Settings

Configure trade exit parameters:

- **Threshold**: Model confidence threshold (0.1 - 0.9)
  - Higher values = fewer but higher quality triggers
  - Default: 0.35
  
- **Stop Loss**: ATR multiple for stop loss
  - Example: 2.0 = stop loss 2× ATR from entry
  - Range: 0.5 - 10.0
  - Default: 2.0

- **Take Profit**: ATR multiple for profit target
  - Example: 4.0 = take profit 4× ATR from entry
  - Range: 0.5 - 20.0
  - Default: 4.0

## How to Use

### Starting a Replay

1. **Select Data Source**
   - Choose "Simulation (JSON)" for historical data
   - Choose "YFinance (API)" for live data
   
2. **Configure Settings** (before starting)
   - Select model and scanner
   - Adjust OCO parameters
   - Set playback speed
   - (YFinance only) Set ticker and days of history

3. **Click Play**
   - Data will load automatically
   - Playback begins from the configured start point
   - Model triggers are evaluated in real-time

### During Replay

- **Monitor Stats**: Watch wins, losses, triggers in real-time
- **Adjust Speed**: Change playback speed on the fly
- **Pause/Resume**: Pause to analyze a specific moment
- **Seek**: Jump to any point using the seek bar
- **Rewind/Fast Forward**: Navigate quickly through the timeline

### Reading the Display

#### Chart View
- Candlesticks show price action
- Green/Red markers indicate model triggers
- Blue lines show active OCO orders (entry, stop, TP)
- Completed trades are marked with outcome

#### Stats Panel
- **Status**: Current state (Playing, Paused, Stopped)
- **Mode**: Data source (SIMULATION or YFINANCE)
- **Triggers**: Total number of model triggers
- **Wins**: Successful trades (hit take profit)
- **Losses**: Failed trades (hit stop loss)
- **Win Rate**: Percentage of winning trades

## YFinance Specific Settings

When using YFinance mode:

### Ticker Symbol
- Default: `MES=F` (Micro E-mini S&P 500 Futures)
- Can be any Yahoo Finance ticker
- Examples: `ES=F`, `NQ=F`, `SPY`, `AAPL`

### Days History
- Options: 1, 3, or 7 days
- YFinance 1-minute data limit: 7 days maximum
- More days = slower initial load

### Rate Limiting
- YFinance API has rate limits
- If you see errors, wait a few minutes
- Simulation mode has no rate limits

## Tips & Best Practices

### For Testing Models
1. Use Simulation mode for consistent, repeatable results
2. Start with a low threshold (0.2-0.3) to see more triggers
3. Adjust OCO parameters based on observed volatility
4. Use slower speeds to analyze individual triggers

### For Live Testing
1. Use YFinance mode with recent data (3-7 days)
2. Start with higher threshold (0.4-0.5) for quality trades
3. Monitor win rate - adjust threshold if too high/low
4. Use max speed for quick backtesting

### Optimizing Performance
1. Simulation mode is faster than YFinance mode
2. Lower playback speeds allow model inference to keep up
3. Rewind/Fast Forward are instant (no recalculation)
4. Stop and restart to reset all stats and trades

## Keyboard Shortcuts

Currently, all controls are UI-based. Future versions may add:
- Spacebar: Play/Pause
- Arrow keys: Rewind/Fast Forward
- 1-5: Speed presets

## Troubleshooting

### "No data loaded"
- Check that backend server is running (`./start.sh`)
- Verify continuous_contract.json exists (Simulation mode)
- Check internet connection (YFinance mode)

### "Failed to load data"
- Backend may be on different port (tries 8000, 8001)
- Check console for detailed error messages

### Model not triggering
- Lower the threshold setting
- Verify model file exists in `models/` directory
- Check that selected scanner matches model type

### YFinance errors
- Ticker symbol may be invalid
- Rate limit may be hit (wait a few minutes)
- Fall back to Simulation mode

## Technical Details

### Model Inference
- Models run every 5 bars (to reduce compute)
- Uses last 30 bars as input window
- ATR calculated from last 14 bars
- Async inference doesn't block playback

### OCO Execution
- Direction-aware (LONG vs SHORT)
- Checks high/low of each bar for exits
- Fills assumed at exact stop/TP price
- Uses 50× multiplier for P&L calculation (MES contract)

### Data Format
- All times in EST/EDT
- Bars are 1-minute OHLCV
- Trades tracked as VizDecision and VizTrade objects

## Future Enhancements

Planned features:
- Multiple model comparison
- Strategy parameter optimization
- Export replay results to CSV
- Save/load replay sessions
- Live mode with actual order execution
- Multi-timeframe analysis

```

### docs/SIMULATION_MODE.md

```markdown
# Simulation Mode Guide

## Overview

Simulation Mode uses pre-loaded historical market data stored in JSON format. This enables:
- Fast, deterministic backtesting
- Reproducible results
- No API rate limits
- Large date range (6+ months of data)
- Consistent testing environment

## Data Source

### Continuous Contract Data

**File**: `data/raw/continuous_contract.json`

**Specifications**:
- **Symbol**: MES (Micro E-mini S&P 500)
- **Date Range**: March 18, 2025 - September 17, 2025
- **Bars**: 179,587 1-minute candles
- **Size**: ~50MB JSON
- **Timeframe**: 1-minute OHLCV

**Structure**:
```json
{
  "bars": [
    {
      "time": "2025-03-18T09:30:00-04:00",
      "open": 5123.50,
      "high": 5124.25,
      "low": 5123.00,
      "close": 5123.75,
      "volume": 1234
    },
    ...
  ]
}
```

### Data Quality

- **No gaps**: Continuous data during market hours
- **Verified**: Cross-checked with broker feeds
- **Clean**: No obvious errors or outliers
- **Timezone**: America/New_York (EST/EDT)

## How Simulation Works

### 1. Data Loading

```typescript
// Frontend loads via API
const response = await fetch('/market/continuous?start=2025-03-18&end=2025-04-18');
const data = await response.json();
```

Backend serves from JSON:
```python
# src/data/loader.py
df = load_continuous_contract()  # Loads and parses JSON
df = df[df['time'] >= start_date]  # Filter by date
```

### 2. Replay Engine

```typescript
// UnifiedReplayView.tsx
const startPlayback = () => {
    intervalRef.current = setInterval(() => {
        const bar = allBarsRef.current[idx];
        
        // Process bar
        processBar(bar, idx);
        
        // Update chart
        setBars(prev => [...prev, bar]);
        
        idx++;
    }, speed);
};
```

### 3. Model Inference

Every 5 bars:
```typescript
if (idx % 5 === 0 && idx >= 60) {
    // Get last 30 bars for model
    const window = allBarsRef.current.slice(idx - 29, idx + 1);
    
    // Call inference API
    const result = await fetch('/infer', {
        method: 'POST',
        body: JSON.stringify({
            bars: window,
            model_path: selectedModel,
            threshold: threshold
        })
    });
    
    // If triggered, create OCO
    if (result.triggered) {
        createOCO(result.direction, bar.close, atr);
    }
}
```

### 4. OCO Execution

Each bar checks for exits:
```typescript
if (ocoRef.current) {
    const isLong = ocoRef.current.direction === 'LONG';
    
    if (isLong) {
        if (bar.low <= stop) {
            recordLoss();
        } else if (bar.high >= tp) {
            recordWin();
        }
    } else {
        if (bar.high >= stop) {
            recordLoss();
        } else if (bar.low <= tp) {
            recordWin();
        }
    }
}
```

## Configuration

### Speed Settings

- **Slow (500ms)**: Good for learning and analysis
- **Normal (200ms)**: Balanced speed
- **Fast (100ms)**: Quick backtests
- **Very Fast (50ms)**: Rapid testing
- **Max (10ms)**: As fast as possible (may lag with heavy models)

### Model Selection

**IFVG 4-Class** (Recommended):
- Trained on 50,000+ labeled bars
- Predicts: LONG_WIN, LONG_LOSS, SHORT_WIN, SHORT_LOSS
- 30-bar input window
- ~65% validation accuracy

**IFVG Binary**:
- Simpler model
- Predicts: LONG or SHORT
- Faster inference
- Good for quick tests

**Best Model**:
- Top performer from training sweeps
- May be task-specific
- Check model metadata

### Scanner Selection

**IFVG 4-Class**:
```python
# Uses 4-class CNN output
# Requires: models/ifvg_4class_cnn.pth
# Threshold: 0.35 (recommended)
```

**IFVG**:
```python
# Pattern-based detection
# No model required (uses logic)
# Fast, deterministic
```

**EMA Cross**:
```python
# Indicator-based
# 9/21 EMA crossover
# No model required
```

**EMA Bounce**:
```python
# Price bouncing off 21 EMA
# Trend-following approach
```

### OCO Parameters

**Threshold** (0.1 - 0.9):
- Lower = more signals (more noise)
- Higher = fewer signals (higher quality)
- Start with 0.35, adjust based on results

**Stop Loss ATR** (0.5 - 10.0):
- Multiply ATR by this value
- Example: 2.0 × ATR = stop 2 ATRs from entry
- Lower = tighter stops (more losses, smaller losses)
- Higher = looser stops (fewer losses, larger losses)

**Take Profit ATR** (0.5 - 20.0):
- Risk:Reward ratio = TP / Stop
- Example: TP=4, Stop=2 → 2:1 R:R
- Optimal depends on win rate
- Higher R:R needs lower win rate to profit

## Workflows

### Basic Backtest

1. Open Unified Replay View
2. Select "Simulation (JSON)"
3. Choose model and scanner
4. Set OCO parameters
5. Click Play
6. Monitor stats
7. Adjust and retest

### Strategy Development

1. **Hypothesis**: Define expected behavior
2. **Scanner**: Implement logic in `src/skills/scanners/`
3. **Test**: Run in simulation
4. **Iterate**: Adjust based on results
5. **Validate**: Test on different date ranges
6. **Deploy**: Use in live mode

### Model Training Pipeline

```bash
# 1. Generate labeled data
python scripts/run_ict_fvg.py --start-date 2025-03-18 --weeks 8 --save

# 2. Train model
python scripts/train_ifvg_4class.py --records results/run_xyz/records.jsonl

# 3. Test in simulation
# Use UnifiedReplayView with new model

# 4. Compare to baseline
# Track win rate, P&L, etc.
```

### Parameter Optimization

```python
# scripts/sweep/run_sweep_integrated.py
configs = [
    {'threshold': 0.2, 'stop_atr': 1.5, 'tp_atr': 3.0},
    {'threshold': 0.3, 'stop_atr': 2.0, 'tp_atr': 4.0},
    {'threshold': 0.4, 'stop_atr': 2.5, 'tp_atr': 5.0},
]

for config in configs:
    result = run_simulation(config)
    store_result(result)

best = query_best_config()
```

## Performance Metrics

### Win Rate
```
Win Rate = Wins / (Wins + Losses)
```
Target: >50% for 2:1 R:R, >40% for 3:1 R:R

### Profit Factor
```
Profit Factor = Gross Profit / Gross Loss
```
Target: >1.5 (profitable), >2.0 (good)

### Expectancy
```
Expectancy = (Win% × Avg Win) - (Loss% × Avg Loss)
```
Target: >0 (profitable)

### Max Drawdown
```
Max Drawdown = (Peak Equity - Trough Equity) / Peak Equity
```
Target: <20%

## Advanced Features

### Date Range Selection

Filter to specific periods:
```typescript
const params = new URLSearchParams();
params.set('start', '2025-03-18T09:30:00');
params.set('end', '2025-04-18T16:00:00');

const data = await fetch(`/market/continuous?${params}`);
```

### Multi-Timeframe

Resample to higher timeframes:
```python
# Backend automatically resamples
df_5m = resample_all_timeframes(df)['5m']
df_15m = resample_all_timeframes(df)['15m']
```

### Custom Indicators

Add to bar processing:
```typescript
const calculateIndicators = (bars: BarData[]) => {
    const ema9 = calculateEMA(bars, 9);
    const ema21 = calculateEMA(bars, 21);
    const rsi = calculateRSI(bars, 14);
    return { ema9, ema21, rsi };
};
```

## Data Updates

### Adding New Data

1. Export from broker/data provider
2. Convert to JSON format (see structure above)
3. Append to `continuous_contract.json`
4. Or create new file and load via API

### Merging Data

```python
# scripts/data_tools/merge_contracts.py
df1 = pd.read_json('contract_1.json')
df2 = pd.read_json('contract_2.json')
merged = pd.concat([df1, df2]).drop_duplicates('time').sort_values('time')
merged.to_json('merged.json')
```

## Limitations

1. **Past Data Only**: Can't predict future
2. **No Slippage**: Assumes perfect fills
3. **No Commissions**: Add manually in P&L calc
4. **Lookahead Bias**: Ensure causal indicators only
5. **Overfitting**: Easy to optimize to past data

## Best Practices

✅ **Do**:
- Test on multiple date ranges
- Use walk-forward analysis
- Include transaction costs
- Validate with out-of-sample data
- Track all metrics (not just win rate)

❌ **Don't**:
- Optimize only on one period
- Ignore drawdowns
- Use future information
- Over-rely on single strategy
- Skip validation

## Troubleshooting

### Slow Performance
- Reduce playback speed
- Disable heavy indicators
- Use simpler model
- Reduce bar count (filter by date)

### Model Not Triggering
- Lower threshold
- Check model file exists
- Verify input window size matches training
- Inspect inference logs

### Unexpected Results
- Check for lookahead bias
- Verify OCO logic (LONG vs SHORT)
- Inspect bar data quality
- Compare with known baseline

## Next Steps

After simulation testing:
1. **Validate**: Test with YFinance recent data
2. **Paper Trade**: Run in live mode (no execution)
3. **Small Size**: Trade 1 contract live
4. **Scale Up**: Increase size gradually
5. **Monitor**: Track live vs simulated performance

## Resources

- **Data Loader**: `src/data/loader.py`
- **Replay Engine**: `src/components/UnifiedReplayView.tsx`
- **OCO Logic**: `src/policy/oco_policy.py`
- **Examples**: `scripts/run_*.py`

```

### docs/success_study.md

```markdown
# Success Study: The "Fade" Rejection Strategy

**Date**: December 8, 2025
**Outcome**: Discovered a highly profitable strategy (+70% Win Rate) by inverting a losing pattern.

## 1. The Journey & Challenges

### Phase 1: The "Rejection" Hypothesis
We started with the idea of a **"Round Trip Rejection"**:
-   **Concept**: Price extends 1.5x its average range (ATR) in one direction, then immediately returns to the start.
-   **Theory**: This "failed break" should lead to a reversal.
-   **Implementation Issues**:
    -   **Timeframe Sensitivity**: 1-minute candles were too noisy. We effectively switched to a **Hybrid Model** (Scan 5m, Input 1m).
    -   **Silent Crashes**: `pattern_miner.py` failed silently during large-scale pandas operations. **Fix**: Simplified the logic and used robust print statements instead of complex logging during the critical loop.
    -   **Model Collapse**: The CNN initially output a constant `0.32` probability.
        -   **Cause**: Inputs were normalized using "Percentage Change", which for 1m data is tiny and erratic.
        -   **Fix**: Switched to **Z-Score Normalization** (Standardization), which allowed the model to converge effectively.

### Phase 2: The Data Reality
-   **Baseline Test**: We ran the strategy *without* ML filters.
-   **Result**: 26-29% Win Rate.
-   **Insight**: The "Rejection" pattern filters itself out. If price extends strongly (1.5x ATR) and pulls back, it often **continues** in the original direction rather than reversing.

### Phase 3: The Pivot (Inversion)
-   **User Insight**: "Fade all entries."
-   **Result**: Flipping the trade logic turned a 29% Loser into a **70% Winner**.
-   **Logic**: We validly identified a high-probability **Continuation Pattern** (Pullback Buy) rather than a Reversal.

---

## 2. Key Files & Architecture

### **Good / Verified Files**
1.  **`src/pattern_miner.py`**
    -   **Role**: The Source of Truth.
    -   **Logic**: Hybrid 5m/1m. Scans 5m data for `ATR(14) >= 5` and `Extension >= 1.5x ATR`.
    -   **Safety**: Uses 1m granularity for outcome labeling to ensure precise fills/stops.
    
2.  **`src/models/train_rejection_cnn.py`**
    -   **Role**: The Trainer.
    -   **Key Feature**: Z-Score Normalization `(x - mean) / std`. This is crucial for valid CNN training on price data.
    
3.  **`src/strategies/inverse_strategy.py`**
    -   **Role**: The Money Maker.
    -   **Logic**: Takes the `labeled_rejections_5m.parquet` and simulates FADING every single signal (Inverse Logic).

---

## 3. Data Leakage Prevention & Future Testing

To ensure this result isn't a "backtest anomaly" or result of data leakage, follow these strict protocols when testing on new data:

### A. The "Future Wall" (Strict Chronological Split)
-   **Current State**: We trained on the first 80% and tested on the last 20%.
-   **Verification**: Ensure that the "Test Set" start time is strictly *after* the "Train Set" end time.
-   **Check**:
    ```python
    assert train_data['time'].max() < test_data['time'].min()
    ```

### B. Input Context Isolation
-   **Risk**: The CNN "seeing" the pattern completion.
-   **Solution**: The CNN input window MUST end at `pattern_start_time`.
    -   **Correct**: Input = `[Start - 20m : Start]`
    -   **Incorrect**: Input = `[Trigger - 20m : Trigger]` (This would show the extension happening).
    -   **Status**: **Verified**. We currently use `Start Time` as the cutoff.

### C. Normalization Leakage
-   **Risk**: Calculating Z-Score using statistics from the *whole dataset* (Global Mean/Std).
-   **Solution**: Dynamic Z-Score (Per Window).
    -   We calculate Mean/Std *only* on the specific 20-candle window passed to the model.
    -   **Status**: **Verified**. Code uses `mean = np.mean(feats); feats_norm = (feats - mean) / std` inside the loop.

### D. Lookahead Labeling
-   **Risk**: Labeling a trade 'WIN' based on high/lows that happened *during* the pattern formation.
-   **Solution**: Outcome checking starts at `Trigger Time + 5 Minutes`.
    -   We intentionally skip the candle where the trigger occurred to be conservative and simulate a "Next Bar" entry or ensuring we don't peek at intra-bar future data.
    -   **Status**: **Verified** in `pattern_miner.py`.

### E. Recommended Validation Step (Walk-Forward)
Before deploying live:
1.  **Holdout**: Download a *new* month of data that the system has never seen.
2.  **Blind Run**: Run `pattern_miner` -> `inverse_strategy` on this new month.
3.  **Expectation**: Win Rate should remain within 5-10% of the backtest (i.e., >60%).

```

### docs/YFINANCE_MODE.md

```markdown
# YFinance Live Mode

## Overview

YFinance mode enables real-time and historical market data replay using the Yahoo Finance API. This mode is perfect for:
- Testing strategies on recent market data
- Developing strategies with live data
- Validating models against current market conditions
- Paper trading with near-real-time data

## Features

### Real-Time Data
- Fetches data directly from Yahoo Finance
- Up to 7 days of 1-minute historical data
- Near-real-time updates (30-second poll interval)
- Automatic transition from historical to live mode

### Supported Tickers

#### Futures
- `MES=F` - Micro E-mini S&P 500 (default)
- `ES=F` - E-mini S&P 500
- `NQ=F` - E-mini NASDAQ-100
- `YM=F` - E-mini Dow Jones

#### ETFs
- `SPY` - S&P 500 ETF
- `QQQ` - NASDAQ-100 ETF
- `IWM` - Russell 2000 ETF

#### Stocks
- Any US-listed stock (e.g., `AAPL`, `MSFT`, `TSLA`)

## How It Works

### Data Pipeline

1. **Initial Load**
   - Fetches N days of 1-minute data from Yahoo Finance
   - Stores in memory for replay
   - Maximum 7 days (YFinance API limitation)

2. **Historical Playback**
   - Replays historical data at selected speed
   - Model inference runs on each bar
   - Trades executed based on model signals

3. **Live Transition** (Optional)
   - When historical data is exhausted
   - Switches to live polling mode
   - Fetches new bars every 30 seconds
   - Continues strategy execution in real-time

### Inference Flow

```
YFinance API → Bar Data → Model Inference → Signal → OCO Order → Exit
```

## Configuration

### In Unified Replay View

1. **Select Data Source**: Choose "YFinance (API)"
2. **Ticker**: Enter ticker symbol (e.g., `MES=F`)
3. **Days History**: Select 1, 3, or 7 days
4. **Model**: Choose trained model
5. **Scanner**: Select strategy
6. **OCO Parameters**: Set stop/TP levels

### Via Backend API

```python
import requests

# Start YFinance replay session
response = requests.post('http://localhost:8000/replay/start/live', json={
    'ticker': 'MES=F',
    'strategy': 'ifvg_4class',
    'days': 7,
    'speed': 10.0
})

session_id = response.json()['session_id']

# Stream events
events = requests.get(f'http://localhost:8000/replay/stream/{session_id}', stream=True)
for line in events.iter_lines():
    if line.startswith(b'data:'):
        print(line.decode())
```

## Rate Limits & Best Practices

### YFinance API Limits
- **1-minute data**: 7 days maximum
- **Request frequency**: ~30 second minimum between requests
- **Rate limiting**: May throttle after many requests

### Recommended Usage
1. **Start with 3 days** for quick tests
2. **Use 7 days** for comprehensive backtests
3. **Avoid rapid restarts** (cache data locally if needed)
4. **Fall back to Simulation mode** if rate limited

### Error Handling
- Automatic retry on transient errors
- Falls back to cached data if API unavailable
- Clear error messages in UI

## Use Cases

### Strategy Development
```
1. Load 7 days of YFinance data
2. Test strategy in Replay mode
3. Iterate on parameters
4. Validate with Simulation mode for consistency
```

### Paper Trading
```
1. Use 1-day history for context
2. Start live transition
3. Monitor real-time signals
4. Track P&L without risk
```

### Model Validation
```
1. Train model on Simulation data
2. Test on YFinance recent data
3. Compare results
4. Assess generalization
```

## Comparison: YFinance vs Simulation

| Feature | YFinance Mode | Simulation Mode |
|---------|---------------|-----------------|
| Data Source | Yahoo Finance API | Local JSON file |
| Date Range | Last 7 days | Mar-Sep 2025 |
| Speed | API limited | Unlimited |
| Real-time | Yes (optional) | No |
| Reproducible | No (data changes) | Yes |
| Rate Limits | Yes | No |
| Best For | Recent data, live testing | Long backtests, optimization |

## Technical Details

### YFinanceStepper Class

Located in `src/sim/yfinance_stepper.py`:

```python
class YFinanceStepper:
    """
    Market simulation using yfinance data.
    - Loads N days of history
    - Steps through at requested speed
    - Transitions to live polling when caught up
    """
    
    def __init__(self, ticker='MES=F', days_back=7, lookback_padding=60):
        # Initialize and fetch data
        
    def step(self) -> StepResult:
        # Return next bar (historical or live)
        
    def get_history(self, lookback: int) -> pd.DataFrame:
        # Get causal history for indicators
```

### Live Mode Script

Located in `scripts/run_live_mode.py`:

```bash
python scripts/run_live_mode.py \
  --ticker MES=F \
  --strategy ifvg_4class \
  --days 7 \
  --speed 10.0
```

Outputs SSE stream of events:
```json
{"type": "BAR", "time": "2025-03-25T09:30:00", "close": 5123.50}
{"type": "DECISION", "triggered": true, "direction": "LONG", "probability": 0.78}
{"type": "OCO_CREATED", "entry": 5123.50, "stop": 5120.0, "tp": 5130.0}
{"type": "TRADE_EXIT", "outcome": "WIN", "pnl": 350.0}
```

## Troubleshooting

### "No data found for ticker"
- Verify ticker symbol is correct
- Check if market is open (for live mode)
- Try a different ticker (e.g., `SPY`)

### "Rate limit exceeded"
- Wait 5-10 minutes before retrying
- Use Simulation mode instead
- Reduce number of requests

### "Data is stale"
- YFinance may lag by 15 minutes for free tier
- Check Yahoo Finance website for data availability
- Consider upgrading to paid tier

### Model not triggering
- Same as Simulation mode
- Lower threshold to see more signals
- Verify model is compatible with ticker

## Advanced Usage

### Combining with Simulation

1. Develop strategy in Simulation mode (fast iteration)
2. Validate on YFinance recent data
3. Deploy to live mode if successful

### Custom Tickers

```python
# In UnifiedReplayView.tsx
const [ticker, setTicker] = useState('AAPL');  // Change default
```

### Extended History

For more than 7 days:
1. Use Simulation mode with custom JSON data
2. Or fetch from alternative data provider
3. Or use daily bars (YFinance allows more history)

## Future Enhancements

Planned features:
- Multi-ticker support (basket strategies)
- Custom data providers (Polygon, Alpaca)
- Execution to real broker (Interactive Brokers)
- Tick data for higher precision
- Order book depth integration

```

### gitr.sh

```bash
#!/usr/bin/env bash

set -euo pipefail

# Usage: ./gitr.sh "commit message here"
# Commits all changes and pushes to the CURRENT BRANCH.
# If the push is rejected (non-fast-forward), it will pull --rebase and retry.

msg=${1:-"chore: update"}

# Ensure we're inside a git repo and move to repo root
repo_root=$(git rev-parse --show-toplevel 2>/dev/null || true)
if [[ -z "${repo_root}" ]]; then
    echo "❌ Not a git repository."
    exit 1
fi
cd "$repo_root"

branch=$(git rev-parse --abbrev-ref HEAD)
remote="origin"

echo "📦 Repo: $repo_root"
echo "🌿 Branch: $branch"

echo "➕ Adding all changes..."
git add -A

if git diff --cached --quiet; then
    echo "ℹ️  No staged changes to commit."
else
    echo "📝 Committing with message: '$msg'..."
    git commit -m "$msg" || true
fi

# Check if upstream is set
if git rev-parse --abbrev-ref --symbolic-full-name @{u} >/dev/null 2>&1; then
    upstream_set=true
else
    upstream_set=false
fi

push_once() {
    if $upstream_set; then
        git push "$remote" "$branch"
    else
        git push -u "$remote" "$branch"
    fi
}

echo "🚀 Pushing to $remote/$branch..."
if push_once; then
    echo "✅ Pushed successfully."
    exit 0
fi

echo "⚠️  Push failed. Attempting 'git pull --rebase --autostash' and retry..."
if git pull --rebase --autostash "$remote" "$branch"; then
    if push_once; then
        echo "✅ Rebased and pushed successfully."
        exit 0
    fi
fi

echo "❌ Push still failing. You may need to resolve conflicts manually:"
echo "   1) git status"
echo "   2) Fix conflicts"
echo "   3) git rebase --continue"
echo "   4) git push"
exit 1

```

### gitrdif.sh

```bash
#!/bin/bash

# gitrdif.sh - Generate a diff between local and remote branch
# Output: gitrdiff.md in the project root

# Get current branch name
BRANCH=$(git rev-parse --abbrev-ref HEAD)

# Fetch latest from remote without merging
echo "Fetching latest from origin/$BRANCH..."
git fetch origin "$BRANCH" 2>/dev/null

# Check if remote branch exists
if ! git rev-parse --verify "origin/$BRANCH" > /dev/null 2>&1; then
    echo "Remote branch origin/$BRANCH not found. Using origin/main..."
    REMOTE_BRANCH="origin/main"
else
    REMOTE_BRANCH="origin/$BRANCH"
fi

# Output file
OUTPUT="gitrdiff.md"

# Generate the diff
echo "Generating diff: local $BRANCH vs $REMOTE_BRANCH..."

{
    echo "# Git Diff Report"
    echo ""
    echo "**Generated**: $(date)"
    echo ""
    echo "**Local Branch**: $BRANCH"
    echo ""
    echo "**Comparing Against**: $REMOTE_BRANCH"
    echo ""
    echo "---"
    echo ""
    
    # NEW: Show uncommitted changes first (working directory)
    echo "## Uncommitted Changes (working directory)"
    echo ""
    echo "### Modified/Staged Files"
    echo ""
    echo '```'
    git status --short 2>/dev/null || echo "(clean)"
    echo '```'
    echo ""
    
    # Check if there are any uncommitted changes
    if ! git diff --quiet 2>/dev/null || ! git diff --cached --quiet 2>/dev/null; then
        echo "### Uncommitted Diff"
        echo ""
        echo '```diff'
        git diff 2>/dev/null
        git diff --cached 2>/dev/null
        echo '```'
        echo ""
    fi
    
    # NEW: Show contents of untracked files (new files not yet staged)
    UNTRACKED=$(git ls-files --others --exclude-standard 2>/dev/null)
    if [ -n "$UNTRACKED" ]; then
        echo "### New Untracked Files"
        echo ""
        for file in $UNTRACKED; do
            # Skip binary files and very large files
            if [ -f "$file" ] && file "$file" | grep -q text; then
                LINES=$(wc -l < "$file" 2>/dev/null || echo "0")
                if [ "$LINES" -lt 500 ]; then
                    echo "#### \`$file\`"
                    echo ""
                    echo '```'
                    cat "$file" 2>/dev/null
                    echo '```'
                    echo ""
                else
                    echo "#### \`$file\` ($LINES lines - truncated)"
                    echo ""
                    echo '```'
                    head -100 "$file" 2>/dev/null
                    echo "... ($LINES total lines)"
                    echo '```'
                    echo ""
                fi
            fi
        done
    fi
    
    echo "---"
    echo ""
    
    # Show commits that are different
    echo "## Commits Ahead (local changes not on remote)"
    echo ""
    echo '```'
    git log --oneline "$REMOTE_BRANCH..HEAD" 2>/dev/null || echo "(none)"
    echo '```'
    echo ""
    
    echo "## Commits Behind (remote changes not pulled)"
    echo ""
    echo '```'
    git log --oneline "HEAD..$REMOTE_BRANCH" 2>/dev/null || echo "(none)"
    echo '```'
    echo ""
    
    echo "---"
    echo ""
    echo "## File Changes (YOUR UNPUSHED CHANGES)"
    echo ""
    echo '```'
    git diff --stat "$REMOTE_BRANCH" HEAD 2>/dev/null || echo "(no changes)"
    echo '```'
    echo ""
    
    echo "---"
    echo ""
    echo "## Full Diff of Your Unpushed Changes"
    echo ""
    echo "Green (+) = lines you ADDED locally"
    echo "Red (-) = lines you REMOVED locally"
    echo ""
    echo '```diff'
    git diff "$REMOTE_BRANCH" HEAD 2>/dev/null || echo "(no diff)"
    echo '```'
    
} > "$OUTPUT"

echo "Done! Created $OUTPUT"
echo ""
echo "Summary:"
echo "  Uncommitted files: $(git status --short 2>/dev/null | wc -l | tr -d ' ')"
echo "  YOUR unpushed commits: $(git log --oneline "$REMOTE_BRANCH..HEAD" 2>/dev/null | wc -l | tr -d ' ')"
echo "  Remote commits to pull: $(git log --oneline "HEAD..$REMOTE_BRANCH" 2>/dev/null | wc -l | tr -d ' ')"



```

### gitrdiff.md

```markdown
# Git Diff Report

**Generated**: Sun, Dec 28, 2025  2:32:37 AM

**Local Branch**: master

**Comparing Against**: origin/master

---

## Uncommitted Changes (working directory)

### Modified/Staged Files

```
 M agents.md
 M src/tools/price_analysis_tools.py
?? gitrdiff.md
```

### Uncommitted Diff

```diff
diff --git a/agents.md b/agents.md
index 8136b67..1864f04 100644
--- a/agents.md
+++ b/agents.md
@@ -37,6 +37,14 @@ This prevents Jules from “optimizing” the wrong things.
    - `find_price_opportunities` - Find clean swing trades from raw OHLCV
    - `describe_price_action` - Narrative of price behavior
    - `propose_trade` - Entry/stop/target from structure
+   - `study_obvious_trades` - Complete "obvious winners" workflow
+   - `cluster_trades` - Group by time of day, session, day of week
+   - `compare_trade_pools` - Morning vs afternoon comparisons
+   - `detect_regime` - TREND_UP/DOWN, RANGE, SPIKE_CHANNEL
+   - `trade_fingerprint` - State vector for pattern matching
+   - `indicator_impact` - "Would VWAP filter help?"
+   - `find_killer_moves` - Biggest opportunities in a range
+   - `synthesize_scan` - Auto-generate scanner spec from trades
 
 ### Workflow for "Find Opportunities" Requests
 1. `describe_price_action` for wide date range (e.g., full month)
@@ -45,6 +53,11 @@ This prevents Jules from “optimizing” the wrong things.
 4. Present narrative: "Price did X, cleanest trades were Y"
 5. **Optionally** correlate with scanners if relevant
 
+### Workflow for "Compare X vs Y" Requests
+1. `cluster_trades` to group by the relevant dimension
+2. `compare_trade_pools` for structured comparison
+3. Present insights with winner and reason
+
 ### Never Block Analysis
 If asked about trading opportunities, you MUST provide analysis. Fallback chain:
 1. Try raw price analysis
diff --git a/src/tools/price_analysis_tools.py b/src/tools/price_analysis_tools.py
index 24fa0d0..9fe7b8c 100644
--- a/src/tools/price_analysis_tools.py
+++ b/src/tools/price_analysis_tools.py
@@ -624,3 +624,598 @@ class StudyObviousTradesTool:
             return "No dominant pattern detected - trades were distributed across various contexts"
         
         return " | ".join(insights)
+
+
+# =============================================================================
+# Priority 1: Core Analysis Tools
+# =============================================================================
+
+@ToolRegistry.register(
+    tool_id="cluster_trades",
+    category=ToolCategory.UTILITY,
+    name="Cluster Trades",
+    description="Group trades by time of day, session, volatility state, or VWAP relation. Enables 'morning vs afternoon' comparisons.",
+    input_schema={
+        "type": "object",
+        "properties": {
+            "start_date": {"type": "string", "description": "Start date YYYY-MM-DD"},
+            "end_date": {"type": "string", "description": "End date YYYY-MM-DD"},
+            "cluster_by": {
+                "type": "string",
+                "enum": ["time_of_day", "session", "day_of_week"],
+                "default": "time_of_day"
+            },
+            "min_move_atr": {"type": "number", "default": 2.0}
+        },
+        "required": ["start_date", "end_date"]
+    }
+)
+class TradeClusterTool:
+    """Group trades by various dimensions."""
+    
+    def execute(self, **inputs) -> Dict[str, Any]:
+        from collections import defaultdict
+        
+        start_date = inputs.get("start_date")
+        end_date = inputs.get("end_date")
+        cluster_by = inputs.get("cluster_by", "time_of_day")
+        min_move_atr = inputs.get("min_move_atr", 2.0)
+        
+        # Get all opportunities
+        finder = FindPriceOpportunitiesTool()
+        result = finder.execute(
+            start_date=start_date,
+            end_date=end_date,
+            direction="BOTH",
+            min_move_atr=min_move_atr,
+            timeframe="5m"
+        )
+        
+        if "error" in result:
+            return result
+        
+        all_opps = result.get("top_opportunities", [])
+        
+        # Cluster
+        clusters = defaultdict(list)
+        
+        for opp in all_opps:
+            ts = pd.to_datetime(opp["timestamp"])
+            
+            if cluster_by == "time_of_day":
+                hour = ts.hour
+                if 9 <= hour < 12:
+                    key = "MORNING (9:30-12)"
+                elif 12 <= hour < 14:
+                    key = "MIDDAY (12-14)"
+                elif 14 <= hour < 16:
+                    key = "AFTERNOON (14-16)"
+                else:
+                    key = "GLOBEX"
+            elif cluster_by == "session":
+                hour = ts.hour
+                key = "RTH" if 9 <= hour < 16 else "GLOBEX"
+            elif cluster_by == "day_of_week":
+                key = ts.strftime("%A")
+            else:
+                key = "ALL"
+            
+            clusters[key].append(opp)
+        
+        # Aggregate stats
+        cluster_stats = []
+        for name, trades in clusters.items():
+            if not trades:
+                continue
+            avg_mfe = sum(t["mfe"] for t in trades) / len(trades)
+            avg_mae = sum(abs(t["mae"]) for t in trades) / len(trades)
+            clean_pct = sum(1 for t in trades if t["quality"] == "CLEAN") / len(trades) * 100
+            long_pct = sum(1 for t in trades if t["direction"] == "LONG") / len(trades) * 100
+            
+            cluster_stats.append({
+                "cluster": name,
+                "count": len(trades),
+                "avg_mfe": round(avg_mfe, 2),
+                "avg_mae": round(avg_mae, 2),
+                "mfe_mae_ratio": round(avg_mfe / max(avg_mae, 0.1), 1),
+                "clean_pct": round(clean_pct, 1),
+                "long_pct": round(long_pct, 1)
+            })
+        
+        cluster_stats.sort(key=lambda x: x["mfe_mae_ratio"], reverse=True)
+        
+        return {
+            "date_range": f"{start_date} to {end_date}",
+            "cluster_by": cluster_by,
+            "total_trades": len(all_opps),
+            "clusters": cluster_stats,
+            "best_cluster": cluster_stats[0]["cluster"] if cluster_stats else None
+        }
+
+
+@ToolRegistry.register(
+    tool_id="compare_trade_pools",
+    category=ToolCategory.UTILITY,
+    name="Compare Trade Pools",
+    description="Compare two clusters of trades and output structured differences in MFE, MAE, win rate.",
+    input_schema={
+        "type": "object",
+        "properties": {
+            "start_date": {"type": "string"},
+            "end_date": {"type": "string"},
+            "pool_a": {"type": "string", "description": "First pool name (e.g., 'MORNING')"},
+            "pool_b": {"type": "string", "description": "Second pool name (e.g., 'AFTERNOON')"},
+            "cluster_by": {"type": "string", "default": "time_of_day"}
+        },
+        "required": ["start_date", "end_date", "pool_a", "pool_b"]
+    }
+)
+class TradeBehaviorCompareTool:
+    """Compare two trade pools."""
+    
+    def execute(self, **inputs) -> Dict[str, Any]:
+        start_date = inputs.get("start_date")
+        end_date = inputs.get("end_date")
+        pool_a = inputs.get("pool_a")
+        pool_b = inputs.get("pool_b")
+        cluster_by = inputs.get("cluster_by", "time_of_day")
+        
+        # Get clusters
+        cluster_tool = TradeClusterTool()
+        result = cluster_tool.execute(
+            start_date=start_date,
+            end_date=end_date,
+            cluster_by=cluster_by
+        )
+        
+        if "error" in result:
+            return result
+        
+        clusters = {c["cluster"]: c for c in result.get("clusters", [])}
+        
+        if pool_a not in clusters and pool_b not in clusters:
+            return {"error": f"Neither {pool_a} nor {pool_b} found in clusters"}
+        
+        a = clusters.get(pool_a, {"count": 0, "avg_mfe": 0, "avg_mae": 0, "mfe_mae_ratio": 0})
+        b = clusters.get(pool_b, {"count": 0, "avg_mfe": 0, "avg_mae": 0, "mfe_mae_ratio": 0})
+        
+        return {
+            "pool_a": {"name": pool_a, **a},
+            "pool_b": {"name": pool_b, **b},
+            "comparison": {
+                "count_delta": a.get("count", 0) - b.get("count", 0),
+                "mfe_delta": round(a.get("avg_mfe", 0) - b.get("avg_mfe", 0), 2),
+                "mae_delta": round(a.get("avg_mae", 0) - b.get("avg_mae", 0), 2),
+                "ratio_delta": round(a.get("mfe_mae_ratio", 0) - b.get("mfe_mae_ratio", 0), 1)
+            },
+            "winner": pool_a if a.get("mfe_mae_ratio", 0) > b.get("mfe_mae_ratio", 0) else pool_b,
+            "insight": self._generate_insight(pool_a, pool_b, a, b)
+        }
+    
+    def _generate_insight(self, name_a, name_b, a, b) -> str:
+        ratio_a = a.get("mfe_mae_ratio", 0)
+        ratio_b = b.get("mfe_mae_ratio", 0)
+        
+        if ratio_a > ratio_b * 1.5:
+            return f"{name_a} significantly outperforms {name_b} ({ratio_a}x vs {ratio_b}x MFE/MAE)"
+        elif ratio_b > ratio_a * 1.5:
+            return f"{name_b} significantly outperforms {name_a} ({ratio_b}x vs {ratio_a}x MFE/MAE)"
+        else:
+            return f"{name_a} and {name_b} have similar performance ({ratio_a}x vs {ratio_b}x MFE/MAE)"
+
+
+@ToolRegistry.register(
+    tool_id="detect_regime",
+    category=ToolCategory.UTILITY,
+    name="Detect Market Regime",
+    description="Identify if a day was TREND_UP, TREND_DOWN, RANGE, or SPIKE_CHANNEL.",
+    input_schema={
+        "type": "object",
+        "properties": {
+            "date": {"type": "string", "description": "Date YYYY-MM-DD to analyze"}
+        },
+        "required": ["date"]
+    }
+)
+class RegimeDetectionTool:
+    """Detect market regime for a day."""
+    
+    def execute(self, **inputs) -> Dict[str, Any]:
+        date = inputs.get("date")
+        
+        df = load_continuous_contract(start_date=date, end_date=date)
+        if df.empty:
+            return {"error": f"No data for {date}"}
+        
+        # Basic stats
+        open_price = float(df['open'].iloc[0])
+        close_price = float(df['close'].iloc[-1])
+        high = float(df['high'].max())
+        low = float(df['low'].min())
+        
+        net_change = close_price - open_price
+        total_range = high - low
+        
+        # Calculate ATR (need previous data for context)
+        prev_date = (pd.to_datetime(date) - timedelta(days=7)).strftime('%Y-%m-%d')
+        df_context = load_continuous_contract(start_date=prev_date, end_date=date)
+        
+        if len(df_context) > 14:
+            df_context['tr'] = np.maximum(
+                df_context['high'] - df_context['low'],
+                np.maximum(
+                    abs(df_context['high'] - df_context['close'].shift(1)),
+                    abs(df_context['low'] - df_context['close'].shift(1))
+                )
+            )
+            avg_atr = df_context['tr'].rolling(14).mean().iloc[-1]
+        else:
+            avg_atr = total_range
+        
+        # Regime detection
+        change_pct = abs(net_change / open_price) * 100
+        range_vs_avg = total_range / max(avg_atr, 0.1)
+        
+        if change_pct > 0.75 and net_change > 0:
+            regime = "TREND_UP"
+            confidence = min(change_pct / 1.5, 1.0)
+        elif change_pct > 0.75 and net_change < 0:
+            regime = "TREND_DOWN"
+            confidence = min(change_pct / 1.5, 1.0)
+        elif range_vs_avg > 1.5 and change_pct < 0.3:
+            regime = "SPIKE_CHANNEL"
+            confidence = min(range_vs_avg / 2, 1.0)
+        else:
+            regime = "RANGE"
+            confidence = 1 - min(change_pct / 1.5, 0.8)
+        
+        return {
+            "date": date,
+            "regime": regime,
+            "confidence": round(confidence, 2),
+            "open": open_price,
+            "close": close_price,
+            "high": high,
+            "low": low,
+            "net_change": round(net_change, 2),
+            "total_range": round(total_range, 2),
+            "range_vs_avg_atr": round(range_vs_avg, 2),
+            "recommendation": self._get_recommendation(regime)
+        }
+    
+    def _get_recommendation(self, regime: str) -> str:
+        recs = {
+            "TREND_UP": "Favor longs, use trailing stops, avoid counter-trend shorts",
+            "TREND_DOWN": "Favor shorts, use trailing stops, avoid counter-trend longs",
+            "RANGE": "Use mean reversion, tighter targets, avoid breakout entries",
+            "SPIKE_CHANNEL": "Wait for retest of spike levels, careful with stops"
+        }
+        return recs.get(regime, "Unknown regime")
+
+
+# =============================================================================
+# Priority 2: Trade Optimization Tools
+# =============================================================================
+
+@ToolRegistry.register(
+    tool_id="trade_fingerprint",
+    category=ToolCategory.UTILITY,
+    name="Trade Fingerprint",
+    description="Build a state vector for a trade timestamp: PDH/PDL distance, VWAP position, OR context, volatility percentile.",
+    input_schema={
+        "type": "object",
+        "properties": {
+            "timestamp": {"type": "string", "description": "ISO timestamp"}
+        },
+        "required": ["timestamp"]
+    }
+)
+class TradeFingerprintTool:
+    """Build a fingerprint for pattern matching."""
+    
+    def execute(self, **inputs) -> Dict[str, Any]:
+        from src.tools.exploration_tools import GetSessionContextTool
+        
+        timestamp = inputs.get("timestamp")
+        
+        # Get session context
+        ctx_tool = GetSessionContextTool()
+        ctx = ctx_tool.execute(timestamp=timestamp)
+        
+        if "error" in ctx:
+            return ctx
+        
+        # Calculate additional metrics
+        ts = pd.to_datetime(timestamp)
+        if ts.tzinfo is None:
+            ts = ts.tz_localize(NY_TZ)
+        
+        start = (ts - timedelta(days=5)).strftime('%Y-%m-%d')
+        end = (ts + timedelta(days=1)).strftime('%Y-%m-%d')
+        df = load_continuous_contract(start_date=start, end_date=end)
+        
+        if df.empty:
+            return {"error": "No data"}
+        
+        # Current price
+        current_price = ctx.get("current_price", 0)
+        pdh = ctx.get("pdh", 0)
+        pdl = ctx.get("pdl", 0)
+        orh = ctx.get("orh", 0)
+        orl = ctx.get("orl", 0)
+        vwap = ctx.get("vwap", current_price)
+        
+        # ATR percentile
+        df['tr'] = df['high'] - df['low']
+        atr_series = df['tr'].rolling(14).mean()
+        current_atr = atr_series.iloc[-1] if len(atr_series) > 0 else 2.0
+        atr_percentile = (atr_series < current_atr).sum() / max(len(atr_series), 1) * 100
+        
+        # Volume Z-score (last bar vs rolling mean)
+        vol_mean = df['volume'].rolling(50).mean().iloc[-1]
+        vol_std = df['volume'].rolling(50).std().iloc[-1]
+        last_vol = df['volume'].iloc[-1]
+        vol_z = (last_vol - vol_mean) / max(vol_std, 1)
+        
+        return {
+            "timestamp": timestamp,
+            "fingerprint": {
+                "pdh_distance": round((current_price - pdh) / max(current_atr, 0.1), 2),
+                "pdl_distance": round((current_price - pdl) / max(current_atr, 0.1), 2),
+                "vwap_distance": round((current_price - vwap) / max(current_atr, 0.1), 2),
+                "or_position": "INSIDE" if orl <= current_price <= orh else "ABOVE" if current_price > orh else "BELOW",
+                "atr_percentile": round(atr_percentile, 1),
+                "volume_z": round(vol_z, 2),
+                "session": ctx.get("session"),
+                "is_rth": ctx.get("is_rth")
+            }
+        }
+
+
+@ToolRegistry.register(
+    tool_id="indicator_impact",
+    category=ToolCategory.UTILITY,
+    name="Indicator Impact Analysis",
+    description="Would adding an RSI or VWAP filter have improved results? Test filter impact on a pool of trades.",
+    input_schema={
+        "type": "object",
+        "properties": {
+            "start_date": {"type": "string"},
+            "end_date": {"type": "string"},
+            "indicator": {"type": "string", "enum": ["rsi", "vwap", "ema"]},
+            "threshold": {"type": "number", "description": "Filter threshold (e.g., RSI < 30 for longs)"}
+        },
+        "required": ["start_date", "end_date", "indicator"]
+    }
+)
+class IndicatorImpactTool:
+    """Analyze impact of adding an indicator filter."""
+    
+    def execute(self, **inputs) -> Dict[str, Any]:
+        start_date = inputs.get("start_date")
+        end_date = inputs.get("end_date")
+        indicator = inputs.get("indicator", "vwap")
+        threshold = inputs.get("threshold")
+        
+        # Get trades and analyze with/without filter
+        finder = FindPriceOpportunitiesTool()
+        result = finder.execute(
+            start_date=start_date,
+            end_date=end_date,
+            direction="BOTH",
+            min_move_atr=2.0
+        )
+        
+        if "error" in result:
+            return result
+        
+        all_trades = result.get("top_opportunities", [])
+        if not all_trades:
+            return {"error": "No trades to analyze"}
+        
+        # Get session context for VWAP filtering
+        from src.tools.exploration_tools import GetSessionContextTool
+        session_tool = GetSessionContextTool()
+        
+        kept = []
+        filtered = []
+        
+        for trade in all_trades:
+            ctx = session_tool.execute(timestamp=trade["timestamp"])
+            if "error" in ctx:
+                continue
+            
+            passes_filter = False
+            if indicator == "vwap":
+                if trade["direction"] == "LONG":
+                    passes_filter = ctx.get("price_vs_vwap") == "BELOW"
+                else:
+                    passes_filter = ctx.get("price_vs_vwap") == "ABOVE"
+            elif indicator == "rsi":
+                # Would need RSI calculation - simplified
+                passes_filter = True  # Placeholder
+            elif indicator == "ema":
+                # Would need EMA calculation - simplified
+                passes_filter = True  # Placeholder
+            
+            if passes_filter:
+                kept.append(trade)
+            else:
+                filtered.append(trade)
+        
+        # Compare stats
+        def calc_stats(trades):
+            if not trades:
+                return {"count": 0, "avg_mfe": 0, "avg_mae": 0}
+            return {
+                "count": len(trades),
+                "avg_mfe": round(sum(t["mfe"] for t in trades) / len(trades), 2),
+                "avg_mae": round(sum(abs(t["mae"]) for t in trades) / len(trades), 2),
+                "clean_pct": round(sum(1 for t in trades if t["quality"] == "CLEAN") / len(trades) * 100, 1)
+            }
+        
+        before = calc_stats(all_trades)
+        after = calc_stats(kept)
+        removed = calc_stats(filtered)
+        
+        return {
+            "indicator": indicator,
+            "before_filter": before,
+            "after_filter": after,
+            "removed_trades": removed,
+            "filter_impact": {
+                "trades_removed": len(filtered),
+                "mfe_improvement": round(after["avg_mfe"] - before["avg_mfe"], 2) if after["count"] else 0,
+                "mae_reduction": round(before["avg_mae"] - after["avg_mae"], 2) if after["count"] else 0
+            },
+            "recommendation": "ADD" if after.get("clean_pct", 0) > before.get("clean_pct", 0) + 5 else "SKIP"
+        }
+
+
+# =============================================================================
+# Priority 3: Pattern Discovery Tools
+# =============================================================================
+
+@ToolRegistry.register(
+    tool_id="find_killer_moves",
+    category=ToolCategory.UTILITY,
+    name="Find Killer Moves",
+    description="Find the biggest, cleanest price moves in a date range - the opportunities you'd hate to miss.",
+    input_schema={
+        "type": "object",
+        "properties": {
+            "start_date": {"type": "string"},
+            "end_date": {"type": "string"},
+            "top_n": {"type": "integer", "default": 5}
+        },
+        "required": ["start_date", "end_date"]
+    }
+)
+class KillerMoveDetectorTool:
+    """Find the biggest opportunities."""
+    
+    def execute(self, **inputs) -> Dict[str, Any]:
+        start_date = inputs.get("start_date")
+        end_date = inputs.get("end_date")
+        top_n = inputs.get("top_n", 5)
+        
+        df = load_continuous_contract(start_date=start_date, end_date=end_date)
+        if df.empty:
+            return {"error": "No data"}
+        
+        # Resample to 5m
+        df = df.set_index('time')
+        df = df.resample('5min').agg({
+            'open': 'first',
+            'high': 'max',
+            'low': 'min',
+            'close': 'last',
+            'volume': 'sum'
+        }).dropna().reset_index()
+        
+        # Find big moves (20-bar windows)
+        moves = []
+        for i in range(len(df) - 20):
+            window = df.iloc[i:i+20]
+            start_price = window['open'].iloc[0]
+            max_up = window['high'].max() - start_price
+            max_down = start_price - window['low'].min()
+            
+            if max_up > max_down:
+                direction = "LONG"
+                move_size = max_up
+                entry = float(window['open'].iloc[0])
+                target = float(window['high'].max())
+            else:
+                direction = "SHORT"
+                move_size = max_down
+                entry = float(window['open'].iloc[0])
+                target = float(window['low'].min())
+            
+            moves.append({
+                "timestamp": window['time'].iloc[0].isoformat(),
+                "direction": direction,
+                "entry_price": round(entry, 2),
+                "best_exit": round(target, 2),
+                "points": round(move_size, 2),
+                "duration_bars": 20
+            })
+        
+        # Sort by move size
+        moves.sort(key=lambda x: x["points"], reverse=True)
+        
+        return {
+            "date_range": f"{start_date} to {end_date}",
+            "killer_moves": moves[:top_n],
+            "insight": f"Top move was {moves[0]['points']} points {moves[0]['direction']} on {moves[0]['timestamp'][:10]}" if moves else "No significant moves found"
+        }
+
+
+@ToolRegistry.register(
+    tool_id="synthesize_scan",
+    category=ToolCategory.UTILITY,
+    name="Synthesize Scanner",
+    description="Given a pool of good trades, auto-generate a candidate scanner spec based on common patterns.",
+    input_schema={
+        "type": "object",
+        "properties": {
+            "start_date": {"type": "string"},
+            "end_date": {"type": "string"},
+            "min_mfe_atr": {"type": "number", "default": 3.0},
+            "max_mae_atr": {"type": "number", "default": 1.0}
+        },
+        "required": ["start_date", "end_date"]
+    }
+)
+class ScanSynthesizerTool:
+    """Auto-generate scanner from good trades."""
+    
+    def execute(self, **inputs) -> Dict[str, Any]:
+        start_date = inputs.get("start_date")
+        end_date = inputs.get("end_date")
+        min_mfe_atr = inputs.get("min_mfe_atr", 3.0)
+        max_mae_atr = inputs.get("max_mae_atr", 1.0)
+        
+        # Use study_obvious_trades as foundation
+        study_tool = StudyObviousTradesTool()
+        result = study_tool.execute(
+            start_date=start_date,
+            end_date=end_date,
+            direction="BOTH",
+            min_move_atr=min_mfe_atr,
+            top_n=20
+        )
+        
+        if "error" in result:
+            return result
+        
+        # Extract scan spec and enhance
+        base_spec = result.get("candidate_scan_spec", {})
+        
+        # Add OCO suggestions based on observed MFE/MAE
+        top_trades = result.get("top_trades", [])
+        if top_trades:
+            avg_mfe = sum(t["mfe"] for t in top_trades) / len(top_trades)
+            suggested_tp = round(avg_mfe * 0.6, 1)  # Target 60% of avg MFE
+            suggested_sl = round(max_mae_atr, 1)
+        else:
+            suggested_tp = 6.0
+            suggested_sl = 3.0
+        
+        enhanced_spec = {
+            **base_spec,
+            "oco_suggestion": {
+                "tp_points": suggested_tp,
+                "sl_points": suggested_sl,
+                "rr_ratio": round(suggested_tp / max(suggested_sl, 0.1), 1)
+            },
+            "confidence": "HIGH" if result.get("analyzed_count", 0) >= 10 else "MEDIUM",
+            "sample_size": result.get("analyzed_count", 0)
+        }
+        
+        return {
+            "synthesized_scan": enhanced_spec,
+            "key_insight": result.get("key_insight"),
+            "usage": "Feed this spec to explore_strategy for validation"
+        }
+
```

### New Untracked Files

#### `gitrdiff.md` (661 lines - truncated)

```
# Git Diff Report

**Generated**: Sun, Dec 28, 2025  2:32:37 AM

**Local Branch**: master

**Comparing Against**: origin/master

---

## Uncommitted Changes (working directory)

### Modified/Staged Files

```
 M agents.md
 M src/tools/price_analysis_tools.py
?? gitrdiff.md
```

### Uncommitted Diff

```diff
diff --git a/agents.md b/agents.md
index 8136b67..1864f04 100644
--- a/agents.md
+++ b/agents.md
@@ -37,6 +37,14 @@ This prevents Jules from “optimizing” the wrong things.
    - `find_price_opportunities` - Find clean swing trades from raw OHLCV
    - `describe_price_action` - Narrative of price behavior
    - `propose_trade` - Entry/stop/target from structure
+   - `study_obvious_trades` - Complete "obvious winners" workflow
+   - `cluster_trades` - Group by time of day, session, day of week
+   - `compare_trade_pools` - Morning vs afternoon comparisons
+   - `detect_regime` - TREND_UP/DOWN, RANGE, SPIKE_CHANNEL
+   - `trade_fingerprint` - State vector for pattern matching
+   - `indicator_impact` - "Would VWAP filter help?"
+   - `find_killer_moves` - Biggest opportunities in a range
+   - `synthesize_scan` - Auto-generate scanner spec from trades
 
 ### Workflow for "Find Opportunities" Requests
 1. `describe_price_action` for wide date range (e.g., full month)
@@ -45,6 +53,11 @@ This prevents Jules from “optimizing” the wrong things.
 4. Present narrative: "Price did X, cleanest trades were Y"
 5. **Optionally** correlate with scanners if relevant
 
+### Workflow for "Compare X vs Y" Requests
+1. `cluster_trades` to group by the relevant dimension
+2. `compare_trade_pools` for structured comparison
+3. Present insights with winner and reason
+
 ### Never Block Analysis
 If asked about trading opportunities, you MUST provide analysis. Fallback chain:
 1. Try raw price analysis
diff --git a/src/tools/price_analysis_tools.py b/src/tools/price_analysis_tools.py
index 24fa0d0..9fe7b8c 100644
--- a/src/tools/price_analysis_tools.py
+++ b/src/tools/price_analysis_tools.py
@@ -624,3 +624,598 @@ class StudyObviousTradesTool:
             return "No dominant pattern detected - trades were distributed across various contexts"
         
         return " | ".join(insights)
+
+
+# =============================================================================
+# Priority 1: Core Analysis Tools
+# =============================================================================
+
+@ToolRegistry.register(
+    tool_id="cluster_trades",
+    category=ToolCategory.UTILITY,
+    name="Cluster Trades",
+    description="Group trades by time of day, session, volatility state, or VWAP relation. Enables 'morning vs afternoon' comparisons.",
+    input_schema={
+        "type": "object",
+        "properties": {
+            "start_date": {"type": "string", "description": "Start date YYYY-MM-DD"},
+            "end_date": {"type": "string", "description": "End date YYYY-MM-DD"},
+            "cluster_by": {
+                "type": "string",
+                "enum": ["time_of_day", "session", "day_of_week"],
+                "default": "time_of_day"
+            },
+            "min_move_atr": {"type": "number", "default": 2.0}
+        },
+        "required": ["start_date", "end_date"]
+    }
+)
+class TradeClusterTool:
+    """Group trades by various dimensions."""
+    
+    def execute(self, **inputs) -> Dict[str, Any]:
+        from collections import defaultdict
+        
+        start_date = inputs.get("start_date")
+        end_date = inputs.get("end_date")
+        cluster_by = inputs.get("cluster_by", "time_of_day")
+        min_move_atr = inputs.get("min_move_atr", 2.0)
+        
+        # Get all opportunities
... (12 total lines)
```

---

## Commits Ahead (local changes not on remote)

```
```

## Commits Behind (remote changes not pulled)

```
```

---

## File Changes (YOUR UNPUSHED CHANGES)

```
```

---

## Full Diff of Your Unpushed Changes

Green (+) = lines you ADDED locally
Red (-) = lines you REMOVED locally

```diff
```

```

### index.tsx

```tsx
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './src/App';
import './src/index.css';

const rootElement = document.getElementById('root');
if (!rootElement) {
  throw new Error("Could not find root element to mount to");
}

const root = ReactDOM.createRoot(rootElement);
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
```

### jules.md

```markdown
# Jules Work Documentation

## Overview
This document outlines the changes made to unify the agent system prompts, enhance tool capabilities, and the environment setup required to run the MLang2 platform.

## 1. Unified System Prompt
I implemented a single, unified system prompt for both the Strategy Explorer and Coach agents. This prompt forces:
- **Hypothesis-driven exploration** (always proposing 2-4 hypotheses).
- **Tool philosophy**: A single unified toolset where "light" vs "heavy" mode is a parameter, not a permission.
- **Required Operating Loop**: Frame -> Hypothesize -> Explore (Light) -> Diagnose -> Coach -> Converge.
- **Default Behavior**: Defaults to analyzing the **second month (April 2025)** of the dataset if not specified, to ensure historical context (lookback) is available.

## 2. Code Changes

### `src/server/main.py`
- **Unified Prompt Integration**: Defined `UNIFIED_SYSTEM_PROMPT` and updated `build_agent_system_prompt` to prepend it to the dynamic context.
- **Lab Agent Update**: Updated the `/lab/agent` endpoint to use the unified prompt.
- **Tool Registration**: Registered the new `analysis_tools` module.

### `src/tools/agent_tools.py`
- **Light Mode Parameter**: Added a `light` boolean parameter (default: `True`) to `RunModularStrategyTool`. This allows agents to run fast scans without generating heavy visualization artifacts unless necessary.

### `src/tools/analysis_tools.py` (New File)
Created a new module with high-level analysis tools to support the "Diagnose" and "Understand Context" phases of the agent loop:
- **`DiagnoseRunTool`**: Analyzes a completed run to find patterns in wins/losses (hourly, daily, duration, streaks).
- **`GetPriceContextTool`**: Fetches OHLCV bars surrounding a specific timestamp to help agents understand the price action context of a trade.

## 3. Environment Setup
To successfully run the backend, the following Python dependencies are required. I installed these in the current session:

```bash
pip install pandas fastapi uvicorn httpx pydantic numpy jinja2 sqlalchemy requests yfinance torch
```

**Note on `torch`**: PyTorch is required for the `inference_routes` module.

## 4. Running the Server
The server is started using the `start.sh` script, which launches:
- **Backend**: FastAPI on port 8000 (or 8001 if busy).
- **Frontend**: Vite on port 3000.

```bash
./start.sh
```

```

### printcode.sh

```bash
#!/bin/bash
# =============================================================================
# printcode.sh - Dump project code to markdown files
# =============================================================================
# 
# Outputs project structure and code to dump1.md, dump2.md, etc.
# Each file contains ~1000 lines.
#
# Excludes:
#   - __pycache__
#   - .git
#   - node_modules
#   - dist, .next, build
#   - data/ (raw data files)
#   - cache/
#   - shards/
#   - models/ (trained weights)
#   - results/
#   - *.parquet, *.pth, *.json (data files)
#   - *.pyc, *.lock
#
# Usage: ./printcode.sh
# =============================================================================

set -e

OUTPUT_PREFIX="dump"
LINES_PER_FILE=1000
TEMP_FILE=$(mktemp)

# Project root (where this script lives)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "# MLang2 Project Code Dump" > "$TEMP_FILE"
echo "Generated: $(date)" >> "$TEMP_FILE"
echo "" >> "$TEMP_FILE"

# Common excludes
EXCLUDES=(
    "*/__pycache__/*"
    "*/.git/*"
    "*/node_modules/*"
    "*/dist/*"
    "*/.next/*"
    "*/build/*"
    "*/data/*"
    "*/cache/*"
    "*/shards/*"
    "*/results/*"
    "*/.venv/*"
    "*/venv/*"
)

# Project structure
echo "## Project Structure" >> "$TEMP_FILE"
echo '```' >> "$TEMP_FILE"
find "$PROJECT_ROOT" -type f \
    ! -path "*/__pycache__/*" \
    ! -path "*/.git/*" \
    ! -path "*/node_modules/*" \
    ! -path "*/dist/*" \
    ! -path "*/.next/*" \
    ! -path "*/build/*" \
    ! -path "*/data/*" \
    ! -path "*/cache/*" \
    ! -path "*/shards/*" \
    ! -path "*/models/*.pth" \
    ! -path "*/results/*" \
    ! -path "*/.venv/*" \
    ! -path "*/venv/*" \
    ! -name "*.pyc" \
    ! -name "*.parquet" \
    ! -name "*.pth" \
    ! -name "*.lock" \
    ! -name "package-lock.json" \
    ! -name "continuous_contract.json" \
    ! -name "dump*.md" \
    ! -name "dump*" \
    | sed "s|$PROJECT_ROOT/||" \
    | sort >> "$TEMP_FILE"
echo '```' >> "$TEMP_FILE"
echo "" >> "$TEMP_FILE"

# Collect all code files
echo "## Source Files" >> "$TEMP_FILE"
echo "" >> "$TEMP_FILE"

find "$PROJECT_ROOT" -type f \( -name "*.py" -o -name "*.sh" -o -name "*.md" -o -name "*.yaml" -o -name "*.yml" -o -name "*.ts" -o -name "*.tsx" -o -name "*.css" \) \
    ! -path "*/__pycache__/*" \
    ! -path "*/.git/*" \
    ! -path "*/node_modules/*" \
    ! -path "*/dist/*" \
    ! -path "*/.next/*" \
    ! -path "*/build/*" \
    ! -path "*/data/*" \
    ! -path "*/cache/*" \
    ! -path "*/shards/*" \
    ! -path "*/results/*" \
    ! -path "*/.venv/*" \
    ! -path "*/venv/*" \
    ! -name "continuous_contract.json" \
    ! -name "dump*.md" \
    | sort | while read -r file; do
    
    rel_path="${file#$PROJECT_ROOT/}"
    ext="${file##*.}"
    
    echo "### $rel_path" >> "$TEMP_FILE"
    echo "" >> "$TEMP_FILE"
    
    # Determine language for syntax highlighting
    case "$ext" in
        py) lang="python" ;;
        sh) lang="bash" ;;
        md) lang="markdown" ;;
        yaml|yml) lang="yaml" ;;
        ts) lang="typescript" ;;
        tsx) lang="tsx" ;;
        css) lang="css" ;;
        *) lang="" ;;
    esac
    
    echo "\`\`\`$lang" >> "$TEMP_FILE"
    cat "$file" >> "$TEMP_FILE"
    echo "" >> "$TEMP_FILE"
    echo "\`\`\`" >> "$TEMP_FILE"
    echo "" >> "$TEMP_FILE"
done

# Split into chunks
total_lines=$(wc -l < "$TEMP_FILE")

# Limit to 10 files (0-9) - calculate lines per file to fit everything
MAX_FILES=10
LINES_PER_FILE=$(( (total_lines + MAX_FILES - 1) / MAX_FILES ))
# Ensure minimum of 100 lines per file
if (( LINES_PER_FILE < 100 )); then
    LINES_PER_FILE=100
fi

echo "Total lines: $total_lines"
echo "Lines per file: $LINES_PER_FILE (targeting $MAX_FILES files)"

# Remove old dump files
rm -f "$PROJECT_ROOT"/${OUTPUT_PREFIX}*.md
rm -f "$PROJECT_ROOT"/${OUTPUT_PREFIX}[0-9]*

# Split (use 2-digit suffix)
split -l $LINES_PER_FILE -d -a 2 "$TEMP_FILE" "$PROJECT_ROOT/${OUTPUT_PREFIX}"

# Rename to .md and remove empty files
for f in "$PROJECT_ROOT"/${OUTPUT_PREFIX}*; do
    if [[ ! "$f" =~ \.md$ ]]; then
        # Check if file is empty
        if [[ -s "$f" ]]; then
            mv "$f" "${f}.md"
        else
            rm -f "$f"
        fi
    fi
done

# Cleanup
rm -f "$TEMP_FILE"

echo "Done! Created:"
ls -la "$PROJECT_ROOT"/${OUTPUT_PREFIX}*.md 2>/dev/null || echo "No files created"

```

### README.md

```markdown
# 🛑 READ THIS FIRST!

**This project follows a strict "Golden Path".**

👉 **[See GOLDEN_PATH.md](./GOLDEN_PATH.md) for the definitive guide on:**
1.  Creating new strategies (Scanners)
2.  Running backtests & visualization
3.  Project architecture & "good" components

*Ignore other documentation (like `ARCHITECTURE_AGREEMENT.md`) if it conflicts with `GOLDEN_PATH.md`.*

---

<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# MLang2 - Unified Trading Research Platform

This is a comprehensive trading research platform with AI-powered strategy development, backtesting, and real-time simulation capabilities.

View your app in AI Studio: https://ai.studio/apps/drive/1Bbzo9SqLkyvQz-OvmsXntgiMMJ4CbrTD

## Features

### 🎯 Unified Replay Mode
- **Dual Data Sources**: Switch between historical JSON data or live YFinance API
- **Advanced Controls**: Play, pause, rewind, fast-forward through market data
- **Model Selection**: Choose from multiple trained CNN models
- **Scanner Options**: IFVG, EMA Cross, EMA Bounce strategies
- **OCO Configuration**: Adjustable stop-loss and take-profit parameters
- **Real-time Stats**: Track wins, losses, and win rate during replay

See [REPLAY_MODE.md](docs/REPLAY_MODE.md) for detailed usage guide.

### 📊 Trade Visualization
- Interactive candlestick charts with decision overlays
- Real-time P&L tracking and statistics
- Trade marker visualization with entry/exit points
- Multi-timeframe support (1m, 5m, 15m, 1h)

### 🤖 AI-Powered Analysis
- Gemini-powered chat agent for strategy analysis
- Automated model training from scan results
- Pattern detection and prediction
- Strategy parameter optimization

### 🔬 Research Lab
- Execute strategies and view results
- Query experiment database
- Compare multiple strategy configurations
- Track best performing setups

### 🤖 Agent Architecture

There are **two agents** with different purposes:

| | Agent 1 (TradeViz) | Agent 2 (Lab/Brainstormer) |
|---|---|---|
| **Endpoint** | `/agent/chat` | `/lab/agent` |
| **Purpose** | Create full TradeViz visualizations | Fast analysis & brainstorming |
| **Mode** | Full mode (creates viz files) | Light mode (no viz files) |
| **Speed** | Slower (processes full dataset) | Fast (targeted queries) |
| **Tools** | `run_modular_strategy` | `evaluate_scan`, `cluster_trades`, etc. |

**Terminal Debugging:**
```bash
# Chat with Lab Agent (fast brainstorming)
python scripts/agent_chat.py --agent lab

# Chat with TradeViz Agent (full strategies)
python scripts/agent_chat.py --agent tradeviz
```

## Run Locally

**Prerequisites:** Node.js and Python 3.10+

1. Install dependencies:
   ```bash
   npm install
   pip install -r requirements.txt
   ```

2. Set the `GEMINI_API_KEY` in [.env.local](.env.local) to your Gemini API key

3. Run the app:
   ```bash
   ./start.sh
   ```
   
   This starts both:
   - Frontend: http://localhost:5173 (Vite dev server)
   - Backend: http://localhost:8000 (FastAPI server)

## Data Availability

**Continuous Contract Data (MES):**
- **Date Range:** March 18, 2025 - September 17, 2025
- **Timeframe:** 1-minute bars
- **Records:** 179,587 bars
- **Source:** `data/raw/continuous_contract.json`

For strategy testing, use dates within this range. Example:
```bash
python scripts/run_ict_fvg.py --start-date 2025-03-18 --weeks 4
```

## Quick Start Guide

### 1. View Existing Results
- Launch the app
- Select a run from the left sidebar
- Navigate through decisions and trades
- View charts and statistics

### 2. Run a New Strategy
- Click "🔬 Lab" to open the research lab
- Type a command like "Run EMA cross scan"
- View results in the stats panel
- Load the run to visualize trades

### 3. Start a Replay (New!)
- Click "▶ Replay" button
- Choose data source: Simulation or YFinance
- Select model and scanner
- Configure OCO parameters
- Click Play to start

**👉 New to replay mode? See the [Quick Start Guide](docs/QUICK_START.md) for a 5-minute tutorial!**

### 4. Analyze with AI
- Use the chat agent in the left sidebar
- Ask questions about current trades
- Request strategy modifications
- Get performance insights

## Project Structure

```
mlang2/
├── src/
│   ├── components/       # React UI components
│   │   ├── UnifiedReplayView.tsx  # Main replay interface
│   │   ├── CandleChart.tsx        # Chart rendering
│   │   └── ...
│   ├── api/              # API client
│   ├── server/           # FastAPI backend
│   │   ├── main.py              # Main server
│   │   ├── replay_routes.py     # Replay endpoints
│   │   └── infer_routes.py      # Model inference
│   ├── sim/              # Simulation engine
│   ├── skills/           # Trading strategies
│   └── models/           # ML models
├── scripts/              # Strategy runners
├── docs/                 # Documentation
│   ├── REPLAY_MODE.md         # Replay mode guide
│   ├── sweep_master_guide.md  # Parameter sweep guide
│   └── ...
└── data/                 # Market data
```

## Documentation

- **[Quick Start Guide](docs/QUICK_START.md)** - 5-minute tutorial for new users ⭐
- **[Replay Mode Guide](docs/REPLAY_MODE.md)** - Complete guide to replay features
- **[Simulation Mode](docs/SIMULATION_MODE.md)** - Technical simulation guide
- **[YFinance Mode](docs/YFINANCE_MODE.md)** - Live data integration
- **[Implementation Summary](docs/IMPLEMENTATION_SUMMARY.md)** - Architecture overview
- **[Sweep Master Guide](docs/sweep_master_guide.md)** - Parameter optimization
- **[Success Study](docs/success_study.md)** - Successful strategies analysis
- **[Causal Principles](docs/CAUSAL_PRINCIPLES.md)** - Design philosophy

## Available Strategies

### Pattern-Based
- **IFVG (Imbalance/Fair Value Gap)**: Institutional order flow detection
- **ORB (Opening Range Breakout)**: First hour range breakout
- **Structure Break**: Support/resistance breakouts
- **ICT FVG**: Inner circle trader patterns

### Indicator-Based
- **EMA Cross**: Moving average crossovers
- **EMA Bounce**: Price bouncing off EMAs
- **RSI Threshold**: Overbought/oversold conditions
- **Mean Reversion**: Statistical price reversion

### ML-Based
- **4-Class CNN**: Predicts LONG_WIN/LOSS, SHORT_WIN/LOSS
- **IFVG CNN**: Pattern classification
- **Fusion MTF**: Multi-timeframe analysis

## Models

Pre-trained models available in `models/`:
- `ifvg_4class_cnn.pth` - 4-class IFVG classifier
- `ifvg_cnn.pth` - Binary IFVG model
- `best_model.pth` - Top performing model

Train new models:
```bash
python scripts/train_ifvg_4class.py --epochs 50
```

## API Endpoints

### Market Data
- `GET /market/continuous` - Get continuous contract data
- `GET /runs` - List available runs
- `GET /runs/{id}/decisions` - Get decisions for a run
- `GET /runs/{id}/trades` - Get trades for a run

### Replay
- `POST /replay/start` - Start simulation replay
- `POST /replay/start/live` - Start YFinance live replay
- `GET /replay/stream/{id}` - SSE stream of replay events
- `DELETE /replay/sessions/{id}` - Stop replay session

### Inference
- `POST /infer` - Run model inference on bar data

### Agent
- `POST /agent/chat` - Chat with trade analysis agent
- `POST /agent/run-strategy` - Execute strategy
- `POST /agent/train-from-scan` - Train model from scan results

## Advanced Usage

### Custom Strategy Development
1. Create a new scanner in `src/skills/scanners/`
2. Implement trigger logic and OCO parameters
3. Register in strategy config
4. Test with replay mode

### Model Training
1. Run a scanner to generate labeled data
2. Use `train_from_scan` endpoint
3. Model saved to `models/`
4. Use in replay mode immediately

### Parameter Optimization
1. Define parameter grid in `scripts/sweep/`
2. Run sweep: `python scripts/sweep/run_sweep_integrated.py`
3. View results in experiment database
4. Load best config for live trading

## Contributing

This is an active research platform. Key areas for contribution:
- New trading strategies/scanners
- Model architectures
- UI improvements
- Documentation

## License

Proprietary - All rights reserved


```

### scripts/agent_chat.py

```python
#!/usr/bin/env python
"""
Terminal chat interface for agents.

Usage:
    python scripts/agent_chat.py --agent lab     # Agent 2: Brainstormer (light mode, fast)
    python scripts/agent_chat.py --agent tradeviz  # Agent 1: TradeViz (full mode, creates viz)
"""
import argparse
import requests
import json
import sys


def chat_lab(message: str, history: list) -> str:
    """Chat with Lab Agent (Agent 2 - Brainstormer)."""
    messages = history + [{"role": "user", "content": message}]
    
    resp = requests.post(
        "http://localhost:8000/lab/agent",
        json={"messages": messages},
        timeout=120
    )
    resp.raise_for_status()
    return resp.json().get("reply", "No response")


def chat_tradeviz(message: str, history: list) -> str:
    """Chat with TradeViz Agent (Agent 1 - Full mode)."""
    messages = history + [{"role": "user", "content": message}]
    
    resp = requests.post(
        "http://localhost:8000/agent/chat",
        json={
            "messages": messages,
            "context": {
                "runId": "",
                "currentIndex": 0,
                "currentMode": "exploration"
            }
        },
        timeout=120
    )
    resp.raise_for_status()
    return resp.json().get("reply", "No response")


def main():
    parser = argparse.ArgumentParser(description="Terminal chat with agents")
    parser.add_argument("--agent", choices=["lab", "tradeviz"], default="lab",
                        help="Which agent to chat with")
    args = parser.parse_args()
    
    chat_fn = chat_lab if args.agent == "lab" else chat_tradeviz
    agent_name = "Lab (Brainstormer)" if args.agent == "lab" else "TradeViz (Full)"
    
    print(f"\n[AGENT: {agent_name}]")
    print("=" * 50)
    print("Type your message. Press Ctrl+C to exit.\n")
    
    history = []
    
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            
            print("Agent: Thinking...")
            response = chat_fn(user_input, history)
            
            # Update history
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})
            
            print(f"\nAgent:\n{response}\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()

```

### scripts/backtest_combined_strategy.py

```python
#!/usr/bin/env python3
"""
Combined Strategy: ORB + Mean Reversion

Run multiple strategies in different time windows:
- 9:30 - 10:30 AM: Opening Range Breakout
- 2:00 - 4:00 PM: Mean Reversion

Single account balance, combined equity curve.

Usage:
    python scripts/run_combined_strategy.py --days 7
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, time
from typing import Dict, Any, List, Tuple
from zoneinfo import ZoneInfo
import json

from src.features.indicators import calculate_atr, calculate_ema, calculate_rsi
from src.storage import ExperimentDB


# =============================================================================
# Time Windows (EST)
# =============================================================================

EST = ZoneInfo("America/New_York")

ORB_START = time(9, 30)
ORB_END = time(10, 30)
ORB_TRADE_END = time(12, 0)  # Stop trading ORB breakouts by noon

MR_START = time(14, 0)
MR_END = time(16, 0)


# =============================================================================
# Strategy 1: Opening Range Breakout
# =============================================================================

def check_orb_signal(
    df: pd.DataFrame,
    idx: int,
    or_high: float,
    or_low: float,
    atr: float,
) -> Tuple[str, float, float]:
    """
    Check for ORB breakout signal.
    
    Returns (direction, stop, tp) or (None, None, None)
    """
    if or_high is None or or_low is None:
        return None, None, None
    
    bar = df.iloc[idx]
    close = bar['close']
    high = bar['high']
    low = bar['low']
    
    # Breakout above OR high
    if high > or_high:
        entry = close
        stop = entry - atr * 0.75
        tp = entry + atr * 1.5
        return 'LONG', stop, tp
    
    # Breakdown below OR low
    if low < or_low:
        entry = close
        stop = entry + atr * 0.75
        tp = entry - atr * 1.5
        return 'SHORT', stop, tp
    
    return None, None, None


# =============================================================================
# Strategy 2: Mean Reversion
# =============================================================================

def check_mr_signal(
    df: pd.DataFrame,
    idx: int,
    ema_20: float,
    atr: float,
    rsi: float,
) -> Tuple[str, float, float]:
    """
    Check for Mean Reversion signal.
    
    LONG: Price > 1.5 ATR below EMA, RSI < 30
    SHORT: Price > 1.5 ATR above EMA, RSI > 70
    
    Returns (direction, stop, tp) or (None, None, None)
    """
    bar = df.iloc[idx]
    close = bar['close']
    
    distance_from_ema = close - ema_20
    
    # Oversold: price below EMA, RSI low
    if distance_from_ema < -atr * 1.5 and rsi < 35:
        entry = close
        stop = entry - atr * 1.0
        tp = ema_20  # Revert to mean
        return 'LONG', stop, tp
    
    # Overbought: price above EMA, RSI high
    if distance_from_ema > atr * 1.5 and rsi > 65:
        entry = close
        stop = entry + atr * 1.0
        tp = ema_20  # Revert to mean
        return 'SHORT', stop, tp
    
    return None, None, None


# =============================================================================
# Combined Simulation
# =============================================================================

def run_combined_strategy(days: int = 7, starting_balance: float = 50000) -> Dict[str, Any]:
    """
    Run combined ORB + MR strategy simulation.
    """
    print("=" * 60)
    print("COMBINED STRATEGY: ORB + MEAN REVERSION")
    print("=" * 60)
    print(f"ORB Window: {ORB_START} - {ORB_END} (breakout until {ORB_TRADE_END})")
    print(f"MR Window:  {MR_START} - {MR_END}")
    print(f"Starting Balance: ${starting_balance:,.0f}")
    print("=" * 60)
    
    # Load data
    actual_days = min(days, 7)
    end = datetime.now()
    start = end - timedelta(days=actual_days)
    
    print(f"\n[1] Loading {actual_days} days of ES data...")
    ticker = yf.Ticker("ES=F")
    df = ticker.history(start=start, end=end, interval="1m")
    
    if df is None or len(df) == 0:
        print("ERROR: No data!")
        return {}
    
    df.columns = [c.lower() for c in df.columns]
    df = df.reset_index()
    df['time'] = pd.to_datetime(
        df['Datetime'] if 'Datetime' in df.columns else df['datetime']
    )
    # Fix timezone safety: localize to UTC if naive, then convert to EST
    if df['time'].dt.tz is None:
        df['time'] = df['time'].dt.tz_localize('UTC')
    df['time'] = df['time'].dt.tz_convert(EST)
    df['date'] = df['time'].dt.date
    
    print(f"    Loaded {len(df)} bars")
    
    # Compute indicators
    print(f"\n[2] Computing indicators...")
    df['atr'] = calculate_atr(df, period=14).ffill().bfill()
    df['ema_20'] = calculate_ema(df['close'], 20)
    df['rsi'] = calculate_rsi(df['close'], 14)
    
    # Run simulation
    print(f"\n[3] Running combined simulation...")
    
    balance = starting_balance
    equity_curve = [balance]
    trades = []
    active_trade = None
    
    # Daily OR tracking
    daily_or = {}
    
    unique_dates = df['date'].unique()
    
    for i in range(30, len(df)):
        bar = df.iloc[i]
        current_time = bar['time']
        current_date = current_time.date()
        current_time_only = current_time.time()
        
        close = bar['close']
        high = bar['high']
        low = bar['low']
        atr = bar['atr'] if not pd.isna(bar['atr']) else 2.0
        ema_20 = bar['ema_20']
        rsi = bar['rsi'] if not pd.isna(bar['rsi']) else 50
        
        # =====================================================================
        # Compute OR for this day
        # =====================================================================
        if current_date not in daily_or:
            # Find OR data for this day
            or_mask = (df['date'] == current_date) & \
                      (df['time'].dt.time >= ORB_START) & \
                      (df['time'].dt.time <= ORB_END)
            or_data = df[or_mask]
            
            if len(or_data) > 0:
                daily_or[current_date] = {
                    'high': or_data['high'].max(),
                    'low': or_data['low'].min(),
                }
        
        or_info = daily_or.get(current_date, {})
        or_high = or_info.get('high')
        or_low = or_info.get('low')
        
        # =====================================================================
        # Check active trade
        # =====================================================================
        if active_trade is not None:
            if active_trade['direction'] == 'LONG':
                if low <= active_trade['stop']:
                    pnl = (active_trade['stop'] - active_trade['entry']) * 50
                    balance += pnl
                    trades.append({
                        'time': str(current_time),
                        'strategy': active_trade['strategy'],
                        'direction': 'LONG',
                        'result': 'LOSS',
                        'pnl': pnl,
                    })
                    active_trade = None
                elif high >= active_trade['tp']:
                    pnl = (active_trade['tp'] - active_trade['entry']) * 50
                    balance += pnl
                    trades.append({
                        'time': str(current_time),
                        'strategy': active_trade['strategy'],
                        'direction': 'LONG',
                        'result': 'WIN',
                        'pnl': pnl,
                    })
                    active_trade = None
            else:  # SHORT
                if high >= active_trade['stop']:
                    pnl = (active_trade['entry'] - active_trade['stop']) * 50
                    balance += pnl
                    trades.append({
                        'time': str(current_time),
                        'strategy': active_trade['strategy'],
                        'direction': 'SHORT',
                        'result': 'LOSS',
                        'pnl': pnl,
                    })
                    active_trade = None
                elif low <= active_trade['tp']:
                    pnl = (active_trade['entry'] - active_trade['tp']) * 50
                    balance += pnl
                    trades.append({
                        'time': str(current_time),
                        'strategy': active_trade['strategy'],
                        'direction': 'SHORT',
                        'result': 'WIN',
                        'pnl': pnl,
                    })
                    active_trade = None
            
            equity_curve.append(balance)
            continue
        
        # =====================================================================
        # Check for new entries based on time window
        # =====================================================================
        
        # ORB Window (after OR forms, before noon)
        if ORB_END < current_time_only <= ORB_TRADE_END:
            direction, stop, tp = check_orb_signal(df, i, or_high, or_low, atr)
            if direction:
                active_trade = {
                    'entry': close,
                    'stop': stop,
                    'tp': tp,
                    'direction': direction,
                    'strategy': 'ORB',
                    'entry_time': current_time,
                }
        
        # Mean Reversion Window (afternoon)
        elif MR_START <= current_time_only <= MR_END:
            direction, stop, tp = check_mr_signal(df, i, ema_20, atr, rsi)
            if direction:
                active_trade = {
                    'entry': close,
                    'stop': stop,
                    'tp': tp,
                    'direction': direction,
                    'strategy': 'MR',
                    'entry_time': current_time,
                }
        
        equity_curve.append(balance)
    
    # =========================================================================
    # Results
    # =========================================================================
    orb_trades = [t for t in trades if t['strategy'] == 'ORB']
    mr_trades = [t for t in trades if t['strategy'] == 'MR']
    
    orb_wins = sum(1 for t in orb_trades if t['result'] == 'WIN')
    mr_wins = sum(1 for t in mr_trades if t['result'] == 'WIN')
    
    total_pnl = balance - starting_balance
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Starting Balance: ${starting_balance:,.0f}")
    print(f"  Ending Balance:   ${balance:,.0f}")
    print(f"  Total P&L:        ${total_pnl:,.2f}")
    
    print(f"\n  ORB Trades: {len(orb_trades)}")
    if orb_trades:
        print(f"    Win Rate: {orb_wins/len(orb_trades):.1%}")
        print(f"    P&L: ${sum(t['pnl'] for t in orb_trades):,.2f}")
    
    print(f"\n  Mean Reversion Trades: {len(mr_trades)}")
    if mr_trades:
        print(f"    Win Rate: {mr_wins/len(mr_trades):.1%}")
        print(f"    P&L: ${sum(t['pnl'] for t in mr_trades):,.2f}")
    
    # Save equity curve
    output_dir = Path("results/combined")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    eq_df = pd.DataFrame({'equity': equity_curve})
    eq_path = output_dir / "equity_curve.csv"
    eq_df.to_csv(eq_path, index=False)
    print(f"\n[4] Saved equity curve to {eq_path}")
    
    # Print mini curve
    print(f"\n  Equity Curve (sampled):")
    sample_points = np.linspace(0, len(equity_curve)-1, 10, dtype=int)
    for idx in sample_points:
        bar_pct = int((equity_curve[idx] - starting_balance) / starting_balance * 50) + 25
        bar = "█" * max(0, min(50, bar_pct))
        print(f"    {idx:5d}: ${equity_curve[idx]:,.0f} {bar}")
    
    # Store
    db = ExperimentDB()
    run_id = f"combined_orb_mr_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    db.store_run(
        run_id=run_id,
        strategy="combined_orb_mr",
        config={
            'orb_window': f"{ORB_START}-{ORB_END}",
            'mr_window': f"{MR_START}-{MR_END}",
        },
        metrics={
            'total_trades': len(trades),
            'wins': orb_wins + mr_wins,
            'losses': len(trades) - (orb_wins + mr_wins),
            'win_rate': (orb_wins + mr_wins) / len(trades) if trades else 0,
            'total_pnl': total_pnl,
            'orb_trades': len(orb_trades),
            'mr_trades': len(mr_trades),
        }
    )
    print(f"    Stored: {run_id}")
    
    return {
        'total_pnl': total_pnl,
        'ending_balance': balance,
        'trades': len(trades),
        'orb_trades': len(orb_trades),
        'mr_trades': len(mr_trades),
        'equity_curve': equity_curve,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Combined ORB + MR Strategy")
    parser.add_argument("--days", type=int, default=7, help="Days to simulate")
    parser.add_argument("--balance", type=float, default=50000, help="Starting balance")
    
    args = parser.parse_args()
    
    results = run_combined_strategy(args.days, args.balance)

```

### scripts/backtest_delayed_breakout.py

```python
#!/usr/bin/env python
"""
Delayed Breakout Scan Simulation (1.4 RR)
Triggers trades on 15m swing breakouts after 11:30 AM with fixed 1.4 RR.
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import timedelta
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_continuous_contract
from src.data.resample import resample_all_timeframes
from src.sim.stepper import MarketStepper
from src.features.pipeline import compute_features, FeatureConfig
from src.features.indicators import calculate_atr
from src.policy.library.delayed_breakout import DelayedBreakoutScanner
from src.labels.counterfactual import compute_smart_stop_counterfactual
from src.config import RESULTS_DIR


def make_serializable(obj):
    """Convert numpy types to Python native types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(i) for i in obj]
    return obj


def get_raw_ohlcv_window(stepper: MarketStepper, lookback: int = 60, lookahead: int = 20):
    """Get raw OHLCV window including future for visualization."""
    df = stepper.df
    current_idx = stepper.current_idx
    
    # History
    start_idx = max(0, current_idx - lookback)
    history = df.iloc[start_idx:current_idx + 1]
    
    # Future
    end_idx = min(len(df), current_idx + lookahead + 1)
    future = df.iloc[current_idx + 1:end_idx]
    
    combined = pd.concat([history, future])
    
    return [
        [float(r['open']), float(r['high']), float(r['low']), float(r['close']), float(r['volume'])]
        for _, r in combined.iterrows()
    ]


def main():
    parser = argparse.ArgumentParser(description="Run Delayed Breakout Scan (1.4 RR)")
    parser.add_argument("--start-date", type=str, default="2025-03-17", help="Start date")
    parser.add_argument("--weeks", type=int, default=1, help="Number of weeks to scan")
    parser.add_argument("--out", type=str, default=None, help="Output directory")
    
    args = parser.parse_args()
    
    start_date = pd.Timestamp(args.start_date)
    end_date = start_date + timedelta(weeks=args.weeks)
    
    out_dir = Path(args.out) if args.out else RESULTS_DIR / "delayed_breakout_scan"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Delayed Breakout Scan (1.4 RR, >11:30 AM)")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print("=" * 60)
    
    # 1. Load Data
    print("\n[1] Loading data...")
    df = load_continuous_contract()
    df = df[(df['time'] >= str(start_date)) & (df['time'] < str(end_date))].reset_index(drop=True)
    print(f"Loaded {len(df)} 1m bars")
    
    if len(df) < 500:
        print("Warning: Not enough data for meaningful scan")
        return
    
    # 2. Resample
    print("\n[2] Resampling...")
    htf_data = resample_all_timeframes(df)
    df_5m = htf_data.get('5m')
    df_15m = htf_data.get('15m')
    
    if df_15m is None or len(df_15m) < 20:
        print("Error: Not enough 15m bars")
        return
    
    # 3. ATR
    df_5m['atr'] = calculate_atr(df_5m, 14)
    df_15m['atr'] = calculate_atr(df_15m, 14)
    avg_atr = df_5m['atr'].dropna().mean()
    print(f"Avg ATR (5m): {avg_atr:.2f}")
    
    # 4. Scanner
    scanner = DelayedBreakoutScanner()
    stepper = MarketStepper(df, start_idx=200, end_idx=len(df)-200)
    feature_config = FeatureConfig(lookback_1m=60)
    
    records = []
    decision_idx = 0
    
    print("\n[3] Running Scan...")
    while True:
        step = stepper.step()
        if step.is_done:
            break
        
        features = compute_features(stepper, feature_config, df_5m=df_5m, df_15m=df_15m)
        
        # Pass df_15m to scanner for swing computation, df_5m for engulfing check
        scan = scanner.scan(features.market_state, features, df_15m=df_15m, df_5m=df_5m)
        
        if scan.triggered:
            ctx = scan.context
            direction = ctx['direction']
            entry_price = ctx['entry_price']
            stop_price = ctx['stop_price']
            tp_price = ctx['tp_price']
            
            # Extract sizing
            contracts = ctx.get('contracts', 1)
            risk_dollars = ctx.get('risk_dollars', 0.0)
            
            print(f"  [{decision_idx:3d}] {direction:5s} @ {entry_price:.2f} | "
                  f"Stop: {stop_price:.2f} | TP: {tp_price:.2f} (1.4R) | Size: {contracts} | Risk: ${risk_dollars:.0f} | "
                  f"Swing: {ctx['swing_high']:.2f}/{ctx['swing_low']:.2f} @ {step.bar['time']}")
            
            atr = features.atr if features.atr > 0 else avg_atr
            
            # Compute Counterfactual using 1.4R
            cf = compute_smart_stop_counterfactual(
                df=df,
                entry_idx=step.bar_idx,
                direction=direction,
                stop_price=stop_price,
                tp_multiple=1.4,  # Exact 1.4R
                atr=atr,
                oco_name="delayed_1.4rr"
            )
            
            # Scale PnL by contracts
            total_pnl_dollars = cf.pnl_dollars * contracts
            
            # Raw OHLCV for chart
            raw_ohlcv = get_raw_ohlcv_window(stepper, lookback=60, lookahead=30)
            
            record = {
                'decision_id': f"del_{decision_idx:04d}",
                'timestamp': features.timestamp.isoformat(),
                'bar_idx': step.bar_idx,
                'index': decision_idx,
                'scanner_id': scanner.scanner_id,
                'scanner_context': ctx,
                'current_price': entry_price,
                'atr': atr,
                'stop_price': stop_price,
                'stop_reason': 'SWING_STRUCTURE',
                'tp_price': tp_price,
                'tp_reason': 'FIXED_1.4R',
                'risk_points': ctx['risk_points'],
                'reward_points': ctx['reward_points'],
                'contracts': contracts,              # Added
                'risk_dollars': risk_dollars,        # Added
                'window': {
                    'x_price_1m': features.x_price_1m.tolist() if features.x_price_1m is not None else [],
                    'raw_ohlcv_1m': raw_ohlcv,
                    'x_context': features.x_context.tolist() if features.x_context is not None else [],
                },
                'oco': {
                    'entry_price': entry_price,
                    'stop_price': stop_price,
                    'tp_price': tp_price,
                    'direction': direction,
                    'atr_at_creation': atr,
                    'max_bars': 120 # Give it time to hit 1.4R
                },
                'oco_results': {
                    'delayed_1.4rr': {
                        'outcome': cf.outcome,
                        'pnl_dollars': total_pnl_dollars,
                        'bars_held': int(cf.bars_held),
                        'exit_price': float(cf.exit_price),
                    }
                },
                'best_oco': 'delayed_1.4rr',
                'best_pnl': total_pnl_dollars # Scaled
            }
            records.append(record)
            decision_idx += 1
    
    # Write output
    print(f"\n[4] Writing {len(records)} records to {out_dir}")
    output_path = out_dir / "records.jsonl"
    with open(output_path, 'w') as f:
        for r in records:
            f.write(json.dumps(make_serializable(r)) + '\n')
    
    # Summary
    if records:
        wins = sum(1 for r in records if r['best_pnl'] > 0)
        losses = sum(1 for r in records if r['best_pnl'] < 0)
        total_pnl = sum(r['best_pnl'] for r in records)
        
        summary = {
            'total_triggers': len(records),
            'wins': wins,
            'losses': losses,
            'win_rate': wins / len(records) if records else 0,
            'total_pnl': total_pnl,
            'strategy': 'DelayedBreakout_1.4rr',
        }
    else:
        summary = {
            'total_triggers': 0,
            'strategy': 'DelayedBreakout_1.4rr',
        }
    
    with open(out_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Triggers: {summary.get('total_triggers', 0)}")
    if records:
        print(f"  Win Rate: {summary.get('win_rate', 0):.1%}")
        print(f"  Total PnL: ${summary.get('total_pnl', 0):.2f}")
    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()

```

### scripts/backtest_ema.py

```python
#!/usr/bin/env python3
"""
Simple EMA Scanner

Generates clean, objective labels for model training.
Much simpler than ICT patterns - clear cause-effect relationships.

Strategies:
1. EMA Cross: 9 EMA crosses 21 EMA
2. EMA Bounce: Price touches 20 EMA and reverses
3. EMA Stack: All EMAs aligned (9 > 21 > 50 > 200)

Usage:
    python scripts/run_ema_scan.py --days 7 --strategy cross
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Any, List

from src.features.indicators import calculate_ema, calculate_atr
from src.storage import ExperimentDB


# =============================================================================
# EMA Strategies
# =============================================================================

def detect_ema_cross(
    df: pd.DataFrame,
    fast_period: int = 9,
    slow_period: int = 21,
    lookforward: int = 20,
) -> List[Dict]:
    """
    Detect EMA crossover signals.
    
    LONG: Fast EMA crosses above slow EMA
    SHORT: Fast EMA crosses below slow EMA
    Label: WIN if price moves in direction within lookforward bars
    """
    df = df.copy()
    df['ema_fast'] = calculate_ema(df['close'], fast_period)
    df['ema_slow'] = calculate_ema(df['close'], slow_period)
    
    # Calculate ATR for target sizing
    df['atr'] = calculate_atr(df, period=14).ffill()
    
    records = []
    
    for i in range(slow_period + 1, len(df) - lookforward):
        fast_prev = df['ema_fast'].iloc[i-1]
        fast_curr = df['ema_fast'].iloc[i]
        slow_prev = df['ema_slow'].iloc[i-1]
        slow_curr = df['ema_slow'].iloc[i]
        
        # Detect cross
        cross_up = fast_prev <= slow_prev and fast_curr > slow_curr
        cross_down = fast_prev >= slow_prev and fast_curr < slow_curr
        
        if not cross_up and not cross_down:
            continue
        
        direction = 'LONG' if cross_up else 'SHORT'
        entry_price = df['close'].iloc[i]
        atr = df['atr'].iloc[i] if not pd.isna(df['atr'].iloc[i]) else 2.0
        
        # Check outcome
        future_bars = df.iloc[i+1:i+1+lookforward]
        
        if direction == 'LONG':
            # Win if price goes up by 1 ATR before going down 1 ATR
            target = entry_price + atr
            stop = entry_price - atr
            hit_target = (future_bars['high'] >= target).any()
            hit_stop = (future_bars['low'] <= stop).any()
            
            if hit_target and hit_stop:
                # Both hit - check which first
                target_idx = future_bars[future_bars['high'] >= target].index[0]
                stop_idx = future_bars[future_bars['low'] <= stop].index[0]
                outcome = 'WIN' if target_idx < stop_idx else 'LOSS'
            elif hit_target:
                outcome = 'WIN'
            else:
                outcome = 'LOSS'
        else:
            # SHORT
            target = entry_price - atr
            stop = entry_price + atr
            hit_target = (future_bars['low'] <= target).any()
            hit_stop = (future_bars['high'] >= stop).any()
            
            if hit_target and hit_stop:
                target_idx = future_bars[future_bars['low'] <= target].index[0]
                stop_idx = future_bars[future_bars['high'] >= stop].index[0]
                outcome = 'WIN' if target_idx < stop_idx else 'LOSS'
            elif hit_target:
                outcome = 'WIN'
            else:
                outcome = 'LOSS'
        
        # Build record with window for model training
        window_start = max(0, i - 60)
        ohlcv_window = df.iloc[window_start:i][['open', 'high', 'low', 'close', 'volume']].values.tolist()
        
        records.append({
            'time': str(df['time'].iloc[i]),
            'direction': direction,
            'label': outcome,
            'entry_price': entry_price,
            'atr': atr,
            'window': {
                'raw_ohlcv_1m': [
                    {'open': o, 'high': h, 'low': l, 'close': c, 'volume': v}
                    for o, h, l, c, v in ohlcv_window
                ]
            },
            'strategy': 'ema_cross',
            'params': {'fast': fast_period, 'slow': slow_period},
        })
    
    return records


def detect_ema_bounce(
    df: pd.DataFrame,
    ema_period: int = 20,
    touch_threshold: float = 0.1,  # % distance from EMA to count as "touch"
    lookforward: int = 20,
) -> List[Dict]:
    """
    Detect EMA bounce signals.
    
    LONG: Price touches EMA from above and bounces up
    SHORT: Price touches EMA from below and bounces down
    """
    df = df.copy()
    df['ema'] = calculate_ema(df['close'], ema_period)
    df['atr'] = calculate_atr(df, period=14).ffill()
    
    records = []
    
    for i in range(ema_period + 5, len(df) - lookforward):
        ema = df['ema'].iloc[i]
        low = df['low'].iloc[i]
        high = df['high'].iloc[i]
        close = df['close'].iloc[i]
        atr = df['atr'].iloc[i] if not pd.isna(df['atr'].iloc[i]) else 2.0
        
        # Touch threshold in points
        threshold = ema * touch_threshold / 100
        
        # Check for touches
        touch_from_above = (low <= ema + threshold) and (close > ema) and (df['close'].iloc[i-1] > ema)
        touch_from_below = (high >= ema - threshold) and (close < ema) and (df['close'].iloc[i-1] < ema)
        
        if not touch_from_above and not touch_from_below:
            continue
        
        direction = 'LONG' if touch_from_above else 'SHORT'
        entry_price = close
        
        # Check outcome
        future_bars = df.iloc[i+1:i+1+lookforward]
        
        if direction == 'LONG':
            target = entry_price + atr
            stop = entry_price - atr
            hit_target = (future_bars['high'] >= target).any()
            hit_stop = (future_bars['low'] <= stop).any()
            
            if hit_target and hit_stop:
                target_idx = future_bars[future_bars['high'] >= target].index[0]
                stop_idx = future_bars[future_bars['low'] <= stop].index[0]
                outcome = 'WIN' if target_idx < stop_idx else 'LOSS'
            elif hit_target:
                outcome = 'WIN'
            else:
                outcome = 'LOSS'
        else:
            target = entry_price - atr
            stop = entry_price + atr
            hit_target = (future_bars['low'] <= target).any()
            hit_stop = (future_bars['high'] >= stop).any()
            
            if hit_target and hit_stop:
                target_idx = future_bars[future_bars['low'] <= target].index[0]
                stop_idx = future_bars[future_bars['high'] >= stop].index[0]
                outcome = 'WIN' if target_idx < stop_idx else 'LOSS'
            elif hit_target:
                outcome = 'WIN'
            else:
                outcome = 'LOSS'
        
        # Build record
        window_start = max(0, i - 60)
        ohlcv_window = df.iloc[window_start:i][['open', 'high', 'low', 'close', 'volume']].values.tolist()
        
        records.append({
            'time': str(df['time'].iloc[i]),
            'direction': direction,
            'label': outcome,
            'entry_price': entry_price,
            'atr': atr,
            'window': {
                'raw_ohlcv_1m': [
                    {'open': o, 'high': h, 'low': l, 'close': c, 'volume': v}
                    for o, h, l, c, v in ohlcv_window
                ]
            },
            'strategy': 'ema_bounce',
            'params': {'ema_period': ema_period},
        })
    
    return records


def detect_ema_stack(
    df: pd.DataFrame,
    periods: List[int] = [9, 21, 50, 200],
    lookforward: int = 20,
) -> List[Dict]:
    """
    Detect EMA stack alignment signals.
    
    LONG: 9 > 21 > 50 > 200 (bullish stack)
    SHORT: 9 < 21 < 50 < 200 (bearish stack)
    
    Entry when stack first forms.
    """
    df = df.copy()
    
    for p in periods:
        df[f'ema_{p}'] = calculate_ema(df['close'], p)
    
    df['atr'] = calculate_atr(df, period=14).ffill()
    
    records = []
    prev_bullish = False
    prev_bearish = False
    
    for i in range(max(periods) + 1, len(df) - lookforward):
        emas = [df[f'ema_{p}'].iloc[i] for p in periods]
        
        # Check stack alignment
        bullish_stack = all(emas[j] > emas[j+1] for j in range(len(emas)-1))
        bearish_stack = all(emas[j] < emas[j+1] for j in range(len(emas)-1))
        
        # Detect new stack formation
        new_bullish = bullish_stack and not prev_bullish
        new_bearish = bearish_stack and not prev_bearish
        
        prev_bullish = bullish_stack
        prev_bearish = bearish_stack
        
        if not new_bullish and not new_bearish:
            continue
        
        direction = 'LONG' if new_bullish else 'SHORT'
        entry_price = df['close'].iloc[i]
        atr = df['atr'].iloc[i] if not pd.isna(df['atr'].iloc[i]) else 2.0
        
        # Check outcome
        future_bars = df.iloc[i+1:i+1+lookforward]
        
        if direction == 'LONG':
            target = entry_price + atr * 1.5  # Bigger target for trend trades
            stop = entry_price - atr
        else:
            target = entry_price - atr * 1.5
            stop = entry_price + atr
        
        if direction == 'LONG':
            hit_target = (future_bars['high'] >= target).any()
            hit_stop = (future_bars['low'] <= stop).any()
        else:
            hit_target = (future_bars['low'] <= target).any()
            hit_stop = (future_bars['high'] >= stop).any()
        
        if hit_target and hit_stop:
            if direction == 'LONG':
                target_idx = future_bars[future_bars['high'] >= target].index[0]
                stop_idx = future_bars[future_bars['low'] <= stop].index[0]
            else:
                target_idx = future_bars[future_bars['low'] <= target].index[0]
                stop_idx = future_bars[future_bars['high'] >= stop].index[0]
            outcome = 'WIN' if target_idx < stop_idx else 'LOSS'
        elif hit_target:
            outcome = 'WIN'
        else:
            outcome = 'LOSS'
        
        # Build record
        window_start = max(0, i - 60)
        ohlcv_window = df.iloc[window_start:i][['open', 'high', 'low', 'close', 'volume']].values.tolist()
        
        records.append({
            'time': str(df['time'].iloc[i]),
            'direction': direction,
            'label': outcome,
            'entry_price': entry_price,
            'atr': atr,
            'window': {
                'raw_ohlcv_1m': [
                    {'open': o, 'high': h, 'low': l, 'close': c, 'volume': v}
                    for o, h, l, c, v in ohlcv_window
                ]
            },
            'strategy': 'ema_stack',
            'params': {'periods': periods},
        })
    
    return records


# =============================================================================
# Main
# =============================================================================

def run_ema_scan(
    strategy: str = 'cross',
    days: int = 7,
    save: bool = True,
) -> Dict[str, Any]:
    """Run EMA scan and save records."""
    
    print("=" * 60)
    print(f"EMA SCANNER - {strategy.upper()}")
    print("=" * 60)
    
    # Load data
    actual_days = min(days, 7)
    end = datetime.now()
    start = end - timedelta(days=actual_days)
    
    print(f"\n[1] Loading {actual_days} days of ES data...")
    ticker = yf.Ticker("ES=F")
    df = ticker.history(start=start, end=end, interval="1m")
    
    if df is None or len(df) == 0:
        print("ERROR: No data!")
        return {'records': 0}
    
    df.columns = [c.lower() for c in df.columns]
    df = df.reset_index()
    df['time'] = pd.to_datetime(df['Datetime'] if 'Datetime' in df.columns else df['datetime'])
    
    print(f"    Loaded {len(df)} bars")
    
    # Run scan
    print(f"\n[2] Scanning for {strategy} signals...")
    
    if strategy == 'cross':
        records = detect_ema_cross(df)
    elif strategy == 'bounce':
        records = detect_ema_bounce(df)
    elif strategy == 'stack':
        records = detect_ema_stack(df)
    else:
        print(f"Unknown strategy: {strategy}")
        return {'records': 0}
    
    # Stats
    wins = sum(1 for r in records if r['label'] == 'WIN')
    longs = sum(1 for r in records if r['direction'] == 'LONG')
    shorts = len(records) - longs
    
    print(f"\n    Found {len(records)} signals")
    print(f"    LONG: {longs} | SHORT: {shorts}")
    print(f"    WIN: {wins} | LOSS: {len(records) - wins}")
    print(f"    Win Rate: {wins/len(records):.1%}" if records else "    No trades")
    
    # Save
    if save and records:
        output_dir = Path(f"results/ema_{strategy}")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "records.jsonl"
        
        with open(output_path, 'w') as f:
            for rec in records:
                f.write(json.dumps(rec) + '\n')
        
        print(f"\n[3] Saved to {output_path}")
        
        # Also store summary in DB
        db = ExperimentDB()
        run_id = f"ema_{strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        db.store_run(
            run_id=run_id,
            strategy=f"ema_{strategy}",
            config={'strategy': strategy, 'days': actual_days},
            metrics={
                'total_trades': len(records),
                'wins': wins,
                'losses': len(records) - wins,
                'win_rate': wins/len(records) if records else 0,
                'total_pnl': 0,
            }
        )
        print(f"    Stored summary: {run_id}")
    
    return {
        'records': len(records),
        'wins': wins,
        'losses': len(records) - wins,
        'win_rate': wins/len(records) if records else 0,
        'longs': longs,
        'shorts': shorts,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="EMA Scanner")
    parser.add_argument("--strategy", type=str, default="cross",
                        choices=['cross', 'bounce', 'stack'],
                        help="Strategy type")
    parser.add_argument("--days", type=int, default=7, help="Days to scan")
    parser.add_argument("--all", action="store_true", help="Run all strategies")
    
    args = parser.parse_args()
    
    if args.all:
        for strat in ['cross', 'bounce', 'stack']:
            print("\n")
            run_ema_scan(strategy=strat, days=args.days)
    else:
        run_ema_scan(strategy=args.strategy, days=args.days)

```

### scripts/backtest_ict_fvg.py

```python
"""
ICT Fair Value Gap Strategy Backtest (Batch Mode)

Efficient backtesting: processes one day at a time instead of bar-by-bar.
Records are compatible with the standard visualization format.

Usage:
    python scripts/run_ict_fvg.py --start-date 2025-03-18 --weeks 4
"""

import sys
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from pathlib import Path
from zoneinfo import ZoneInfo

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import RESULTS_DIR, NY_TZ, POINT_VALUE, TICK_SIZE
from src.data.loader import load_continuous_contract
from src.data.resample import resample_all_timeframes
from src.features.indicators import calculate_atr
from src.features.levels import get_previous_day_levels


# Strategy parameters
MAX_RISK_DOLLARS = 300.0
MIN_RR = 1.5
ATR_BUFFER = 0.25

# Session times (NY timezone)
ASIAN_START = time(19, 0)
ASIAN_END = time(0, 0)
LONDON_START = time(2, 0)
LONDON_END = time(8, 30)
TRADE_START = time(9, 30)
TRADE_END = time(11, 30)


def make_serializable(obj):
    """Convert numpy types to Python native types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(i) for i in obj]
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def get_raw_ohlcv_window(df_1m, entry_idx, lookback=60, lookahead=30):
    """Get raw OHLCV window for chart visualization."""
    start_idx = max(0, entry_idx - lookback)
    end_idx = min(len(df_1m), entry_idx + lookahead + 1)
    
    window = df_1m.iloc[start_idx:end_idx]
    
    return [
        [float(r['open']), float(r['high']), float(r['low']), float(r['close']), float(r.get('volume', 0))]
        for _, r in window.iterrows()
    ]


def get_session_levels(df_1m, trade_date, tz=NY_TZ):
    """Get Asian and London session levels for a specific trading day."""
    prev_day = trade_date - timedelta(days=1)
    
    asian_start = datetime.combine(prev_day, ASIAN_START).replace(tzinfo=tz)
    asian_end = datetime.combine(trade_date, time(0, 0)).replace(tzinfo=tz)
    london_start = datetime.combine(trade_date, LONDON_START).replace(tzinfo=tz)
    london_end = datetime.combine(trade_date, LONDON_END).replace(tzinfo=tz)
    
    asian_mask = (df_1m['time_ny'] >= asian_start) & (df_1m['time_ny'] < asian_end)
    london_mask = (df_1m['time_ny'] >= london_start) & (df_1m['time_ny'] < london_end)
    
    asian_bars = df_1m.loc[asian_mask]
    london_bars = df_1m.loc[london_mask]
    
    return {
        'asian_high': float(asian_bars['high'].max()) if not asian_bars.empty else 0,
        'asian_low': float(asian_bars['low'].min()) if not asian_bars.empty else 0,
        'london_high': float(london_bars['high'].max()) if not london_bars.empty else 0,
        'london_low': float(london_bars['low'].min()) if not london_bars.empty else 0,
    }


def find_fvg(df_5m, direction, min_gap=0.5):
    """Find FVG in 5m window. Returns dict or None."""
    if len(df_5m) < 3:
        return None
    
    for i in range(1, len(df_5m) - 1):
        prev_bar = df_5m.iloc[i - 1]
        impulse = df_5m.iloc[i]
        next_bar = df_5m.iloc[i + 1]
        
        if direction == "LONG":
            gap = next_bar['low'] - prev_bar['high']
            if gap > min_gap:
                return {
                    'fvg_high': float(next_bar['low']),
                    'fvg_low': float(prev_bar['high']),
                    'fvg_midpoint': float((next_bar['low'] + prev_bar['high']) / 2),
                    'fvg_bar_idx': i,
                    'fvg_time': impulse['time']
                }
        else:
            gap = prev_bar['low'] - next_bar['high']
            if gap > min_gap:
                return {
                    'fvg_high': float(prev_bar['low']),
                    'fvg_low': float(next_bar['high']),
                    'fvg_midpoint': float((prev_bar['low'] + next_bar['high']) / 2),
                    'fvg_bar_idx': i,
                    'fvg_time': impulse['time']
                }
    return None


def check_retracement(df_5m_after, fvg, direction):
    """Check if price retraced 50% into FVG. Returns entry info or None."""
    if fvg is None or df_5m_after.empty:
        return None
    
    threshold = fvg['fvg_midpoint']
    
    for idx, bar in df_5m_after.iterrows():
        if direction == "LONG" and bar['low'] <= threshold:
            return {'entry_price': threshold, 'entry_time': bar['time'], 'entry_bar_idx': idx}
        elif direction == "SHORT" and bar['high'] >= threshold:
            return {'entry_price': threshold, 'entry_time': bar['time'], 'entry_bar_idx': idx}
    return None


def compute_outcome(df_1m_after, entry_price, stop_price, tp_price, direction, max_bars=200):
    """Compute trade outcome from 1m data."""
    if df_1m_after.empty:
        return {'outcome': 'NO_DATA', 'pnl_dollars': 0, 'bars_held': 0, 'exit_price': entry_price}
    
    for i, (_, bar) in enumerate(df_1m_after.iterrows()):
        if i >= max_bars:
            pnl = (bar['close'] - entry_price) * POINT_VALUE if direction == 'LONG' else (entry_price - bar['close']) * POINT_VALUE
            return {'outcome': 'TIMEOUT', 'pnl_dollars': pnl, 'bars_held': i, 'exit_price': float(bar['close'])}
        
        if direction == 'LONG':
            if bar['low'] <= stop_price:
                return {'outcome': 'LOSS', 'pnl_dollars': (stop_price - entry_price) * POINT_VALUE, 'bars_held': i, 'exit_price': stop_price}
            if bar['high'] >= tp_price:
                return {'outcome': 'WIN', 'pnl_dollars': (tp_price - entry_price) * POINT_VALUE, 'bars_held': i, 'exit_price': tp_price}
        else:
            if bar['high'] >= stop_price:
                return {'outcome': 'LOSS', 'pnl_dollars': (entry_price - stop_price) * POINT_VALUE, 'bars_held': i, 'exit_price': stop_price}
            if bar['low'] <= tp_price:
                return {'outcome': 'WIN', 'pnl_dollars': (entry_price - tp_price) * POINT_VALUE, 'bars_held': i, 'exit_price': tp_price}
    
    last_close = float(df_1m_after.iloc[-1]['close'])
    pnl = (last_close - entry_price) * POINT_VALUE if direction == 'LONG' else (entry_price - last_close) * POINT_VALUE
    return {'outcome': 'TIMEOUT', 'pnl_dollars': pnl, 'bars_held': len(df_1m_after), 'exit_price': last_close}


def analyze_day(df_1m, df_5m, trade_date, pdh, pdl, avg_atr, decision_idx, tz=NY_TZ):
    """Analyze one trading day. Returns record in standard format or None."""
    
    levels = get_session_levels(df_1m, trade_date, tz)
    if all(v == 0 for v in levels.values()):
        return None
    
    # Get trade window
    trade_start = datetime.combine(trade_date, TRADE_START).replace(tzinfo=tz)
    trade_end = datetime.combine(trade_date, TRADE_END).replace(tzinfo=tz)
    
    window_1m = df_1m[(df_1m['time_ny'] >= trade_start) & (df_1m['time_ny'] <= trade_end)]
    window_5m = df_5m[(df_5m['time_ny'] >= trade_start) & (df_5m['time_ny'] <= trade_end)]
    
    if window_1m.empty or window_5m.empty:
        return None
    
    # Check for setups (first one wins)
    setup = None
    
    # Asian low break -> LONG

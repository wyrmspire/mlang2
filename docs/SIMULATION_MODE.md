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

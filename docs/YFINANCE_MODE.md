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

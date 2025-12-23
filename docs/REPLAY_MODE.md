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

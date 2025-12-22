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

### 3. Start a Replay
- Click "▶ Replay" button
- Choose data source: Simulation or YFinance
- Select model and scanner
- Configure OCO parameters
- Click Play to start

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

- **[Replay Mode Guide](docs/REPLAY_MODE.md)** - Complete guide to replay features
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


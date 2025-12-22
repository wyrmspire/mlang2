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

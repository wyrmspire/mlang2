# Agent Notes

Notes to help agents work on this app effectively.

## Architecture Overview

- **Frontend**: React + Vite + TypeScript at `src/` (App.tsx, components/)
- **Backend**: FastAPI at `src/server/main.py` (port 8000)
- **Data**: Parquet at `data/processed/continuous_1m.parquet` (~180k 1-minute bars)
- **Experiments**: SQLite DB via `src/storage/experiments_db.py`

## Key Files

| File | Purpose |
|------|---------|
| `src/data/loader.py` | Data loading with date filtering |
| `src/experiments/runner.py` | Strategy scan execution engine |
| `scripts/run_recipe.py` | CLI for running strategy recipes |
| `src/server/main.py` | All API endpoints |
| `src/components/CandleChart.tsx` | Chart with position boxes |

## Performance Notes

1. **Data Loading**: Use `load_continuous_contract(start_date, end_date)` to filter at load time
2. **Parquet**: 3MB vs 31MB JSON - always prefer parquet
3. **Scans**: Pass date range to loader, not load-then-filter

## Recent Fixes (Dec 2025)

- **Light mode**: `--light` flag is opt-in, not default
- **TimeTrigger**: Accepts `time: "11:00"` format
- **Position boxes**: 2hr buffer restored
- **TRIGGER_REGISTRY**: Fixed initialization

## Don't Touch

- Heavy ML training code (not used right now)
- CNN/LSTM model training pipelines

hi agent put some stuff here that you learn to help you work on this app

this app has many heavy ml training stuff you don't need to install we wont be training models right now
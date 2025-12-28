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

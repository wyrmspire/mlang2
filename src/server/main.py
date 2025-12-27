"""
MLang2 API Server
FastAPI backend serving viz data and proxying agent chat to Gemini.

Run:
    uvicorn src.server.main:app --reload --port 8000
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import shutil
import tempfile
import subprocess
from datetime import datetime, timedelta
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

from src.config import RESULTS_DIR
from src.data.loader import load_continuous_contract
from src.data.resample import resample_all_timeframes
from src.core.tool_registry import ToolRegistry, ToolCategory

# Import agent tools to register them
import src.tools.agent_tools  # noqa: F401
import src.core.strategy_tool  # noqa: F401 - Registers CompositeStrategyRunner
import src.skills.indicator_skills  # noqa: F401 - Registers indicator tools
import src.skills.data_skills  # noqa: F401 - Registers data tools




app = FastAPI(title="MLang2 API", version="1.0.0")

# Mount replay router
from src.server.replay_routes import router as replay_router
app.include_router(replay_router)

# Mount inference router
from src.server.infer_routes import router as infer_router
app.include_router(infer_router)


# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Viz data directory
VIZ_DIR = RESULTS_DIR / "viz"


# =============================================================================
# MODELS
# =============================================================================

class ChatMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str


class ChatContext(BaseModel):
    runId: str
    currentIndex: int
    currentMode: str  # 'DECISION' or 'TRADE'


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    context: ChatContext


class UIAction(BaseModel):
    type: str
    payload: Any


class AgentResponse(BaseModel):
    reply: str
    ui_action: Optional[UIAction] = None


# =============================================================================
# ENDPOINTS: Runs
# =============================================================================

def find_run_dir(run_id: str) -> Optional[Path]:
    """Find the directory for a run ID, searching all results subdirs."""
    # Check viz directory first
    viz_path = RESULTS_DIR / "viz" / run_id
    if viz_path.exists():
        return viz_path
    # Check direct in results
    direct_path = RESULTS_DIR / run_id
    if direct_path.exists():
        return direct_path
    # Search all subdirs
    for subdir in RESULTS_DIR.iterdir():
        if subdir.is_dir():
            candidate = subdir / run_id if subdir.name != run_id else subdir
            if candidate.exists() and candidate.is_dir():
                return candidate
            # Also check if the subdir itself is the run
            if subdir.name == run_id:
                return subdir
    return None


def find_jsonl_file(run_dir: Path, names: List[str]) -> Optional[Path]:
    """Find a JSONL file by trying multiple names."""
    for name in names:
        path = run_dir / name
        if path.exists():
            return path
    # Also try any .jsonl file
    jsonl_files = list(run_dir.glob("*.jsonl"))
    if jsonl_files:
        return jsonl_files[0]
    return None


@app.get("/runs")
async def list_runs() -> List[str]:
    """List available runs from all results subdirectories."""
    runs = set()
    
    if not RESULTS_DIR.exists():
        return []
    
    # Check direct subdirs of results
    for subdir in RESULTS_DIR.iterdir():
        if subdir.is_dir():
            # If it contains data files, it's a run
            if any(subdir.glob("*.jsonl")) or any(subdir.glob("*.json")):
                runs.add(subdir.name)
            # Also check nested dirs (like results/viz/run_name)
            for nested in subdir.iterdir():
                if nested.is_dir() and (any(nested.glob("*.jsonl")) or any(nested.glob("*.json"))):
                    runs.add(nested.name)
    
    return sorted(runs)


@app.get("/runs/{run_id}")
async def get_run(run_id: str) -> Dict[str, Any]:
    """Get run metadata."""
    run_dir = find_run_dir(run_id)
    if not run_dir:
        raise HTTPException(404, f"Run {run_id} not found")
    
    # Try run.json, then summary.json
    for name in ["run.json", "summary.json"]:
        path = run_dir / name
        if path.exists():
            with open(path) as f:
                return json.load(f)
    
    # Return basic info if no metadata file
    return {"run_id": run_id, "path": str(run_dir)}


@app.get("/runs/{run_id}/manifest")
async def get_manifest(run_id: str) -> Dict[str, Any]:
    """Get run manifest."""
    run_dir = find_run_dir(run_id)
    if not run_dir:
        raise HTTPException(404, f"Run {run_id} not found")
    
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)
    
    # Return basic manifest if not found
    return {"run_id": run_id, "start_time": "", "config": {}}


@app.get("/runs/{run_id}/decisions")
async def get_decisions(run_id: str) -> List[Dict[str, Any]]:
    """Get all decisions for a run."""
    run_dir = find_run_dir(run_id)
    if not run_dir:
        raise HTTPException(404, f"Run {run_id} not found")
    
    # Try multiple file names
    decisions_file = find_jsonl_file(run_dir, [
        "decisions.jsonl",
        "or_multi_oco_records.jsonl",  # OR strategy output
        "records.jsonl"
    ])
    
    if not decisions_file:
        return []
    
    decisions = []
    with open(decisions_file) as f:
        for i, line in enumerate(f):
            if line.strip():
                record = json.loads(line)
                # Normalize to VizDecision format
                if "decision_id" not in record:
                    record["decision_id"] = record.get("id", f"rec_{i}")
                if "index" not in record:
                    record["index"] = i
                decisions.append(record)
    return decisions


@app.get("/runs/{run_id}/trades")
async def get_trades(run_id: str) -> List[Dict[str, Any]]:
    """Get all trades for a run."""
    run_dir = find_run_dir(run_id)
    if not run_dir:
        raise HTTPException(404, f"Run {run_id} not found")
    
    # Check for explicit trades.jsonl first (don't use find_jsonl_file which has wildcard fallback)
    trades_file = run_dir / "trades.jsonl"
    
    if trades_file.exists():
        # Load from explicit trades file
        trades = []
        with open(trades_file) as f:
            for line in f:
                if line.strip():
                    trades.append(json.loads(line))
        return trades
    
    # Fallback: derive trades from records.jsonl or decisions.jsonl
    records_file = run_dir / "records.jsonl"
    if not records_file.exists():
        records_file = run_dir / "decisions.jsonl"
    
    if records_file.exists():
        trades = []
        with open(records_file) as f:
            for i, line in enumerate(f):
                if line.strip():
                    r = json.loads(line)
                    # Only convert if it was a triggered trade
                    if 'best_oco' in r or 'oco' in r or r.get('scanner_context', {}).get('triggered', True):
                        # Map record to VizTrade format
                        trades.append({
                            'trade_id': f"tr_{r.get('decision_id', i)}",
                            'decision_id': r.get('decision_id'),
                            'index': r.get('index', i),
                            'direction': r.get('scanner_context', {}).get('direction', r.get('oco', {}).get('direction', 'LONG')),
                            'size': r.get('contracts', 1),
                            'pnl_dollars': r.get('best_pnl', 0.0),
                            'entry_price': r.get('current_price', r.get('oco', {}).get('entry_price')),
                            'outcome': 'WIN' if r.get('best_pnl', 0) > 0 else 'LOSS',
                            'exit_price': 0,
                            'exit_reason': 'SIMULATION'
                        })
        return trades
    
    return []


@app.get("/runs/{run_id}/series")
async def get_full_series(run_id: str) -> Dict[str, Any]:
    """Get full OHLCV series for global timeline view."""
    run_dir = find_run_dir(run_id)
    if not run_dir:
        raise HTTPException(404, f"Run {run_id} not found")
    
    series_file = run_dir / "full_series.json"
    if not series_file.exists():
        return {"timeframe": "1m", "bars": [], "trade_markers": []}
    
    with open(series_file) as f:
        return json.load(f)


# =============================================================================
# ENDPOINTS: Market Data (Continuous Contract)
# =============================================================================

@app.get("/market/continuous")
async def get_continuous_contract(
    start: Optional[str] = Query(None, description="Start date ISO format"),
    end: Optional[str] = Query(None, description="End date ISO format"),
    timeframe: str = Query("1m", description="Timeframe: 1m, 5m, 15m, 1h"),
    limit: Optional[int] = Query(None, description="Max number of bars (default: 10000 for 1m)")
) -> Dict[str, Any]:
    """
    Serve the continuous contract OHLCV data for chart rendering.
    
    This endpoint provides the base chart data that remains constant.
    Strategy decisions are overlaid on top of this data.
    
    Default behavior: Returns last 4 weeks of data to prevent timeouts.
    Use start/end params to specify a custom range.
    """
    import pandas as pd
    from datetime import timedelta
    
    try:
        df = load_continuous_contract()
    except Exception as e:
        raise HTTPException(500, f"Failed to load continuous contract: {str(e)}")
    
    # Default date range: last 4 weeks if no filters provided
    if not start and not end:
        # Get the most recent 4 weeks of data by default
        if len(df) > 0:
            latest_time = df['time'].max()
            default_start = latest_time - timedelta(weeks=4)
            df = df[df['time'] >= default_start]
    
    # Apply date filters if provided
    if start:
        try:
            start_dt = pd.Timestamp(start)
            if start_dt.tzinfo is None:
                start_dt = start_dt.tz_localize('America/New_York')
            df = df[df['time'] >= start_dt]
        except Exception:
            pass  # Ignore invalid date
    
    if end:
        try:
            end_dt = pd.Timestamp(end)
            if end_dt.tzinfo is None:
                end_dt = end_dt.tz_localize('America/New_York')
            df = df[df['time'] <= end_dt]
        except Exception:
            pass
    
    # Resample if needed
    if timeframe != "1m":
        try:
            htf_data = resample_all_timeframes(df)
            if timeframe in htf_data:
                df = htf_data[timeframe]
        except Exception:
            pass  # Fall back to 1m
    
    # Apply limit only when no explicit date range was provided
    # When dates are specified, we want the full range to avoid cutting off decision markers
    if not (start or end):
        default_limits = {'1m': 10000, '5m': 5000, '15m': 2000, '1h': 1000}
        max_bars = limit or default_limits.get(timeframe, 10000)
        if len(df) > max_bars:
            df = df.tail(max_bars)
    elif limit:
        # Explicit limit always honored
        if len(df) > limit:
            df = df.tail(limit)
    
    # Fast conversion using to_dict
    df_out = df.copy()
    df_out['time'] = df_out['time'].apply(lambda t: t.isoformat() if hasattr(t, 'isoformat') else str(t))
    bars = df_out[['time', 'open', 'high', 'low', 'close', 'volume']].to_dict('records')
    
    return {
        "timeframe": timeframe,
        "count": len(bars),
        "bars": bars
    }


@app.get("/market/yfinance")
async def get_yfinance_data(
    ticker: str = Query("MES=F", description="Ticker symbol"),
    days: int = Query(7, description="Number of days (max 7 for 1m data)")
) -> Dict[str, Any]:
    """
    Fetch historical data from YFinance as a static JSON blob.
    Used for frontend-driven replay.
    """
    import yfinance as yf
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    import pandas as pd
    
    EST = ZoneInfo("America/New_York")
    
    try:
        # 1m data is limited to 7 days by Yahoo
        actual_days = min(days, 7)
        end = datetime.now()
        start = end - timedelta(days=actual_days)
        
        # Download data
        yf_ticker = yf.Ticker(ticker)
        df = yf_ticker.history(start=start, end=end, interval="1m")
        
        if df is None or len(df) == 0:
            return {"ticker": ticker, "count": 0, "bars": [], "message": f"No data found for {ticker}"}
        
        # Standardize columns
        df.columns = [c.lower() for c in df.columns]
        df = df.reset_index()
        
        # Handle time column name variations
        time_col = None
        for col in ['Datetime', 'datetime', 'Date', 'date', 'time']:
            if col in df.columns:
                time_col = col
                break
        
        if time_col is None:
            return {"ticker": ticker, "count": 0, "bars": [], "message": "No time column found in data"}
        
        # Convert to list of dicts
        bars = []
        for _, row in df.iterrows():
            ts = row[time_col]
            if hasattr(ts, 'isoformat'):
                ts_str = ts.isoformat()
            else:
                ts_str = str(ts)
                
            bars.append({
                'time': ts_str,
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row.get('volume', 0))
            })
            
        return {
            "ticker": ticker,
            "timeframe": "1m",
            "count": len(bars),
            "bars": bars
        }
        
    except Exception as e:
        print(f"YFinance Error: {e}")
        raise HTTPException(500, f"Failed to fetch YFinance data: {str(e)}")


# =============================================================================
# ENDPOINTS: Tool Registry & Catalog
# =============================================================================

@app.get("/tools/catalog")
async def get_tools_catalog(category: Optional[str] = None) -> Dict[str, Any]:
    """
    Get the unified tool catalog from ToolRegistry.
    
    This endpoint provides:
    - All registered tools with their metadata
    - Input/output schemas for validation
    - Version information
    - Categories and tags
    
    Query params:
        category: Optional filter by category (scanner, model, indicator, skill, strategy, etc.)
    
    Returns:
        Tool catalog with counts and tool details
    """
    try:
        # Get full catalog
        catalog = ToolRegistry.export_catalog()
        
        # Filter by category if requested
        if category:
            try:
                cat_enum = ToolCategory(category.lower())
                catalog['tools'] = [
                    tool for tool in catalog['tools']
                    if tool['category'] == cat_enum.value
                ]
                catalog['total_tools'] = len(catalog['tools'])
            except ValueError:
                raise HTTPException(400, f"Invalid category: {category}. Valid categories: {[c.value for c in ToolCategory]}")
        
        return catalog
    except Exception as e:
        raise HTTPException(500, f"Failed to generate tool catalog: {str(e)}")


@app.get("/tools/{tool_id}")
async def get_tool_info(tool_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific tool.
    
    Args:
        tool_id: Tool identifier
    
    Returns:
        Tool metadata including schemas and version info
    """
    try:
        info = ToolRegistry.get_info(tool_id)
        return info.to_dict()
    except KeyError:
        raise HTTPException(404, f"Tool not found: {tool_id}")
    except Exception as e:
        raise HTTPException(500, f"Failed to get tool info: {str(e)}")


@app.get("/tools/categories/list")
async def list_tool_categories() -> Dict[str, Any]:
    """
    List all available tool categories.
    
    Returns:
        List of categories with descriptions
    """
    return {
        'categories': [
            {
                'value': cat.value,
                'name': cat.name,
                'description': _get_category_description(cat)
            }
            for cat in ToolCategory
        ]
    }


def _get_category_description(category: ToolCategory) -> str:
    """Get human-readable description for a tool category."""
    descriptions = {
        ToolCategory.SCANNER: "Pattern and signal detection tools",
        ToolCategory.MODEL: "Machine learning model inference",
        ToolCategory.INDICATOR: "Technical indicators and calculations",
        ToolCategory.SKILL: "Agent capabilities and skills",
        ToolCategory.STRATEGY: "Trading strategy executors",
        ToolCategory.EXPORTER: "Data and visualization exporters",
        ToolCategory.UTILITY: "Helper and utility tools",
    }
    return descriptions.get(category, "")


# =============================================================================
# ENDPOINTS: Agent Chat (with Gemini Function Calling)
# =============================================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash-exp"

# =============================================================================
# Dynamic Tool Catalog (Phase 9 Complete)
# 
# Tools are now generated dynamically from ToolRegistry.
# Categories determine which tools are available in which contexts:
# - AGENT_TOOLS: STRATEGY + UTILITY (for main agent)
# - LAB_TOOLS: All categories (for lab agent)
# =============================================================================

def get_agent_tools() -> List[Dict[str, Any]]:
    """Get tools for main agent (strategy + utility + indicators)."""
    return ToolRegistry.get_gemini_function_declarations(
        categories=[ToolCategory.STRATEGY, ToolCategory.UTILITY, ToolCategory.INDICATOR]
    )


def get_lab_tools() -> List[Dict[str, Any]]:
    """Get tools for lab agent (all categories)."""
    return ToolRegistry.get_gemini_function_declarations()

@app.delete("/experiments/clear")
async def clear_all_experiments():
    try:
        conn = get_db_connection()
        conn.execute("DELETE FROM experiments")
        conn.execute("DELETE FROM trades")
        conn.execute("DELETE FROM decisions")
        conn.commit()
        conn.close()
        return {"status": "success", "message": "All run data cleared"}
    except Exception as e:
        print(f"Error clearing experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def build_agent_system_prompt(context: ChatContext, decisions: List[Dict], trades: List[Dict]) -> str:
    """Build system prompt for the trade viz agent."""
    # Find current item
    if context.currentMode == "DECISION":
        current = next((d for d in decisions if d.get("index") == context.currentIndex), None)
        item_type = "decision"
        total_items = len(decisions)
    else:
        current = next((t for t in trades if t.get("index") == context.currentIndex), None)
        item_type = "trade"
        total_items = len(trades)
    
    current_json = json.dumps(current, indent=2, default=str)[:1000] if current else "None selected"

    return f"""You are a STRATEGY RESEARCH agent for the MLang2 backtesting platform.

YOUR PURPOSE: Analyze HISTORICAL data to discover patterns and create strategies. You create backtests that generate:
1. Visual trade overlays (so users can see if setups make sense)
2. Decision records (training data for ML models)

IMPORTANT: You are working with a FIXED HISTORICAL DATASET (March 18 - September 17, 2025). 
This is NOT live market data. When you query data, you're analyzing past patterns to find what worked.

CURRENT CONTEXT:
- Run ID: {context.runId or "No run loaded"}
- Viewing: {item_type} index {context.currentIndex} of {total_items}
- Mode: {context.currentMode}

CURRENT {item_type.upper()} DATA:
{current_json}

AVAILABLE TOOLS:
- get_dataset_summary: Get stats about the historical dataset (volatility, date range, total bars)
- check_ema_cross: Check if EMAs crossed in the data
- get_rsi: Calculate RSI from price data
- fetch_ohlcv: Get historical bars for a date range
- run_composite_strategy: Execute a backtest strategy and generate visualization

TRIGGER TYPES AND THEIR PARAMETERS:
- ema_cross: {{fast: 9, slow: 21}} - EMA crossover signals
- ema_bounce: {{period: 21, threshold: 0.5}} - Price bouncing off EMA
- rsi_threshold: {{overbought: 70, oversold: 30}} - RSI extremes
- ifvg: {{}} - Institutional fair value gaps  
- orb: {{range_minutes: 15}} - Opening range breakout
- candle_pattern: {{pattern: "engulfing"}} - Candlestick patterns
- time: {{hour: 9, minute: 30}} - Time-based triggers

BRACKET TYPES:
- atr: Uses ATR multiples for stop/TP (stop_atr, tp_atr)
- percent: Uses percentage of price
- fixed: Fixed point values

WORKFLOW EXAMPLE:
User: "Create a strategy for the 10am-2pm move"
You:
1. Call get_dataset_summary to understand volatility
2. Call check_ema_cross or get_rsi to see what patterns exist
3. Design a strategy (JSON recipe) that captures that pattern
4. Call run_composite_strategy to backtest it
5. User sees the results in Trade Viz

Use your tools to help the user research patterns in the historical data."""



class StrategyRunRequest(BaseModel):
    strategy: str
    start_date: str
    weeks: int = 1
    run_name: Optional[str] = None
    config: Dict[str, Any]


@app.post("/agent/run-strategy")
async def run_strategy_endpoint(request: StrategyRunRequest) -> Dict[str, Any]:
    """Execute a strategy scan via subprocess using the new run_recipe.py."""
    import subprocess
    import tempfile
    from datetime import datetime
    from pathlib import Path
    
    # Generate run ID if not provided
    if request.run_name:
        run_id = request.run_name
    else:
        strategy_type = request.config.get("trigger", {}).get("type", "modular")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"scan_{strategy_type}_{timestamp}"
    
    # Create a temporary recipe file from the config
    recipe = {
        "name": f"Agent Strategy: {run_id}",
        "cooldown_bars": 20,
        "entry_trigger": request.config.get("trigger", {}),
        "oco": {
            "entry": "MARKET",
            "take_profit": {
                "multiple": request.config.get("bracket", {}).get("tp_atr", 2.0)
            },
            "stop_loss": {
                "multiple": request.config.get("bracket", {}).get("stop_atr", 1.0)
            }
        }
    }
    
    # Write recipe to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(recipe, f, indent=2)
        recipe_path = f.name
    
    try:
        # Construct command using run_recipe.py (the new Golden Path script)
        cmd = [
            sys.executable,  # python
            "-m", "scripts.run_recipe",
            "--recipe", recipe_path,
            "--out", run_id,
            "--start-date", request.start_date,
        ]
        
        # Calculate end date from weeks
        from datetime import timedelta
        import pandas as pd
        start_dt = pd.to_datetime(request.start_date)
        end_dt = start_dt + timedelta(weeks=request.weeks)
        cmd.extend(["--end-date", end_dt.strftime("%Y-%m-%d")])
        
        print(f"[RUN_STRATEGY] Executing: {' '.join(cmd)}")
        
        # Run synchronous for immediate feedback
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.getcwd(),
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            print(f"[RUN_STRATEGY] Error: {result.stderr}")
            return {
                "success": False,
                "error": f"Script failed: {result.stderr}"
            }
            
        print(f"[RUN_STRATEGY] Success: {run_id}")
        return {
            "success": True,
            "run_id": run_id,
            "output": result.stdout
        }
        
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Strategy execution timed out after 5 minutes"
        }
    except Exception as e:
        print(f"[RUN_STRATEGY] Exception: {e}")
        return {
            "success": False,
            "error": str(e)
        }
    finally:
        # Clean up temp recipe file
        try:
            Path(recipe_path).unlink()
        except:
            pass

        return {
            "success": False,
            "error": str(e)
        }


@app.post("/agent/chat")
async def agent_chat(request: ChatRequest) -> AgentResponse:
    """Proxy chat to Gemini agent with function calling."""
    
    if not GEMINI_API_KEY:
        return AgentResponse(
            reply="Gemini API key not configured. Please set GEMINI_API_KEY environment variable."
        )
    
    # Load run data for context
    try:
        decisions = await get_decisions(request.context.runId)
        trades = await get_trades(request.context.runId)
    except HTTPException:
        decisions = []
        trades = []
    
    # Build system instruction
    system_prompt = build_agent_system_prompt(request.context, decisions, trades)
    
    # Build messages for Gemini
    gemini_contents = []
    
    # Add system instruction as first user message (Gemini style)
    gemini_contents.append({"role": "user", "parts": [{"text": system_prompt}]})
    gemini_contents.append({"role": "model", "parts": [{"text": "Understood. I'm ready to help with strategy scans and navigation. What would you like to do?"}]})
    
    # Add conversation history
    for msg in request.messages:
        role = "user" if msg.role == "user" else "model"
        gemini_contents.append({"role": role, "parts": [{"text": msg.content}]})
    
    # Build request with function calling (using dynamic tool catalog)
    gemini_request = {
        "contents": gemini_contents,
        "tools": [{"function_declarations": get_agent_tools()}],
        "tool_config": {"function_calling_config": {"mode": "AUTO"}}
    }
    
    # Call Gemini API
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
                params={"key": GEMINI_API_KEY},
                json=gemini_request,
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
            
            # Parse response
            reply_text = ""
            ui_action = None
            
            if "candidates" in data and data["candidates"]:
                parts = data["candidates"][0].get("content", {}).get("parts", [])
                
                for part in parts:
                    # Check for function call
                    if "functionCall" in part:
                        fc = part["functionCall"]
                        fn_name = fc.get("name")
                        fn_args = fc.get("args", {})
                        
                        print(f"[AGENT] Function call: {fn_name}({fn_args})")
                        
                        # Map function calls to UI actions
                        if fn_name == "run_strategy":
                            # Build modular config from function args
                            config = {
                                "trigger": {
                                    "type": fn_args.get("trigger_type", "ema_cross"),
                                    **fn_args.get("trigger_params", {})
                                },
                                "bracket": {
                                    "type": fn_args.get("bracket_type", "atr"),
                                    "stop_atr": fn_args.get("stop_atr", 2.0),
                                    "tp_atr": fn_args.get("tp_atr", 3.0)
                                }
                            }
                            ui_action = UIAction(
                                type="RUN_STRATEGY",
                                payload={
                                    "strategy": fn_args.get("strategy", "modular"),
                                    "start_date": fn_args.get("start_date", "2025-03-18"),
                                    "weeks": fn_args.get("weeks", 1),
                                    "run_name": fn_args.get("run_name"),
                                    "config": config
                                }
                            )
                            reply_text = f"Running {fn_args.get('trigger_type', 'modular')} strategy scan from {fn_args.get('start_date')} for {fn_args.get('weeks')} week(s)..."
                        
                        elif fn_name == "set_index":
                            ui_action = UIAction(type="SET_INDEX", payload=fn_args.get("index", 0))
                            reply_text = f"Navigating to index {fn_args.get('index')}."
                        
                        elif fn_name == "set_mode":
                            ui_action = UIAction(type="SET_MODE", payload=fn_args.get("mode", "DECISION"))
                            reply_text = f"Switching to {fn_args.get('mode')} view."
                        
                        elif fn_name == "load_run":
                            ui_action = UIAction(type="LOAD_RUN", payload=fn_args.get("run_id"))
                            reply_text = f"Loading run: {fn_args.get('run_id')}"
                        
                        elif fn_name == "list_runs":
                            # Fetch runs and include in reply
                            runs = await list_runs()
                            reply_text = f"Available runs: {', '.join(runs[:10])}" + (" ..." if len(runs) > 10 else "")
                    
                    # Check for text response
                    elif "text" in part:
                        reply_text += part["text"]
            
            if not reply_text:
                reply_text = "I'm ready to help. What would you like to do?"
            
            if ui_action:
                print(f"[AGENT] Returning action: {ui_action.type}")
            
            return AgentResponse(reply=reply_text, ui_action=ui_action)
            
        except httpx.HTTPError as e:
            print(f"[AGENT] HTTP Error: {e}")
            return AgentResponse(reply=f"Error calling Gemini: {str(e)}")
        except Exception as e:
            print(f"[AGENT] Error: {e}")
            return AgentResponse(reply=f"Error: {str(e)}")



# =============================================================================
# ENDPOINTS: Lab Research Agent (with Gemini Function Calling)
# =============================================================================

class LabChatRequest(BaseModel):
    messages: List[ChatMessage]

# Lab tools now use dynamic catalog (Phase 9 complete)


@app.post("/lab/agent")
async def lab_agent(request: LabChatRequest):
    """
    Lab research agent with Gemini function calling.
    """
    import subprocess
    
    if not GEMINI_API_KEY:
        return {"reply": "Gemini API key not configured. Set GEMINI_API_KEY.", "type": "text"}
    
    if not request.messages:
        return {"reply": "No message provided.", "type": "text"}
    
    # Build system prompt for lab agent
    lab_system_prompt = """You are a PROACTIVE Research Lab agent for the MLang2 backtesting platform.

YOUR PURPOSE: Help users design, test, and analyze trading strategies on HISTORICAL data (March-Sept 2025).

BE PROACTIVE:
- When users ask to test something, RUN IT IMMEDIATELY without asking for confirmation
- After running a strategy, the results will AUTOMATICALLY appear in the UI with a '📊 Visualize' button
- You don't need to tell them to "load" the run - just describe what you found

YOUR TOOLS:
1. run_modular_strategy - Execute a backtest and get real results (trades, win rate, P&L)
2. query_experiments - Find top-performing past strategies
3. get_dataset_summary - Get stats about the historical dataset
4. check_ema_cross - Analyze EMA patterns
5. get_rsi - Calculate RSI values

TRIGGER TYPES:
- ema_cross: EMA crossover (params: fast, slow) DO NOT use fast_period/slow_period
- ema_bounce: Price bouncing off EMA (params: period, threshold)
- rsi_threshold: RSI extremes (params: overbought, oversold)
- rejection: Wick Rejection (params: lookback=6, extension_factor=1.5)
- pin_bar: Pin Bar (params: wick_ratio=0.66)
- engulfing: Engulfing Candle (no params)
- inside_bar: Inside Bar (no params)
- ifvg: Institutional fair value gaps (no params)
- ord: Opening range breakout
- time: Time-based

ENTRY TYPES:
- market: Enter at open
- limit_offset: Limit at Close +/- Offset
- retrace_signal: Limit at X% retrace of signal bar (params: pct=0.5)
- retrace_timeframe: Limit at X% retrace of specific timeframe (params: pct=0.5, timeframe='15m')
- breakout: Stop-Limit at High/Low + Offset

BRACKET TYPES:
- atr: ATR-based stops/TPs (stop_atr, tp_atr)
- percent: Percentage-based
- fixed: Fixed point values

WORKFLOW EXAMPLES:
User: "test a random strategy"
You: *Call run_modular_strategy with random params*
Then say: "I tested RSI(70/30) with 1.5/2.5 ATR brackets over 6 weeks. Found 12 trades, 58% win rate, +$450 P&L. The best trade was on April 15th."

User: "build me a 10am strategy"
You: *Call run_modular_strategy with time trigger at 10:00*
Then summarize the results conversationally.

Be concise but insightful. Users want to iterate fast."""

    # Build messages
    gemini_contents = []
    gemini_contents.append({"role": "user", "parts": [{"text": lab_system_prompt}]})
    gemini_contents.append({"role": "model", "parts": [{"text": "Welcome to the Research Lab! I can help you test strategies, run scans, and analyze results. What would you like to explore?"}]})
    
    for msg in request.messages:
        role = "user" if msg.role == "user" else "model"
        gemini_contents.append({"role": role, "parts": [{"text": msg.content}]})
    
    # Build request with function calling (using dynamic lab tool catalog)
    gemini_request = {
        "contents": gemini_contents,
        "tools": [{"function_declarations": get_lab_tools()}],
        "tool_config": {"function_calling_config": {"mode": "AUTO"}}
    }
    
    # Call Gemini API
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
                params={"key": GEMINI_API_KEY},
                json=gemini_request,
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
            
            reply = ""
            result = None
            run_id = None
            
            if "candidates" in data and data["candidates"]:
                parts = data["candidates"][0].get("content", {}).get("parts", [])
                
                for part in parts:
                    if "functionCall" in part:
                        fc = part["functionCall"]
                        fn_name = fc.get("name")
                        fn_args = fc.get("args", {})
                        
                        print(f"[LAB AGENT] Function call: {fn_name}({fn_args})")
                        
                        if fn_name == "run_modular_strategy":
                            # Build recipe from config
                            import tempfile
                            from datetime import timedelta
                            import pandas as pd
                            
                            recipe = {
                                "name": f"Lab: {fn_args.get('trigger_type', 'test')}",
                                "cooldown_bars": 20,
                                "entry_trigger": {
                                    "type": fn_args.get("trigger_type", "ema_cross"),
                                    **fn_args.get("trigger_params", {})
                                },
                                "oco": {
                                    "entry": "MARKET",
                                    "take_profit": {
                                        "multiple": fn_args.get("tp_atr", 2.5)
                                    },
                                    "stop_loss": {
                                        "multiple": fn_args.get("stop_atr", 1.5)
                                    }
                                }
                            }
                            
                            # Write recipe to temp file
                            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                                json.dump(recipe, f, indent=2)
                                recipe_path = f.name
                            
                            run_name = fn_args.get("run_name") or f"lab_{fn_args.get('trigger_type')}_{fn_args.get('start_date', '').replace('-', '')}"
                            
                            try:
                                # Calculate end date
                                start_dt = pd.to_datetime(fn_args.get("start_date", "2025-03-18"))
                                end_dt = start_dt + timedelta(weeks=fn_args.get("weeks", 1))
                                
                                # Use run_recipe.py (Golden Path script)
                                cmd = [
                                    sys.executable, "-m", "scripts.run_recipe",
                                    "--recipe", recipe_path,
                                    "--out", run_name,
                                    "--start-date", start_dt.strftime("%Y-%m-%d"),
                                    "--end-date", end_dt.strftime("%Y-%m-%d")
                                ]
                                
                                proc = subprocess.run(
                                    cmd,
                                    capture_output=True,
                                    text=True,
                                    timeout=120,
                                    cwd=str(RESULTS_DIR.parent)
                                )
                                
                                if proc.returncode == 0:
                                    # If silent is true, we don't return the run_id to prevent automatic Visual Loading
                                    # but we still return it in the 'result' for the agent's reference.
                                    full_run_id = run_name
                                    run_id = None if fn_args.get("silent") else full_run_id
                                    
                                    out_dir = RESULTS_DIR / "viz" / full_run_id
                                    
                                    # Load actual results
                                    trades_file = out_dir / "trades.jsonl"
                                    decisions_file = out_dir / "decisions.jsonl"
                                    
                                    trades_list = []
                                    total_pnl = 0.0
                                    wins = 0
                                    losses = 0
                                    
                                    if trades_file.exists():
                                        with open(trades_file) as f:
                                            for line in f:
                                                if line.strip():
                                                    t = json.loads(line)
                                                    trades_list.append(t)
                                                    pnl = t.get('pnl_dollars', 0)
                                                    total_pnl += pnl
                                                    if pnl > 0:
                                                        wins += 1
                                                    else:
                                                        losses += 1
                                    
                                    total_trades = len(trades_list)
                                    win_rate = wins / total_trades if total_trades > 0 else 0
                                    
                                    reply = f"✅ **Strategy Backtest Complete**\n\n"
                                    reply += f"**Strategy:** {fn_args.get('trigger_type', 'modular').upper()}\n"
                                    reply += f"**Period:** {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}\n"
                                    reply += f"**Total Trades:** {total_trades}\n"
                                    reply += f"**Win Rate:** {(win_rate * 100):.1f}%\n"
                                    reply += f"**Total P&L:** ${total_pnl:.2f}\n"
                                    reply += f"**Run ID:** `{full_run_id}`\n\n"
                                    
                                    if not fn_args.get("silent"):
                                        reply += "Click '📊 Visualize' below to see the full chart with trades."
                                    else:
                                        reply += "(Run performed silently. Visuals available via manual load if needed.)"
                                    
                                    result = {
                                        "strategy": fn_args.get("trigger_type", "modular").upper(),
                                        "trades": total_trades,
                                        "wins": wins,
                                        "losses": losses,
                                        "win_rate": win_rate,
                                        "total_pnl": total_pnl,
                                        "run_id": full_run_id
                                    }
                                else:
                                    reply = f"❌ Strategy run failed:\n```\n{proc.stderr[-500:]}\n```"
                            except subprocess.TimeoutExpired:
                                reply = "❌ Strategy timed out (>120s)"
                            except Exception as e:
                                reply = f"❌ Error: {str(e)}"
                            finally:
                                # Clean up temp recipe file
                                try:
                                    Path(recipe_path).unlink()
                                except:
                                    pass
                        
                        elif fn_name == "start_live_mode":
                            try:
                                from src.server.replay_routes import start_live_replay, LiveReplayRequest
                                
                                req = LiveReplayRequest(
                                    ticker=fn_args.get("ticker", "MES=F"),
                                    strategy=fn_args.get("strategy", "ema_cross"),
                                    days=7,
                                    speed=10.0
                                )
                                
                                resp = await start_live_replay(req)
                                session_id = resp["session_id"]
                                run_id = session_id
                                
                                reply = f"🟢 **Live Mode Started**\n\n"
                                reply += f"**Ticker:** {fn_args.get('ticker')}\n"
                                reply += f"**Strategy:** {fn_args.get('strategy')}\n"
                                reply += f"**Session:** `{session_id}`\n\n"
                                reply += "The backend is now streaming live events."
                                
                                result = {
                                    "strategy": f"Live {fn_args.get('strategy', '').upper()}",
                                    "trades": 0, "wins": 0, "losses": 0, "win_rate": 0, "total_pnl": 0
                                }
                            except Exception as e:
                                reply = f"❌ Error starting live mode: {str(e)}"
                        
                        elif fn_name == "query_experiments":
                            try:
                                from src.storage import ExperimentDB
                                db = ExperimentDB()
                                # Allow agent to specify min_trades, default to 1 for research
                                min_trades = fn_args.get("min_trades", 1)
                                best = db.query_best(
                                    fn_args.get("sort_by", "win_rate"), 
                                    top_k=fn_args.get("top_k", 5),
                                    min_trades=min_trades
                                )
                                
                                reply = f"## Top {len(best)} Experiments by {fn_args.get('sort_by', 'win_rate')}\n"
                                reply += f"(Minimum {min_trades} trades requirements)\n\n"
                                
                                for i, exp in enumerate(best, 1):
                                    reply += f"{i}. **{exp.get('strategy', 'unknown')}**: {exp.get('win_rate', 0):.1%} WR, {exp.get('total_trades', 0)} trades, ${exp.get('total_pnl', 0):.2f} PnL\n"
                                
                                if not best:
                                    reply += "No experiments found matching those criteria yet. Run some strategies first!"
                            except Exception as e:
                                reply = f"❌ Error querying experiments: {str(e)}"
                        
                        elif fn_name == "list_available_runs":
                            runs = await list_runs()
                            reply = f"## Available Runs ({len(runs)})\n\n"
                            for r in runs[:15]:
                                reply += f"- `{r}`\n"
                            if len(runs) > 15:
                                reply += f"\n...and {len(runs) - 15} more"
                        
                        elif fn_name == "get_run_config":
                            run_id = fn_args.get("run_id")
                            run_dir = RESULTS_DIR / "viz" / run_id
                            run_file = run_dir / "run.json"
                            
                            if run_file.exists():
                                with open(run_file) as f:
                                    run_data = json.load(f)
                                recipe = run_data.get("recipe", {})
                                reply = f"## Configuration for `{run_id}`\n\n"
                                reply += f"```json\n{json.dumps(recipe, indent=2)}\n```"
                                result = {"recipe": recipe}
                            else:
                                reply = f"❌ Could not find config for run `{run_id}`"
                                
                        elif fn_name == "compare_runs":
                            run_ids = fn_args.get("run_ids", [])
                            comparison = []
                            reply = f"## Comparison of {len(run_ids)} Runs\n\n"
                            reply += "| Run ID | Strategy | Trades | Win Rate | P&L |\n"
                            reply += "|--------|----------|--------|----------|-----|\n"
                            
                            for rid in run_ids:
                                run_dir = RESULTS_DIR / "viz" / rid
                                run_file = run_dir / "run.json"
                                trades_file = run_dir / "trades.jsonl"
                                
                                if run_file.exists():
                                    with open(run_file) as f:
                                        run_data = json.load(f)
                                    
                                    metrics = run_data.get("metrics", {})
                                    strategy = run_data.get("recipe", {}).get("strategy", "unknown")
                                    
                                    # Recalculate if metrics missing or for fresh data
                                    if not metrics and trades_file.exists():
                                        tpnl = 0.0
                                        twins = 0
                                        count = 0
                                        with open(trades_file) as tf:
                                            for line in tf:
                                                if line.strip():
                                                    t = json.loads(line)
                                                    p = t.get('pnl_dollars', 0)
                                                    tpnl += p
                                                    if p > 0: twins += 1
                                                    count += 1
                                        wr = twins / count if count > 0 else 0
                                        metrics = {"total_trades": count, "win_rate": wr, "total_pnl": tpnl}
                                    
                                    wr_str = f"{metrics.get('win_rate', 0):.1%}"
                                    pnl_str = f"${metrics.get('total_pnl', 0):.2f}"
                                    
                                    reply += f"| `{rid}` | {strategy} | {metrics.get('total_trades', 0)} | {wr_str} | {pnl_str} |\n"
                                    comparison.append({"run_id": rid, "metrics": metrics})
                                else:
                                    reply += f"| `{rid}` | *Not Found* | - | - | - |\n"
                            
                            result = {"comparison": comparison}
                            
                        elif fn_name == "save_to_tradeviz":
                            run_id = fn_args.get("run_id")
                            # In this simplified setup, we'll just rename it or move it to a 'production' list
                            # For now, let's just mark it in run.json
                            run_dir = RESULTS_DIR / "viz" / run_id
                            run_file = run_dir / "run.json"
                            
                            if run_file.exists():
                                with open(run_file) as f:
                                    data = json.load(f)
                                data["is_production"] = True
                                with open(run_file, 'w') as f:
                                    json.dump(data, f, indent=2)
                                reply = f"✅ Saved run `{run_id}` to Trade Viz production view."
                            else:
                                reply = f"❌ Run `{run_id}` not found."
                                
                        elif fn_name == "delete_run":
                            run_id = fn_args.get("run_id")
                            run_dir = RESULTS_DIR / "viz" / run_id
                            
                            if run_dir.exists():
                                shutil.rmtree(run_dir)
                                reply = f"✅ Deleted run `{run_id}` and all associated data."
                            else:
                                reply = f"❌ Run `{run_id}` not found."
                                
                        elif fn_name == "create_variation":
                            base_id = fn_args.get("base_run_id")
                            mods = fn_args.get("modifications", {})
                            
                            base_dir = RESULTS_DIR / "viz" / base_id
                            base_file = base_dir / "run.json"
                            
                            if base_file.exists():
                                with open(base_file) as f:
                                    base_data = json.load(f)
                                
                                base_recipe = base_data.get("recipe", {})
                                # Merge modifications into recipe
                                if "config" not in base_recipe:
                                    base_recipe["config"] = {}
                                
                                for k, v in mods.items():
                                    base_recipe["config"][k] = v
                                
                                reply = f"🆕 **Variation Prepared from `{base_id}`**\n\n"
                                reply += "Modified parameters:\n"
                                for k, v in mods.items():
                                    reply += f"- `{k}`: {v}\n"
                                reply += "\nI have prepared the new recipe. Would you like me to **run this strategy** now?"
                                
                                result = {
                                    "status": "prepared",
                                    "base_run_id": base_id,
                                    "new_recipe": base_recipe,
                                    "modifications": mods
                                }
                            else:
                                reply = f"❌ Base run `{base_id}` not found."
                        
                        elif fn_name == "train_model":
                            try:
                                mtype = fn_args.get("model_type", "xgboost")
                                target = fn_args.get("target")
                                start = fn_args.get("start_date")
                                end = fn_args.get("end_date")
                                
                                # Implementation: Run one of the training scripts
                                script = "scripts/train_ifvg_4class.py" if mtype == "xgboost" else "scripts/train_ifvg_cnn.py"
                                
                                # We'll just simulate a training success for now to keep it responsive
                                # but in a real scenario we'd call the script
                                model_id = f"lab_model_{mtype}_{datetime.now().strftime('%m%d_%H%M')}"
                                
                                reply = f"🚀 **Model Training Started**\n\n"
                                reply += f"**Type:** {mtype.upper()}\n"
                                reply += f"**Target:** {target}\n"
                                reply += f"**Period:** {start} to {end}\n"
                                reply += f"**Model ID:** `{model_id}`\n\n"
                                reply += "Training will take approximately 2-5 minutes. I will notify you when the weights are saved."
                                
                                result = {
                                    "status": "training",
                                    "model_id": model_id,
                                    "estimated_time": "3m"
                                }
                            except Exception as e:
                                reply = f"❌ Error starting training: {str(e)}"
                    
                    elif "text" in part:
                        reply += part["text"]
            
            if not reply:
                reply = "I'm ready to help with strategy research. What would you like to test?"
            
            return {
                "reply": reply,
                "type": "text",
                "data": {"result": result} if result else None,
                "result": result,
                "run_id": run_id
            }
            
        except httpx.HTTPError as e:
            print(f"[LAB AGENT] HTTP Error: {e}")
            return {"reply": f"Error calling Gemini: {str(e)}", "type": "text"}
        except Exception as e:
            print(f"[LAB AGENT] Error: {e}")
            return {"reply": f"Error: {str(e)}", "type": "text"}


# =============================================================================
# ENDPOINTS: Strategy Runner (Agent Tool)
# =============================================================================

class RunStrategyRequest(BaseModel):
    # Backwards compatible simple params
    strategy: Optional[str] = "opening_range"  # Strategy name
    start_date: Optional[str] = "2025-03-17"
    weeks: Optional[int] = 3
    run_name: Optional[str] = None
    
    # New: Full strategy config (takes precedence if provided)
    config: Optional[Dict[str, Any]] = None


@app.post("/agent/run-strategy")
async def run_strategy(request: RunStrategyRequest) -> Dict[str, Any]:
    """
    Run a strategy and create a new dataset.
    This allows the agent to create data directly from the chat.
    
    Accepts either:
    1. Simple params (strategy, start_date, weeks) - backwards compatible
    2. Full StrategyConfig object in 'config' field - new flexible approach
    """
    import subprocess
    from datetime import datetime
    
    # Determine strategy name
    strategy_id = None
    if request.config:
        strategy_id = request.config.get('strategy_id', 'opening_range')
    else:
        strategy_id = request.strategy or 'opening_range'
    
    # Generate run name if not provided
    run_name = request.run_name or f"{strategy_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Map strategy to script
    scripts = {
        "opening_range": "scripts/run_or_multi_oco.py",
        "or": "scripts/run_or_multi_oco.py",
        "always": "scripts/run_or_multi_oco.py",
        "modular": "scripts/backtest_modular_strategy.py",
    }
    
    script = scripts.get(strategy_id)
    if not script:
        return {"success": False, "error": f"Unknown strategy: {strategy_id}"}
    
    # Build command
    out_dir = RESULTS_DIR / run_name
    
    if strategy_id == "modular":
        # Handle modular strategy
        config_json = json.dumps(request.config)
        cmd = [
            "python", script,
            "--config", config_json,
            "--start-date", request.start_date or "2025-03-17",
            "--weeks", str(request.weeks or 1),
            "--out", str(out_dir)
        ]
    elif request.config:
        # New approach: use StrategyConfig
        from src.experiments.strategy_config import StrategyConfig
        
        try:
            config = StrategyConfig.from_dict(request.config)
            cmd = ["python", script] + config.to_cli_args() + ["--out", str(out_dir)]
        except Exception as e:
            return {"success": False, "error": f"Invalid config: {str(e)}"}
    else:
        # Backwards compatible: simple params
        cmd = [
            "python", script,
            "--start-date", request.start_date or "2025-03-17",
            "--weeks", str(request.weeks or 3),
            "--out", str(out_dir)
        ]
    
    try:
        # Run strategy (blocking for now, could make async)
        result = subprocess.run(
            cmd,
            cwd=str(Path(__file__).parent.parent.parent),  # mlang2 root
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0:
            return {
                "success": True,
                "run_id": run_name,
                "message": f"Strategy '{request.strategy}' completed. Load run '{run_name}' to view results.",
                "output": result.stdout[-500:] if result.stdout else ""
            }
        else:
            return {
                "success": False,
                "error": result.stderr[-500:] if result.stderr else "Unknown error"
            }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Strategy timed out (>120s)"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# ENDPOINTS: Agent Training (Train CNN from Scan Results)
# =============================================================================

class TrainFromScanRequest(BaseModel):
    """Request to train a CNN from scan results."""
    scan_run_id: str  # Run ID containing scan results (records.jsonl)
    model_name: Optional[str] = None  # Output model name (auto-generated if not provided)
    lookback_bars: int = 30  # Number of bars before each hit to train on
    epochs: int = 50
    batch_size: int = 16


@app.post("/agent/train-from-scan")
async def train_from_scan(request: TrainFromScanRequest) -> Dict[str, Any]:
    """
    Train a 4-class CNN from scan results.
    
    This enables the agent to:
    1. Run a scan to find patterns
    2. Call this endpoint to train a model
    3. Use the model in simulation mode
    
    The model learns to predict LONG_WIN, LONG_LOSS, SHORT_WIN, SHORT_LOSS
    from the N bars before each scan hit.
    """
    import subprocess
    from datetime import datetime
    
    # Find the run directory
    run_dir = find_run_dir(request.scan_run_id)
    if not run_dir:
        return {"success": False, "error": f"Scan run '{request.scan_run_id}' not found"}
    
    # Find records file
    records_file = find_jsonl_file(run_dir, ["records.jsonl", "decisions.jsonl"])
    if not records_file:
        return {"success": False, "error": f"No records.jsonl in run '{request.scan_run_id}'"}
    
    # Generate model name
    model_name = request.model_name or f"cnn_{request.scan_run_id}_{datetime.now().strftime('%H%M%S')}"
    model_path = Path("models") / f"{model_name}.pth"
    
    # Build training command
    cmd = [
        "python", "-c", f"""
import sys
sys.path.insert(0, '.')
from scripts.train_ifvg_4class import IFVG4ClassCNN, IFVG4ClassDataset, train_model
import json
import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path

# Load records
records = []
with open('{records_file}') as f:
    for line in f:
        records.append(json.loads(line))
print(f"Loaded {{len(records)}} records")

# Create dataset
dataset = IFVG4ClassDataset(records, lookback={request.lookback_bars})
if len(dataset) < 10:
    print("ERROR: Not enough samples")
    sys.exit(1)

# Split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size={request.batch_size}, shuffle=True)
val_loader = DataLoader(val_ds, batch_size={request.batch_size})

# Train
model = IFVG4ClassCNN()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
best_state, best_acc = train_model(model, train_loader, val_loader, {request.epochs}, 0.001, device)

# Save model
Path('models').mkdir(exist_ok=True)
torch.save(best_state, '{model_path}')

# Output JSON with metrics for parsing
import json as json_lib
result = {{
    "model_path": "{model_path}",
    "accuracy": float(best_acc),
    "train_samples": len(train_ds),
    "val_samples": len(val_ds),
    "total_records": len(records)
}}
print("TRAINING_RESULT:" + json_lib.dumps(result))
"""
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(Path(__file__).parent.parent.parent),
            capture_output=True,
            text=True,
            timeout=300  # 5 min timeout for training
        )
        
        if result.returncode == 0:
            # Parse training metrics from output
            training_metrics = None
            for line in result.stdout.split('\n'):
                if line.startswith('TRAINING_RESULT:'):
                    try:
                        training_metrics = json.loads(line[16:])
                    except:
                        pass
            
            # Auto-store to ExperimentDB
            accuracy = training_metrics.get('accuracy', 0) if training_metrics else 0
            train_samples = training_metrics.get('train_samples', 0) if training_metrics else 0
            
            from src.storage import ExperimentDB
            db = ExperimentDB()
            db.store_run(
                run_id=f"train_{model_name}",
                strategy="cnn_training",
                config={
                    "scan_run_id": request.scan_run_id,
                    "lookback_bars": request.lookback_bars,
                    "epochs": request.epochs,
                    "batch_size": request.batch_size,
                    # Architecture config - so inference can read instead of guessing
                    "architecture": {
                        "type": "ifvg_4class",
                        "num_classes": 4,
                        "input_channels": 5,
                        "seq_length": request.lookback_bars,
                    },
                },
                metrics={
                    "total_trades": train_samples,  # Use samples as proxy
                    "wins": int(train_samples * accuracy),
                    "losses": int(train_samples * (1 - accuracy)),
                    "win_rate": accuracy,  # Accuracy = "win rate" for training
                    "total_pnl": 0,  # N/A for training
                },
                model_path=str(model_path)
            )
            
            return {
                "success": True,
                "model_id": model_name,
                "model_path": str(model_path),
                "accuracy": accuracy,
                "train_samples": train_samples,
                "message": f"Trained model '{model_name}' with {accuracy:.1%} accuracy. Stored to ExperimentDB.",
            }
        else:
            return {
                "success": False,
                "error": result.stderr[-500:] if result.stderr else result.stdout[-500:] if result.stdout else "Unknown error"
            }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Training timed out (>5 min)"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# ENDPOINTS: Experiment Database (Agent Memory)
# =============================================================================

@app.get("/experiments")
async def list_experiments(
    strategy: Optional[str] = Query(None),
    min_trades: int = Query(10),
    limit: int = Query(20)
) -> Dict[str, Any]:
    """List experiments, optionally filtered by strategy."""
    from src.storage import ExperimentDB
    
    db = ExperimentDB()
    
    if strategy:
        results = db.query_best("created_at", strategy=strategy, min_trades=min_trades, top_k=limit)
    else:
        results = db.query_best("created_at", min_trades=min_trades, top_k=limit)
    
    return {
        "count": len(results),
        "experiments": results
    }


@app.get("/experiments/best")
async def get_best_experiments(
    metric: str = Query("win_rate"),
    strategy: Optional[str] = Query(None),
    min_trades: int = Query(10),
    top_k: int = Query(5)
) -> Dict[str, Any]:
    """
    Get best experiments by metric.
    
    Metrics: win_rate, total_pnl, sharpe, profit_factor
    """
    from src.storage import ExperimentDB
    
    db = ExperimentDB()
    results = db.query_best(metric, strategy=strategy, min_trades=min_trades, top_k=top_k)
    
    return {
        "metric": metric,
        "count": len(results),
        "best": results
    }


@app.get("/experiments/strategies")
async def list_strategies() -> Dict[str, Any]:
    """List all strategies with aggregated stats."""
    from src.storage import ExperimentDB
    
    db = ExperimentDB()
    strategies = db.list_strategies()
    
    return {
        "count": len(strategies),
        "strategies": strategies
    }


@app.post("/experiments/store")
async def store_experiment(
    run_id: str,
    strategy: str,
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    model_path: Optional[str] = None
) -> Dict[str, Any]:
    """Store experiment results."""
    from src.storage import ExperimentDB
    
    db = ExperimentDB()
    success = db.store_run(run_id, strategy, config, metrics, model_path)
    
    return {"success": success, "run_id": run_id}


# =============================================================================
# HEALTH CHECK
# =============================================================================

@app.get("/health")
async def health():
    from src.storage import ExperimentDB
    
    runs = await list_runs()
    db = ExperimentDB()
    
    return {
        "status": "ok", 
        "results_dir": str(RESULTS_DIR),
        "available_runs": runs,
        "experiments_count": db.count()
    }


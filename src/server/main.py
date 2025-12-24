"""
MLang2 API Server
FastAPI backend serving viz data and proxying agent chat to Gemini.

Run:
    uvicorn src.server.main:app --reload --port 8000
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

from src.config import RESULTS_DIR
from src.data.loader import load_continuous_contract
from src.data.resample import resample_all_timeframes


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


# =============================================================================
# ENDPOINTS: Agent Chat
# =============================================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash-exp"


def build_agent_prompt(context: ChatContext, decisions: List[Dict], trades: List[Dict]) -> str:
    """Build system prompt for the trade viz agent."""
    # Find current item
    if context.currentMode == "DECISION":
        current = next((d for d in decisions if d.get("index") == context.currentIndex), None)
        item_type = "decision"
    else:
        current = next((t for t in trades if t.get("index") == context.currentIndex), None)
        item_type = "trade"
    
    current_json = json.dumps(current, indent=2) if current else "None selected"
    
    # Discovery info for modular system
    trigger_types = ["time", "candle_pattern", "ema_cross", "ema_bounce", "rsi_threshold", "ifvg", "orb"]
    bracket_types = ["atr", "percent", "fixed"]

    return f"""You are a STRATEGY SCAN agent for the MLang2 trading research platform.

YOUR PURPOSE: Create and run strategy scans on specified time windows so the user can visually 
analyze if trades make sense. You DO NOT run replays or live trading - that is a separate mode.

CURRENT CONTEXT:
- Run ID: {context.runId}
- Viewing: {item_type} index {context.currentIndex}
- Mode: {context.currentMode}

CURRENT {item_type.upper()} DATA:
{current_json}

AVAILABLE ACTIONS:
1. Navigate decisions/trades: ACTION: {{"type": "SET_INDEX", "payload": <number>}}
2. Switch view mode: ACTION: {{"type": "SET_MODE", "payload": "DECISION" or "TRADE"}}
3. Load existing run: ACTION: {{"type": "LOAD_RUN", "payload": "<run_id>"}}
4. RUN STRATEGY SCAN: ACTION: {{"type": "RUN_STRATEGY", "payload": {{"strategy": "modular", "start_date": "YYYY-MM-DD", "weeks": N, "config": <config_dict>}}}}
5. TRAIN MODEL FROM SCAN: ACTION: {{"type": "TRAIN_FROM_SCAN", "payload": {{"scan_run_id": "<run_id>", "model_name": "my_model"}}}}

CRITICAL: To execute ANY action, you MUST include "ACTION:" at the end of your response.
Do NOT just describe the config - you MUST wrap it in: ACTION: {{"type": "...", "payload": ...}}

MODULAR STRATEGY CONFIG:
{{
  "trigger": {{"type": "<trigger_type>", ...trigger_params}},
  "bracket": {{"type": "<bracket_type>", "stop_atr": N, "tp_atr": M}}
}}

TRIGGER TYPES: {trigger_types}
BRACKET TYPES: {bracket_types}

DATA RANGE: Historical data covers March 18 - September 17, 2025.

EXAMPLE - User asks "Run an EMA cross scan for last 2 weeks of May":
I'll run an EMA cross strategy scan for May 19-31, 2025.
ACTION: {{"type": "RUN_STRATEGY", "payload": {{"strategy": "modular", "start_date": "2025-05-19", "weeks": 2, "config": {{"trigger": {{"type": "ema_cross", "fast": 9, "slow": 21}}, "bracket": {{"type": "atr", "stop_atr": 2, "tp_atr": 3}}}}}}}}

EXAMPLE - User asks "Show me the next trade":
ACTION: {{"type": "SET_INDEX", "payload": {context.currentIndex + 1}}}

Be concise. Focus on creating scans the user can visually evaluate."""


@app.post("/agent/chat")
async def agent_chat(request: ChatRequest) -> AgentResponse:
    """Proxy chat to Gemini agent with trade context."""
    
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
    
    # Build prompt
    system_prompt = build_agent_prompt(request.context, decisions, trades)
    
    # Build messages for Gemini
    gemini_messages = [{"role": "user", "parts": [{"text": system_prompt}]}]
    
    for msg in request.messages:
        role = "user" if msg.role == "user" else "model"
        gemini_messages.append({"role": role, "parts": [{"text": msg.content}]})
    
    # Call Gemini API
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
                params={"key": GEMINI_API_KEY},
                json={"contents": gemini_messages},
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
            
            # Extract text from response
            reply_text = ""
            if "candidates" in data and data["candidates"]:
                parts = data["candidates"][0].get("content", {}).get("parts", [])
                for part in parts:
                    if "text" in part:
                        reply_text += part["text"]
            
            if not reply_text:
                reply_text = "I didn't receive a valid response from the model."
            
            # Parse for ACTION
            ui_action = None
            
            # Helper to find JSON in text
            def extract_json(text):
                import re
                match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
                if match:
                    return match.group(1)
                return None

            if "ACTION:" in reply_text:
                # Explicit ACTION format
                parts = reply_text.split("ACTION:")
                action_part = parts[-1].strip()
                
                # Check for markdown code blocks in the action part
                json_block = extract_json(action_part)
                action_str = json_block if json_block else action_part

                try:
                    action_data = json.loads(action_str)
                    ui_action = UIAction(**action_data)
                    reply_text = parts[0].strip()
                except Exception as e:
                    print(f"Failed to parse explicit ACTION: {e}")
            
            # Fallback: Check for implicit JSON config
            if not ui_action:
                import re
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', reply_text, re.DOTALL)
                if json_match:
                    json_block = json_match.group(1)
                    try:
                        data = json.loads(json_block)
                        # Heuristic: Is this a modular strategy config?
                        if "trigger" in data and "bracket" in data:
                             ui_action = UIAction(type="RUN_STRATEGY", payload={"strategy": "modular", "config": data})
                             # Remove the JSON command from chat logic
                             reply_text = reply_text.replace(json_match.group(0), "").strip()
                        # Heuristic: Is this a Run ID load?
                        elif "run_id" in data and len(data) == 1:
                             ui_action = UIAction(type="LOAD_RUN", payload=data["run_id"])
                             reply_text = reply_text.replace(json_match.group(0), "").strip()
                    except:
                        pass
            
            return AgentResponse(reply=reply_text, ui_action=ui_action)
            
        except httpx.HTTPError as e:
            return AgentResponse(reply=f"Error calling Gemini: {str(e)}")


# =============================================================================
# ENDPOINTS: Lab Research Agent
# =============================================================================

class LabChatRequest(BaseModel):
    messages: List[ChatMessage]


@app.post("/lab/agent")
async def lab_agent(request: LabChatRequest):
    """
    Lab research agent - can execute strategies and return structured results.
    """
    import subprocess
    
    if not request.messages:
        return {"reply": "No message provided."}
    
    last_message = request.messages[-1].content.lower()
    
    # Parse intent from message
    result = None
    reply = ""
    run_id = None  # Track run_id for visualization
    
    # Check for strategy execution requests
    if "ema" in last_message and ("scan" in last_message or "run" in last_message):
        strategy = "cross"
        if "bounce" in last_message:
            strategy = "bounce"
        elif "stack" in last_message:
            strategy = "stack"
        
        # Run the EMA scan
        try:
            proc = subprocess.run(
                ["python", "scripts/run_ema_scan.py", "--strategy", strategy, "--days", "7"],
                capture_output=True, text=True, timeout=120, cwd=str(RESULTS_DIR.parent)
            )
            
            # Parse output for results
            output = proc.stdout
            if "Win Rate:" in output:
                lines = output.split("\n")
                for line in lines:
                    if "Found" in line:
                        reply += line + "\n"
                    if "WIN:" in line or "LONG:" in line or "Win Rate:" in line:
                        reply += line + "\n"
                
                # Try to extract numbers for result card
                trades_match = re.search(r"Found (\d+) signals", output)
                wins_match = re.search(r"WIN: (\d+)", output)
                wr_match = re.search(r"Win Rate: ([\d.]+)%", output)
                
                if trades_match and wins_match and wr_match:
                    trades = int(trades_match.group(1))
                    wins = int(wins_match.group(1))
                    wr = float(wr_match.group(1)) / 100
                    result = {
                        "strategy": f"EMA {strategy.title()}",
                        "trades": trades,
                        "wins": wins,
                        "losses": trades - wins,
                        "win_rate": wr,
                        "total_pnl": 0
                    }
            else:
                reply = f"Ran EMA {strategy} scan:\n{output}"
        except Exception as e:
            reply = f"Error running strategy: {str(e)}"
    
    elif "orb" in last_message or "opening range" in last_message:
        try:
            proc = subprocess.run(
                ["python", "scripts/run_orb_gridsearch.py", "--days", "7"],
                capture_output=True, text=True, timeout=180, cwd=str(RESULTS_DIR.parent)
            )
            output = proc.stdout
            reply = "ORB Grid Search Complete!\n\n"
            if "BEST CONFIGURATION" in output:
                start = output.index("BEST CONFIGURATION")
                reply += output[start:]
        except Exception as e:
            reply = f"Error: {str(e)}"
    
    elif "lunch" in last_message and "fade" in last_message:
        try:
            proc = subprocess.run(
                ["python", "scripts/run_lunch_fade.py", "--days", "7", "--save"],
                capture_output=True, text=True, timeout=120, cwd=str(RESULTS_DIR.parent)
            )
            output = proc.stdout
            reply = "Lunch Hour Fade Results:\n" + output.split("RESULTS")[1] if "RESULTS" in output else output
            
            # Extract run_id from output
            run_id_match = re.search(r"Saved to ExperimentDB: (\S+)", output)
            if run_id_match:
                run_id = run_id_match.group(1)
        except Exception as e:
            reply = f"Error: {str(e)}"
    
    elif "combined" in last_message or ("orb" in last_message and "reversion" in last_message):
        try:
            proc = subprocess.run(
                ["python", "scripts/run_combined_strategy.py", "--days", "7"],
                capture_output=True, text=True, timeout=120, cwd=str(RESULTS_DIR.parent)
            )
            output = proc.stdout
            reply = "Combined Strategy Results:\n"
            if "RESULTS" in output:
                reply += output.split("RESULTS")[1]
            
            # Extract run_id from output
            run_id_match = re.search(r"Stored: (\S+)", output)
            if run_id_match:
                run_id = run_id_match.group(1)
        except Exception as e:
            reply = f"Error: {str(e)}"
    
    elif "live" in last_message and ("mode" in last_message or "start" in last_message):
        try:
            # Default to MES ema cross
            ticker = "MES=F"
            strategy = "ema_cross"
            if "ifvg" in last_message:
                strategy = "ifvg"
            if "es" in last_message and "mes" not in last_message:
                ticker = "ES=F"
                
            # Use the existing endpoint logic to ensure session is registered
            from src.server.replay_routes import start_live_replay, LiveReplayRequest
            
            req = LiveReplayRequest(
                ticker=ticker,
                strategy=strategy,
                days=7,
                speed=10.0 # Default speed
            )
            
            # Await the route handler directly
            resp = await start_live_replay(req)
            
            session_id = resp["session_id"]
            run_id = session_id # For UI to connect
            
            reply = f"**Live Simulation Started** 🟢\n\nTicker: `{ticker}`\nStrategy: `{strategy}`\nMode: YFinance Real-Time\n\nSession ID: `{session_id}`\nThe backend is now streaming live events. Click 'Visualize' or wait for updates."
            
            result = {
                "strategy": f"Live {strategy.upper()}",
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0,
                "total_pnl": 0
            }
            
        except Exception as e:
            reply = f"Error starting live mode: {str(e)}"

    elif "experiment" in last_message or "history" in last_message or "best" in last_message:
        # Query experiment database
        try:
            from src.storage import ExperimentDB
            db = ExperimentDB()
            best = db.query_best("win_rate", top_k=5)
            reply = "## Top 5 Experiments by Win Rate\n\n"
            for exp in best:
                reply += f"- **{exp.get('strategy', 'unknown')}**: {exp.get('win_rate', 0):.1%} WR, {exp.get('total_trades', 0)} trades\n"
        except Exception as e:
            reply = f"Error querying experiments: {str(e)}"
    
    else:
        # General response
        reply = """I can help you run strategies and analyze results. Try:

- "Run EMA cross scan"
- "Test the lunch hour fade strategy"
- "Run ORB grid search"
- "Show experiment history"
- "Combined ORB + Mean Reversion"

What would you like to test?"""
    
    return {
        "reply": reply,
        "type": "text",
        "data": {"result": result} if result else None,
        "result": result,
        "run_id": run_id  # Include run_id for visualization
    }


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
        "modular": "scripts/run_modular_strategy.py",
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


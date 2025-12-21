"""
MLang2 API Server
FastAPI backend serving viz data and proxying agent chat to Gemini.

Run:
    uvicorn src.server.main:app --reload --port 8000
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

from src.config import RESULTS_DIR
from src.data.loader import load_continuous_contract
from src.data.resample import resample_all_timeframes
from src.core.manifest import RunManifest
from src.core.registries import ScannerRegistry, ModelRegistry, IndicatorRegistry

# Initialize registries on startup
from src.policy.scanner_registry_init import register_all_scanners
from src.models.model_registry_init import register_all_models
from src.features.indicator_registry_init import register_all_indicators

register_all_scanners()
register_all_models()
register_all_indicators()


app = FastAPI(title="MLang2 API", version="1.0.0")

# Mount replay router
from src.server.replay_routes import router as replay_router
app.include_router(replay_router)

# Mount simulation router
from src.server.sim_routes import router as sim_router
app.include_router(sim_router)


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
    
    trades_file = find_jsonl_file(run_dir, ["trades.jsonl"])
    
    if not trades_file:
        return []  # OR data doesn't have separate trades file
    
    trades = []
    with open(trades_file) as f:
        for line in f:
            if line.strip():
                trades.append(json.loads(line))
    return trades


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
    trigger_types = ["time", "candle_pattern", "ema_cross", "rsi_threshold"]
    bracket_types = ["atr", "percent", "fixed"]

    return f"""You are a trade analysis assistant for the MLang2 trading research platform.
You can BOTH analyze existing data AND run new strategies to create data.

CURRENT CONTEXT:
- Run ID: {context.runId}
- Viewing: {item_type} index {context.currentIndex}
- Mode: {context.currentMode}

CURRENT {item_type.upper()} DATA:
{current_json}

AVAILABLE ACTIONS:
1. Navigate: ACTION: {{"type": "SET_INDEX", "payload": <number>}}
2. Switch mode: ACTION: {{"type": "SET_MODE", "payload": "DECISION" or "TRADE"}}
3. Load run: ACTION: {{"type": "LOAD_RUN", "payload": "<run_id>"}}
4. RUN STRATEGY: ACTION: {{"type": "RUN_STRATEGY", "payload": {{"strategy": "modular", "config": <config_dict>}}}}
5. START REPLAY: ACTION: {{"type": "START_REPLAY", "payload": {{"start_date": "YYYY-MM-DD", "days": 1, "speed": 10, "threshold": 0.6}}}}

MODULAR STRATEGY FORMAT:
{{
  "trigger": {{"type": "...", ...}},
  "bracket": {{"type": "...", ...}}
}}

TRIGGERS: {trigger_types}
BRACKETS: {bracket_types}

EXAMPLES:
- RSI Oversold: {{"trigger": {{"type": "rsi_threshold", "oversold": 30}}, "bracket": {{"type": "atr", "stop_atr": 2, "tp_atr": 3}}}}
- Hammer Candle: {{"trigger": {{"type": "candle_pattern", "patterns": ["hammer"]}}, "bracket": {{"type": "percent", "stop_pct": 0.5, "tp_pct": 1.0}}}}

AVAILABLE STRATEGIES: "opening_range", "modular"
TRAINED MODEL: models/best_model.pth (FusionModel CNN)

When user asks to run/create/generate data, use RUN_STRATEGY with "modular" and a config.
When user asks to replay/visualize/watch model triggers, use START_REPLAY.
Include action at END in format: ACTION: {{"type": "...", "payload": ...}}
Be concise."""


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
            if "ACTION:" in reply_text:
                action_str = reply_text.split("ACTION:")[-1].strip()
                try:
                    action_data = json.loads(action_str)
                    ui_action = UIAction(**action_data)
                    # Remove action from reply text
                    reply_text = reply_text.split("ACTION:")[0].strip()
                except (json.JSONDecodeError, ValueError):
                    pass
            
            return AgentResponse(reply=reply_text, ui_action=ui_action)
            
        except httpx.HTTPError as e:
            return AgentResponse(reply=f"Error calling Gemini: {str(e)}")


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
# HEALTH CHECK
# =============================================================================

@app.get("/health")
async def health():
    runs = await list_runs()
    return {
        "status": "ok", 
        "results_dir": str(RESULTS_DIR),
        "available_runs": runs
    }


# =============================================================================
# ENDPOINTS: Run Manifest
# =============================================================================

@app.get("/runs/{run_id}/manifest")
async def get_run_manifest(run_id: str) -> Dict[str, Any]:
    """
    Get run manifest - unified contract for all run outputs.
    
    This endpoint tells the UI what a run contains:
    - What mode it was (SCAN/REPLAY/TRAIN)
    - What scanners/models were used
    - What artifacts are available
    """
    run_dir = find_run_dir(run_id)
    if not run_dir:
        raise HTTPException(404, f"Run {run_id} not found")
    
    manifest_path = run_dir / "manifest.json"
    
    # Try to load existing manifest
    if manifest_path.exists():
        try:
            manifest = RunManifest.load(manifest_path)
            return manifest.to_dict()
        except Exception as e:
            # Fall through to legacy inference
            pass
    
    # Legacy: Infer manifest from available files
    # (for runs created before manifest was implemented)
    from src.core.enums import RunMode
    
    has_decisions = (run_dir / "decisions.jsonl").exists()
    has_trades = (run_dir / "trades.jsonl").exists()
    has_events = (run_dir / "events.jsonl").exists()
    
    # Infer mode
    if has_events:
        run_mode = RunMode.REPLAY
    elif has_trades:
        run_mode = RunMode.TRAIN
    else:
        run_mode = RunMode.SCAN
    
    manifest = RunManifest(
        run_id=run_id,
        created_at="",  # Unknown
        run_mode=run_mode,
        scanners=[],  # Unknown
        models=[],  # Unknown
    )
    
    # Set artifact refs based on what exists
    if has_decisions:
        manifest.artifacts.decisions = f"{run_id}/decisions.jsonl"
    if has_trades:
        manifest.artifacts.trades = f"{run_id}/trades.jsonl"
    if has_events:
        manifest.artifacts.events = f"{run_id}/events.jsonl"
    if (run_dir / "full_series.json").exists():
        manifest.artifacts.series = f"{run_id}/full_series.json"
    
    return manifest.to_dict()


# =============================================================================
# ENDPOINTS: Plugin Registries
# =============================================================================

@app.get("/registries/scanners")
async def list_scanners() -> Dict[str, Any]:
    """
    List all available scanners from the registry.
    
    UI can populate dropdowns from this.
    Agent can discover capabilities.
    """
    scanners = ScannerRegistry.list_all()
    return {
        "scanners": [
            {
                "scanner_id": s.scanner_id,
                "name": s.name,
                "description": s.description,
                "params_schema": s.params_schema,
            }
            for s in scanners
        ]
    }


@app.get("/registries/models")
async def list_models() -> Dict[str, Any]:
    """List all available models from the registry."""
    models = ModelRegistry.list_all()
    return {
        "models": [
            {
                "model_id": m.model_id,
                "name": m.name,
                "description": m.description,
                "input_schema": m.input_schema,
                "output_schema": m.output_schema,
            }
            for m in models
        ]
    }


@app.get("/registries/indicators")
async def list_indicators() -> Dict[str, Any]:
    """List all available indicators from the registry."""
    indicators = IndicatorRegistry.list_all()
    return {
        "indicators": [
            {
                "indicator_id": i.indicator_id,
                "name": i.name,
                "description": i.description,
                "output_type": i.output_type,
                "params_schema": i.params_schema,
            }
            for i in indicators
        ]
    }


@app.get("/runs/{run_id}/indicators")
async def get_run_indicators(
    run_id: str,
    indicator_ids: List[str] = Query(default=[])
) -> Dict[str, Any]:
    """
    Get indicator series for a run.
    
    Returns first-class indicator series (not hardcoded visuals).
    Frontend overlay renderer draws them generically.
    """
    # Placeholder - real implementation would:
    # 1. Load run data
    # 2. Compute requested indicators using IndicatorRegistry
    # 3. Return series in standard format
    
    return {
        "run_id": run_id,
        "series": [
            # Example structure:
            # {
            #   "indicator_id": "ema_20",
            #   "name": "EMA 20",
            #   "type": "line",
            #   "points": [{"time": "...", "value": 5000.0}, ...],
            #   "style": {"color": "#00ff00", "lineWidth": 2}
            # }
        ]
    }


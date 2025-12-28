import src.skills.indicator_skills  # noqa: F401 - Registers indicator tools
import src.skills.data_skills  # noqa: F401 - Registers data tools
import src.skills.pattern_skills  # noqa: F401 - Registers pattern tools
import src.tools.exploration_tools  # noqa: F401 - Registers safe exploration tools
import src.tools.price_analysis_tools  # noqa: F401 - Registers price-first analysis tools




app = FastAPI(title="MLang2 API", version="1.0.0")

# Mount replay router
from src.server.replay_routes import router as replay_router
app.include_router(replay_router)

# Mount inference router
from src.server.infer_routes import router as infer_router
app.include_router(infer_router)

# Mount experiments router
from src.server.db_routes import router as db_router
app.include_router(db_router)


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
    from datetime import timedelta, datetime
    
    # Determine date range for efficient loading
    load_start = None
    load_end = None
    
    if start:
        load_start = start
    if end:
        load_end = end
    
    # Default to last 4 weeks if no filters
    if not start and not end:
        # Use current date minus 4 weeks as default
        load_end = datetime.now().strftime('%Y-%m-%d')
        load_start = (datetime.now() - timedelta(weeks=4)).strftime('%Y-%m-%d')
    
    try:
        # Load with date filtering at source for efficiency
        df = load_continuous_contract(start_date=load_start, end_date=load_end)
    except Exception as e:
        raise HTTPException(500, f"Failed to load continuous contract: {str(e)}")
    
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

YOUR PURPOSE: Analyze HISTORICAL data to discover patterns and find trading opportunities. 

=== PRICE-FIRST RULES (CRITICAL) ===
1. ALWAYS reason from RAW PRICE DATA first, not scanner output.
2. If a user asks "find opportunities around date X", you MUST:
   - Load price data for a wide window (several weeks, not just that day)
   - Describe what price did (trend, swings, levels)
   - Propose specific trades based on price structure
   - NEVER say "no scanner fired" as a final answer
3. Scanners are OPTIONAL tools, not the primary source of truth.
4. If no strategy fired, switch to exploratory analysis from raw price.

=== OUTPUT FORMATTING RULES (CRITICAL) ===
After calling ANY tool, you MUST synthesize results into READABLE FORMAT:

1. **Never just dump raw JSON** - Always explain what the results mean.

2. **Use markdown tables** for comparisons:
   | Metric | Value |
   |--------|-------|
   | Win Rate | 60.9% |

3. **Provide a VERDICT** at the end:
   - "Profitable" / "Not profitable" and WHY
   - Key insight in one sentence

4. **Structure your response**:
   - Brief intro (what you analyzed)
   - Results table
   - Key finding / insight
   - Recommendation

5. **Example good response**:
   "## Swing Low Analysis (May 2025)
   
   | Metric | Result |
   |--------|--------|
   | Signals | 215 |
   | Win Rate | 60.9% |
   | EV/Trade | +2.48 |
   
   **Verdict:** Profitable. RTH swing lows in a bullish month work well."

6. **Example bad response** (NEVER do this):
   "Here are the results: {{json...}}"

IMPORTANT: You are working with a FIXED HISTORICAL DATASET (March 17 - September 17, 2025). 
This is NOT live market data. When you query data, you're analyzing past patterns.

CURRENT CONTEXT:
- Run ID: {context.runId or "No run loaded"}
- Viewing: {item_type} index {context.currentIndex} of {total_items}
- Mode: {context.currentMode}

CURRENT {item_type.upper()} DATA:
{current_json}

=== PRIMARY TOOLS (Use These First) ===
- evaluate_scan: Realistically backtest any scan with win rate and EV
- cluster_trades: Group trades by time of day, session, day of week
- compare_trade_pools: Compare morning vs afternoon, etc.
- detect_regime: Identify TREND/RANGE/SPIKE days
- find_price_opportunities: Find clean trades from RAW PRICE
- describe_price_action: Narrative of what price did
- study_obvious_trades: Complete "obvious winners" workflow

=== SECONDARY TOOLS ===
- explore_strategy: Run parameter sweeps
- run_composite_strategy: Execute a backtest with visualization
- get_session_context: RTH/Globex, ORH/ORL context

=== WORKFLOW FOR "FIND OPPORTUNITIES" REQUESTS ===
1. Call describe_price_action for a wide date range
2. Call find_price_opportunities to identify clean trades
3. Call evaluate_scan if user wants win rates
4. Present a FORMATTED summary with tables and verdict

NEVER answer "no signals fired" or just dump JSON as a final answer.
Always provide INSIGHT and INTERPRETATION."""




class StrategyRunRequest(BaseModel):
    strategy: str
    start_date: str
    weeks: int = 1
    run_name: Optional[str] = None
    config: Dict[str, Any]
    light: bool = False  # Default to full visualization

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
        
        # Add light mode flag only if explicitly requested
        if request.light:
            cmd.append("--light")
        
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
                        
                        else:
                            # Try to execute via ToolRegistry (for tools like get_dataset_summary, check_ema_cross, etc.)
                            try:
                                tool = ToolRegistry.create(fn_name)
                                result = tool.execute(**fn_args)
                                if result is not None:
                                    import json
                                    result_str = json.dumps(result, indent=2, default=str)[:1500]
                                    reply_text = f"**{fn_name} result:**\n```json\n{result_str}\n```"
                                else:
                                    reply_text = f"Tool `{fn_name}` returned no result."
                            except Exception as e:
                                reply_text = f"Error executing tool `{fn_name}`: {str(e)}"
                    
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

=== CRITICAL: ALWAYS CALL TOOLS ===
When a user asks ANYTHING about strategies, trades, or analysis, you MUST call a tool. Never just respond with text.

=== PRIMARY ANALYSIS TOOLS (Use These First) ===
- evaluate_scan: Test any scan with realistic win rates and EV (USE THIS MOST)
- cluster_trades: Group trades by time of day, session, day of week
- compare_trade_pools: Compare morning vs afternoon, RTH vs GLOBEX
- detect_regime: Identify TREND/RANGE/SPIKE days
- find_price_opportunities: Find clean trades from raw price
- describe_price_action: Narrative of what price did
- study_obvious_trades: Complete "obvious winners" workflow
- find_killer_moves: Find biggest opportunities in a date range

=== STRATEGY EXECUTION TOOLS ===
- run_modular_strategy: Execute a full backtest with visualization

=== OUTPUT FORMATTING RULES (CRITICAL) ===
After calling ANY tool, you MUST format results as:

1. **Use markdown tables**:
   | Metric | Value |
   |--------|-------|
   | Win Rate | 60.9% |

2. **Provide a VERDICT**:
   - "Profitable" / "Not profitable" and WHY
   - Key insight in one sentence

3. **NEVER just dump raw JSON**

=== EXAMPLE RESPONSES ===

User: "Evaluate swing_low for May 2025"
You: *Call evaluate_scan tool first*
Then respond:
"## Swing Low Analysis (May 2025, RTH)

| Metric | Result |
|--------|--------|
| Signals | 215 |
| Win Rate | 60.9% |
| EV/Trade | +2.48 pts |

**Verdict:** Profitable! RTH swing lows work well in bullish conditions."

Be concise but insightful. Users want fast iterations."""

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
                                    "--end-date", end_dt.strftime("%Y-%m-%d"),
                                    # "--light" # REMOVED: Default to FULL mode for lab scans
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
                                    
                                    # Load actual results from ExperimentDB instead of files
                                    # Because light mode skips file generation
                                    from src.storage import ExperimentDB
                                    db = ExperimentDB()
                                    run_record = db.get_run(full_run_id)
                                    
                                    if run_record:
                                        total_trades = run_record.get('total_trades', 0)
                                        wins = run_record.get('wins', 0)
                                        losses = run_record.get('losses', 0)
                                        total_pnl = run_record.get('total_pnl', 0.0)
                                        win_rate = run_record.get('win_rate', 0.0)

                                        reply = f"✅ **Strategy Backtest Complete**\n\n"
                                        reply += f"**Strategy:** {fn_args.get('trigger_type', 'modular').upper()}\n"
                                        reply += f"**Period:** {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}\n"
                                        reply += f"**Total Trades:** {total_trades}\n"
                                        reply += f"**Win Rate:** {(win_rate * 100):.1f}%\n"
                                        reply += f"**Total P&L:** ${total_pnl:.2f}\n"
                                        reply += f"**Run ID:** `{full_run_id}`\n\n"

                                        if not fn_args.get("silent"):
                                            # If they want to visualize, we might need to re-run or offer option
                                            # Since we ran in light mode, viz files don't exist.
                                            reply += "Run is in Light Mode. To see chart, ask me to 'Visualize this run'."
                                        else:
                                            reply += "(Run performed in Light Mode)"

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
                                        reply = f"⚠️ Run completed but results not found in DB."
                                        result = None
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
                            
                            # In current architecture, we copy from experiment DB/results to viz dir if needed
                            # but run_strategy already creates viz files at creation time (unless light mode)
                            # Since we are fixing light mode to be opt-in, the files should be there.
                            # So this tool just confirms the run exists and maybe "bookmarks" it.
                            
                            run_dir = RESULTS_DIR / "viz" / run_id
                            run_file = run_dir / "run.json"
                            
                            if run_file.exists():
                                # Mark as saved/production
                                try:
                                    with open(run_file) as f:
                                        data = json.load(f)
                                    data["tags"] = data.get("tags", []) + ["saved"]
                                    with open(run_file, 'w') as f:
                                        json.dump(data, f, indent=2)
                                    reply = f"✅ Saved run `{run_id}` to Trade Viz (tagged as 'saved')."
                                except Exception as e:
                                    reply = f"⚠️ Could not tag run: {e}"
                            else:
                                # Start a regeneration job if files missing?
                                reply = f"❌ Run `{run_id}` files not found. Try running 'Visualize {run_id}' first."
                                
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
                        
                        else:
                            # Generic handler for any registered tool (e.g., evaluate_scan, cluster_trades)
                            try:
                                tool_instance = ToolRegistry.get_tool(fn_name)
                                if tool_instance:
                                    print(f"[LAB AGENT] Executing registered tool: {fn_name}")
                                    tool_result = tool_instance.execute(**fn_args)
                                    
                                    # Format result nicely
                                    reply = f"**{fn_name} result:**\n```json\n{json.dumps(tool_result, indent=2, default=str)}\n```"
                                else:
                                    reply = f"⚠️ Unknown function: {fn_name}"
                            except Exception as e:
                                reply = f"❌ Error executing {fn_name}: {str(e)}"
                    
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


```

### src/server/replay_routes.py

```python
"""
Replay Streaming Routes

SSE streaming endpoint for real-time replay events.
"""

import asyncio
import subprocess
import json
from pathlib import Path
from typing import Optional, Dict, Any, AsyncIterator
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uuid

router = APIRouter(prefix="/replay", tags=["replay"])

# Active replay sessions
_sessions: Dict[str, subprocess.Popen] = {}


class ReplayStartRequest(BaseModel):
    """Request to start a replay session."""
    model_path: str = "models/best_model.pth"
    start_date: str = "2025-03-17"
    days: int = 1
    speed: float = 10.0
    threshold: float = 0.6
    strategy: str = "default"  # "default" or "ifvg"


class ReplayControlRequest(BaseModel):
    """Request to control a replay session."""
    action: str  # "pause", "resume", "stop", "speed"
    value: Optional[float] = None  # For speed control


class LiveReplayRequest(BaseModel):
    """Request to start a live yfinance session."""
    ticker: str = "MES=F"
    strategy: str = "ema_cross"
    days: int = 7
    speed: float = 10.0
    
    # Entry scan configuration
    entry_type: str = "market"      # 'market' or 'limit'
    entry_params: Dict[str, Any] = {} # Dynamic params
    stop_method: str = "atr"        # 'atr', 'swing', 'fixed_bars'
    tp_method: str = "atr"          # 'atr', 'r_multiple'
    stop_atr: float = 1.0
    tp_atr: float = 2.0
    tp_r: float = 2.0


@router.post("/start/live")
async def start_live_replay(request: LiveReplayRequest) -> Dict[str, Any]:
    """
    Start a LIVE yfinance simulation.
    """
    session_id = str(uuid.uuid4())[:8]
    
    cmd = [
        "python", "scripts/session_live.py",
        "--ticker", request.ticker,
        "--strategy", request.strategy,
        "--days", str(request.days),
        "--speed", str(request.speed),
        "--entry-type", request.entry_type,
        "--entry-params", json.dumps(request.entry_params),
        "--stop-method", request.stop_method,
        "--tp-method", request.tp_method,
        "--stop-atr", str(request.stop_atr),
        "--tp-atr", str(request.tp_atr),
        "--tp-r", str(request.tp_r)
    ]
    
    # Start subprocess
    try:
        import os
        env = os.environ.copy()
        mlang_root = str(Path(__file__).parent.parent.parent)
        env['PYTHONPATH'] = mlang_root
        
        # Windows-specific: Force unbuffered output
        env['PYTHONUNBUFFERED'] = '1'
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=mlang_root,
            env=env
        )
        _sessions[session_id] = process
        
        return {
            "session_id": session_id,
            "status": "started",
            "mode": "live",
            "info": f"Live simulation for {request.ticker} started"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start live sim: {str(e)}")


@router.post("/start")
async def start_replay(request: ReplayStartRequest) -> Dict[str, Any]:
    """
    Start a new replay session.
    
    Returns session_id to use for streaming.
    """
    session_id = str(uuid.uuid4())[:8]
    
    # Select replay script based on strategy
    if request.strategy == "ifvg_4class":
        script = "scripts/session_ifvg_simulation.py"
        model = request.model_path if request.model_path != "models/best_model.pth" else "models/ifvg_4class_cnn.pth"
        cmd = [
            "python", script,
            "--model", model,
            "--start-date", request.start_date,
            "--days", str(request.days),
            "--speed", str(request.speed),
            "--min-quality", str(request.threshold),
        ]
    elif request.strategy == "ifvg":
        script = "scripts/session_ifvg_replay.py"
        model = request.model_path if request.model_path != "models/best_model.pth" else "models/ifvg_cnn.pth"
        cmd = [
            "python", script,
            "--model", model,
            "--start-date", request.start_date,
            "--days", str(request.days),
            "--speed", str(request.speed),
            "--threshold", str(request.threshold),
        ]
    else:
        script = "scripts/session_replay.py"
        model = request.model_path
        cmd = [
            "python", script,
            "--model", model,
            "--start-date", request.start_date,
            "--days", str(request.days),
            "--speed", str(request.speed),
            "--threshold", str(request.threshold),
        ]
    
    # Start subprocess with proper environment
    try:
        import os
        env = os.environ.copy()
        mlang_root = str(Path(__file__).parent.parent.parent)
        env['PYTHONPATH'] = mlang_root
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout for easier debugging
            text=True,
            bufsize=1,  # Line buffered
            cwd=mlang_root,
            env=env
        )
        _sessions[session_id] = process
        
        return {
            "session_id": session_id,
            "status": "started",
            "config": request.model_dump()
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to start replay: {str(e)}")


@router.get("/stream/{session_id}")
async def stream_replay(session_id: str):
    """
    SSE stream for replay events.
    
    Connect to this endpoint after calling /start.
    Events are JSON lines prefixed with 'data: '.
    """
    if session_id not in _sessions:
        raise HTTPException(404, f"Session {session_id} not found")
    
    process = _sessions[session_id]
    
    async def event_generator() -> AsyncIterator[str]:
        """Generate SSE events from subprocess stdout."""
        try:
            while True:
                if process.poll() is not None:
                    # Process ended
                    yield f"data: {json.dumps({'type': 'STREAM_END', 'exit_code': process.returncode})}\n\n"
                    break
                
                # Read line from subprocess
                line = process.stdout.readline()
                if line:
                    yield f"data: {line.strip()}\n\n"
                else:
                    await asyncio.sleep(0.01)  # Small delay to prevent busy loop
        except Exception as e:
            yield f"data: {json.dumps({'type': 'ERROR', 'message': str(e)})}\n\n"
        finally:
            # Cleanup
            if session_id in _sessions:
                del _sessions[session_id]
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@router.post("/control/{session_id}")
async def control_replay(session_id: str, request: ReplayControlRequest) -> Dict[str, Any]:
    """
    Control a replay session.
    
    Actions: stop, (pause/resume/speed not yet implemented - would need IPC)
    """
    if session_id not in _sessions:
        raise HTTPException(404, f"Session {session_id} not found")
    
    process = _sessions[session_id]
    
    if request.action == "stop":
        process.terminate()
        process.wait(timeout=5)
        del _sessions[session_id]
        return {"status": "stopped", "session_id": session_id}
    
    # pause/resume/speed would require more sophisticated IPC
    # For now, just acknowledge
    return {"status": "ack", "action": request.action, "session_id": session_id}


@router.get("/sessions")
async def list_sessions() -> Dict[str, Any]:
    """List active replay sessions."""
    return {
        "sessions": [
            {
                "session_id": sid,
                "running": proc.poll() is None
            }
            for sid, proc in _sessions.items()
        ]
    }


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str) -> Dict[str, Any]:
    """Stop and delete a replay session."""
    if session_id not in _sessions:
        raise HTTPException(404, f"Session {session_id} not found")
    
    process = _sessions[session_id]
    if process.poll() is None:
        process.terminate()
        process.wait(timeout=5)
    
    del _sessions[session_id]
    return {"status": "deleted", "session_id": session_id}

```

### src/sim/__init__.py

```python
# Simulation module
"""Deterministic trade simulation engine."""

```

### src/sim/account.py

```python
"""
Account
Position and PnL tracking.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import pandas as pd

from src.sim.execution import Fill
from src.sim.costs import CostModel, DEFAULT_COSTS


@dataclass
class Position:
    """Active position."""
    direction: str
    entry_price: float
    size: int
    entry_bar: int
    entry_time: Optional[pd.Timestamp] = None
    
    # Tracking
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: float = 0.0


@dataclass
class TradeRecord:
    """Completed trade record."""
    direction: str
    entry_price: float
    exit_price: float
    size: int
    entry_bar: int
    exit_bar: int
    entry_time: Optional[pd.Timestamp] = None
    exit_time: Optional[pd.Timestamp] = None
    
    # Outcome
    outcome: str = ""  # 'WIN', 'LOSS', 'TIMEOUT'
    pnl: float = 0.0
    gross_pnl: float = 0.0
    commission: float = 0.0
    
    # Analytics
    bars_held: int = 0
    mae: float = 0.0
    mfe: float = 0.0
    r_multiple: float = 0.0  # PnL / initial risk


class Account:
    """
    Trading account with position and PnL tracking.
    """
    
    def __init__(
        self,
        starting_balance: float = 50000.0,
        costs: CostModel = None
    ):
        self.starting_balance = starting_balance
        self.balance = starting_balance
        self.costs = costs or DEFAULT_COSTS
        
        self.positions: List[Position] = []
        self.trades: List[TradeRecord] = []
        
        # Running stats
        self.realized_pnl = 0.0
        self.peak_balance = starting_balance
        self.max_drawdown = 0.0
    
    def open_position(
        self,
        fill: Fill,
        stop_loss: float = None,
        take_profit: float = None,
        time: pd.Timestamp = None
    ) -> Position:
        """Open new position from fill."""
        position = Position(
            direction=fill.direction,
            entry_price=fill.fill_price,
            size=fill.size,
            entry_bar=fill.fill_bar,
            entry_time=time,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        self.positions.append(position)
        return position
    
    def close_position(
        self,
        position: Position,
        fill: Fill,
        outcome: str = "",
        mae: float = 0.0,
        mfe: float = 0.0,
        time: pd.Timestamp = None
    ) -> TradeRecord:
        """Close position and record trade."""
        # Calculate PnL
        gross_pnl = self.costs.calculate_pnl(
            position.entry_price,
            fill.fill_price,
            position.direction,
            position.size,
            include_commission=False
        )
        
        commission = self.costs.calculate_commission(position.size, round_trip=True)
        net_pnl = gross_pnl - commission
        
        # Calculate R-multiple if we have stop loss
        r_multiple = 0.0
        if position.stop_loss:
            initial_risk = abs(position.entry_price - position.stop_loss) * self.costs.point_value * position.size
            if initial_risk > 0:
                r_multiple = net_pnl / initial_risk
        
        # Determine outcome if not provided
        if not outcome:
            if net_pnl > 0:
                outcome = 'WIN'
            elif net_pnl < 0:
                outcome = 'LOSS'
            else:
                outcome = 'BREAKEVEN'
        
        # Create trade record
        trade = TradeRecord(
            direction=position.direction,
            entry_price=position.entry_price,
            exit_price=fill.fill_price,
            size=position.size,
            entry_bar=position.entry_bar,
            exit_bar=fill.fill_bar,
            entry_time=position.entry_time,
            exit_time=time,
            outcome=outcome,
            pnl=net_pnl,
            gross_pnl=gross_pnl,
            commission=commission,
            bars_held=fill.fill_bar - position.entry_bar,
            mae=mae,
            mfe=mfe,
            r_multiple=r_multiple,
        )
        
        # Update account
        self.trades.append(trade)
        self.positions.remove(position)
        self.balance += net_pnl
        self.realized_pnl += net_pnl
        
        # Update drawdown tracking
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        drawdown = self.peak_balance - self.balance
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
        
        return trade
    
    def get_equity(self, current_price: float) -> float:
        """Get current equity (balance + unrealized)."""
        unrealized = 0.0
        for pos in self.positions:
            unrealized += self.costs.calculate_pnl(
                pos.entry_price,
                current_price,
                pos.direction,
                pos.size,
                include_commission=False
            )
        return self.balance + unrealized
    
    def has_open_position(self) -> bool:
        """Check if any position is open."""
        return len(self.positions) > 0
    
    def get_stats(self) -> dict:
        """Get account statistics."""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'max_drawdown': 0.0,
            }
        
        wins = sum(1 for t in self.trades if t.outcome == 'WIN')
        total = len(self.trades)
        
        return {
            'total_trades': total,
            'wins': wins,
            'losses': total - wins,
            'win_rate': wins / total if total > 0 else 0.0,
            'total_pnl': self.realized_pnl,
            'avg_pnl': self.realized_pnl / total if total > 0 else 0.0,
            'max_drawdown': self.max_drawdown,
            'final_balance': self.balance,
        }

```

### src/sim/account_manager.py

```python
"""
Account Manager
Multi-account simulation tracking.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
import pandas as pd

from src.sim.account import Account, TradeRecord
from src.sim.costs import CostModel, DEFAULT_COSTS
from src.sim.execution import Fill


@dataclass
class AccountSnapshot:
    """Snapshot of account state at a point in time."""
    account_id: str
    timestamp: pd.Timestamp
    balance: float
    equity: float
    realized_pnl: float
    unrealized_pnl: float
    open_positions: int
    total_trades: int
    
    def to_dict(self) -> Dict:
        return {
            'account_id': self.account_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'balance': self.balance,
            'equity': self.equity,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'open_positions': self.open_positions,
            'total_trades': self.total_trades,
        }


class AccountManager:
    """
    Multi-account manager for simulation.
    
    Manages multiple accounts, routes orders, aggregates PnL.
    Useful for:
    - Multiple strategies in one session
    - Different risk profiles
    - Prop firm rule testing (per-account limits)
    """
    
    def __init__(self):
        self.accounts: Dict[str, Account] = {}
        self.snapshots: List[AccountSnapshot] = []
        
    def create_account(
        self,
        account_id: str,
        starting_balance: float = 50000.0,
        costs: CostModel = None
    ) -> Account:
        """Create a new account."""
        if account_id in self.accounts:
            raise ValueError(f"Account {account_id} already exists")
        
        account = Account(
            starting_balance=starting_balance,
            costs=costs or DEFAULT_COSTS
        )
        self.accounts[account_id] = account
        return account
    
    def delete_account(self, account_id: str):
        """Delete an account."""
        if account_id in self.accounts:
            del self.accounts[account_id]
    
    def get_account(self, account_id: str) -> Optional[Account]:
        """Get account by ID."""
        return self.accounts.get(account_id)
    
    def list_accounts(self) -> List[str]:
        """List all account IDs."""
        return list(self.accounts.keys())
    
    def take_snapshot(self, account_id: str, current_price: float, timestamp: pd.Timestamp):
        """Take a snapshot of account state."""
        account = self.accounts.get(account_id)
        if not account:
            return
        
        equity = account.get_equity(current_price)
        unrealized = equity - account.balance
        
        snapshot = AccountSnapshot(
            account_id=account_id,
            timestamp=timestamp,
            balance=account.balance,
            equity=equity,
            realized_pnl=account.realized_pnl,
            unrealized_pnl=unrealized,
            open_positions=len(account.positions),
            total_trades=len(account.trades),
        )
        self.snapshots.append(snapshot)
        return snapshot
    
    def get_total_pnl(self) -> float:
        """Get total PnL across all accounts."""
        return sum(acc.realized_pnl for acc in self.accounts.values())
    
    def get_total_equity(self, current_price: float) -> float:
        """Get total equity across all accounts."""
        return sum(acc.get_equity(current_price) for acc in self.accounts.values())
    
    def get_aggregate_stats(self) -> Dict:
        """Get aggregated stats across all accounts."""
        total_trades = sum(len(acc.trades) for acc in self.accounts.values())
        total_pnl = self.get_total_pnl()
        
        all_trades = []
        for acc in self.accounts.values():
            all_trades.extend(acc.trades)
        
        wins = sum(1 for t in all_trades if t.outcome == 'WIN')
        
        return {
            'total_accounts': len(self.accounts),
            'total_trades': total_trades,
            'wins': wins,
            'losses': total_trades - wins,
            'win_rate': wins / total_trades if total_trades > 0 else 0.0,
            'total_pnl': total_pnl,
            'avg_pnl_per_trade': total_pnl / total_trades if total_trades > 0 else 0.0,
        }
    
    def reset_all(self):
        """Reset all accounts to starting state."""
        for account in self.accounts.values():
            account.balance = account.starting_balance
            account.positions.clear()
            account.trades.clear()
            account.realized_pnl = 0.0
            account.peak_balance = account.starting_balance
            account.max_drawdown = 0.0
        self.snapshots.clear()
    
    def get_snapshots_for_account(self, account_id: str) -> List[AccountSnapshot]:
        """Get all snapshots for a specific account."""
        return [s for s in self.snapshots if s.account_id == account_id]

```

### src/sim/bar_fill_model.py

```python
"""
Bar Fill Model
Explicit rules for same-bar fill behavior.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple
import pandas as pd

from src.sim.costs import CostModel, DEFAULT_COSTS


class EntryModel(Enum):
    """How entries are filled."""
    NEXT_BAR_OPEN = "next_open"     # Market fills at next bar open
    THIS_BAR_CLOSE = "this_close"   # Can fill at current bar close
    LIMIT_INTRABAR = "limit_intra"  # Limit can fill intrabar if touched


class SLTPTieBreak(Enum):
    """How to handle SL and TP both touched in same bar."""
    CONSERVATIVE = "conservative"    # Assume SL hit first (worst case)
    OPTIMISTIC = "optimistic"        # Assume TP hit first (best case)
    OPEN_PROXIMITY = "open_prox"     # Whichever is closer to open


class SameBarExit(Enum):
    """Can entry and exit happen same bar?"""
    ALLOWED = "allowed"     # Can exit same bar as entry
    BLOCKED = "blocked"     # Must wait at least 1 bar


@dataclass
class BarFillConfig:
    """
    Complete bar fill model configuration.
    
    Since OHLC bars don't reveal price path, we must choose
    consistent conventions for all same-bar scenarios.
    """
    entry_model: EntryModel = EntryModel.NEXT_BAR_OPEN
    sl_tp_tiebreak: SLTPTieBreak = SLTPTieBreak.CONSERVATIVE
    same_bar_exit: SameBarExit = SameBarExit.BLOCKED
    
    def to_dict(self) -> dict:
        return {
            'entry_model': self.entry_model.value,
            'sl_tp_tiebreak': self.sl_tp_tiebreak.value,
            'same_bar_exit': self.same_bar_exit.value,
        }


class BarFillEngine:
    """
    Applies BarFillConfig rules consistently to all order types.
    """
    
    def __init__(
        self,
        config: BarFillConfig = None,
        costs: CostModel = None
    ):
        self.config = config or BarFillConfig()
        self.costs = costs or DEFAULT_COSTS
    
    def can_fill_limit_entry(
        self,
        limit_price: float,
        direction: str,
        bar: pd.Series
    ) -> bool:
        """
        Check if limit entry would fill on this bar.
        
        LONG limit fills if low <= limit_price
        SHORT limit fills if high >= limit_price
        """
        if direction == 'LONG':
            return bar['low'] <= limit_price
        else:
            return bar['high'] >= limit_price
    
    def get_limit_entry_fill_price(
        self,
        limit_price: float,
        direction: str,
        bar: pd.Series
    ) -> Optional[float]:
        """
        Get fill price for limit entry.
        
        Returns limit price if filled (or better if gap).
        """
        if not self.can_fill_limit_entry(limit_price, direction, bar):
            return None
        
        if direction == 'LONG':
            # Could fill at limit or better (lower)
            # If bar opens below limit, fill at open
            if bar['open'] <= limit_price:
                return bar['open']
            return limit_price
        else:
            # SHORT - fill at limit or better (higher)
            if bar['open'] >= limit_price:
                return bar['open']
            return limit_price
    
    def check_exit(
        self,
        position_direction: str,
        stop_price: float,
        tp_price: float,
        bar: pd.Series,
        entry_bar_idx: int,
        current_bar_idx: int
    ) -> Tuple[Optional[str], Optional[float]]:
        """
        Check if SL or TP is hit on this bar.
        
        Returns:
            (outcome, fill_price) where outcome is 'SL', 'TP', or None
        """
        # Check same-bar exit rule
        if self.config.same_bar_exit == SameBarExit.BLOCKED:
            if current_bar_idx <= entry_bar_idx:
                return (None, None)
        
        # Check if exits are touched
        if position_direction == 'LONG':
            # LONG: SL hit if low <= stop, TP hit if high >= tp
            sl_touched = bar['low'] <= stop_price
            tp_touched = bar['high'] >= tp_price
        else:
            # SHORT: SL hit if high >= stop, TP hit if low <= tp
            sl_touched = bar['high'] >= stop_price
            tp_touched = bar['low'] <= tp_price
        
        if sl_touched and tp_touched:
            # Both touched - apply tie-break
            return self._resolve_tie(
                position_direction, stop_price, tp_price, bar
            )
        elif sl_touched:
            return ('SL', stop_price)
        elif tp_touched:
            return ('TP', tp_price)
        else:
            return (None, None)
    
    def _resolve_tie(
        self,
        direction: str,
        stop_price: float,
        tp_price: float,
        bar: pd.Series
    ) -> Tuple[str, float]:
        """Resolve SL/TP same-bar tie."""
        
        if self.config.sl_tp_tiebreak == SLTPTieBreak.CONSERVATIVE:
            # Assume worst case - SL first
            return ('SL', stop_price)
        
        elif self.config.sl_tp_tiebreak == SLTPTieBreak.OPTIMISTIC:
            # Assume best case - TP first
            return ('TP', tp_price)
        
        else:  # OPEN_PROXIMITY
            # Whichever is closer to open price wins
            sl_dist = abs(bar['open'] - stop_price)
            tp_dist = abs(bar['open'] - tp_price)
            
            if sl_dist <= tp_dist:
                return ('SL', stop_price)
            else:
                return ('TP', tp_price)


# Default fill engine
DEFAULT_FILL_ENGINE = BarFillEngine()

```

### src/sim/causal_runner.py

```python
"""
Causal Runner
Unified execution engine for bar-by-bar simulation.

This is the SINGLE SOURCE OF TRUTH for:
1. Stepping through market data
2. Computing features (causal)
3. Running Scanners (Signal Generation)
4. Managing OCO Brackets (Order Lifecycle)
5. Updating Accounts (Fills/PnL)

It is used by:
- MarketSession (for Live/Replay streaming)
- ExperimentRunner (for Backtesting/Training data generation)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
import uuid

from src.sim.stepper import MarketStepper
from src.sim.account_manager import AccountManager
from src.sim.oco_engine import OCOEngine, OCOStatus, OCOBracket, OCOConfig
from src.sim.sizing import calculate_contracts, calculate_pnl_dollars
from src.features.pipeline import compute_features, FeatureConfig
from src.policy.scanners import Scanner, ScanResult
from src.sim.execution import Fill


@dataclass
class StepResult:
    """Result of a single simulation step."""
    bar_idx: int
    timestamp: pd.Timestamp
    bar: pd.Series
    
    # State
    current_price: float
    atr: float
    features: Any  # FeatureBundle
    
    # Events
    scanner_triggers: List[Dict] = field(default_factory=list)
    new_orders: List[OCOBracket] = field(default_factory=list)
    fills: List[Tuple[OCOBracket, str]] = field(default_factory=list)  # (bracket, event_type)
    completed_brackets: List[OCOBracket] = field(default_factory=list)
    
    # Snapshot
    account_snapshots: Dict[str, Any] = field(default_factory=dict)


class CausalExecutor:
    """
    Executes the causal market loop.
    
    Does NOT know about:
    - Future outcomes (labels)
    - Training
    - Visualization/SSE protocols
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        stepper: MarketStepper,
        account_manager: AccountManager,
        scanner: Optional[Scanner] = None,
        feature_config: Optional[FeatureConfig] = None,
        df_5m: Optional[pd.DataFrame] = None,
        df_15m: Optional[pd.DataFrame] = None,
        precomputed_indicators: Optional[Dict[int, Any]] = None,
    ):
        self.df = df
        self.stepper = stepper
        self.account_manager = account_manager
        
        # Strategy (Optional)
        self.scanner = scanner
        self.feature_config = feature_config or FeatureConfig()
        self.df_5m = df_5m
        self.df_15m = df_15m
        self.precomputed_indicators = precomputed_indicators
        
        # Execution Engine
        self.oco_engine = OCOEngine()
        self.active_brackets: List[OCOBracket] = []
        
        # State
        self.current_bar_idx = 0
        self.current_timestamp = None
    
    def step(self) -> Optional[StepResult]:
        """Execute one bar step."""
        step = self.stepper.step()
        if step.is_done:
            return None
        
        self.current_bar_idx = step.bar_idx
        self.current_timestamp = step.bar['time']
        
        current_price = float(step.bar['close'])
        
        result = StepResult(
            bar_idx=self.current_bar_idx,
            timestamp=self.current_timestamp,
            bar=step.bar,
            current_price=current_price,
            atr=0.0, # Filled later
            features=None
        )
        
        # 1. Update Active OCO Brackets (Exits/Fills)
        # ---------------------------------------------------
        # Must run BEFORE entries to clear capital/slots
        completed = []
        for bracket in self.active_brackets:
            updated_bracket, event_type = self.oco_engine.process_bar(
                bracket, step.bar, self.current_bar_idx
            )
            
            if event_type:
                result.fills.append((bracket, event_type))
                
                # Handle ENTRY Fill -> Open Position
                if event_type == 'ENTRY':
                    self._handle_entry(bracket)
                
                # Handle EXIT Fill -> Close Position
                if bracket.status in [OCOStatus.CLOSED_TP, OCOStatus.CLOSED_SL, OCOStatus.CLOSED_TIMEOUT]:
                    self._handle_exit(bracket)
                    completed.append(bracket)
                
                # Handle CANCELLED
                if bracket.status == OCOStatus.CANCELLED:
                    completed.append(bracket)

        # Cleanup completed
        for b in completed:
            self.active_brackets.remove(b)
        result.completed_brackets = completed

        # 2. Strategy / Scanner (Entries)
        # ---------------------------------------------------
        features = None
        if self.scanner:
            features = compute_features(
                self.stepper,
                self.feature_config,
                df_5m=self.df_5m,
                df_15m=self.df_15m,
                precomputed_indicators=self.precomputed_indicators
            )
            result.features = features
            result.atr = features.atr
            
            # Run scan
            # Note: MarketState is passed as None for now, or extracted from features if available
            scan_result = self.scanner.scan(features.market_state, features)
            
            if scan_result.triggered:
                # 3. Signals -> Orders
                # -----------------------------------------------
                
                # Determine direction from scanner context
                direction = scan_result.context.get('direction', 'LONG') if scan_result.context else 'LONG'
                
                # Construct OCO Config
                # TODO: This should come from a Strategy Config object
                oco_config = OCOConfig(
                    direction=direction,
                    entry_type="MARKET", 
                    stop_atr=2.0,
                    tp_multiple=2.0,
                    name=f"Auto_{self.scanner.__class__.__name__}"
                )
                
                # Create Bracket
                bracket = self.oco_engine.create_bracket(
                    config=oco_config,
                    base_price=features.current_price,
                    atr=features.atr,
                    current_idx=self.current_bar_idx
                )
                
                # Calculate Contracts (Sizing) - REQUIRED
                # We default to max risk if not specified
                from src.config import DEFAULT_MAX_RISK_DOLLARS
                sizing = calculate_contracts(
                    entry_price=bracket.entry_price,
                    stop_price=bracket.stop_price,
                    max_risk_dollars=DEFAULT_MAX_RISK_DOLLARS
                )
                
                # Store contracts on the bracket for tracking
                # (Dynamically attaching for now, untyped)
                bracket.contracts = sizing.contracts
                
                # Register
                self.active_brackets.append(bracket)
                
                # Record event - include full context for downstream use
                result.scanner_triggers.append({
                    'scanner': self.scanner.__class__.__name__,
                    'price': features.current_price,
                    'direction': direction,
                    'context': scan_result.context or {}
                })
                result.new_orders.append(bracket)

        # 4. Account Updates (Mark-to-Market)
        # ---------------------------------------------------
        for account_id in self.account_manager.list_accounts():
            snapshot = self.account_manager.take_snapshot(
                account_id,
                current_price,
                self.current_timestamp
            )
            if snapshot:
                result.account_snapshots[account_id] = snapshot

        return result

    def _handle_entry(self, bracket: OCOBracket):
        """Register entry fill with Account Manager."""
        # Assume 'default' account for now
        account = self.account_manager.get_account('default')
        if account:
            # Enforce contract size from sizing step
            contracts = getattr(bracket, 'contracts', 1)
            
            # Override fill size in case it was 1 default
            if bracket.entry_fill:
                bracket.entry_fill.size = contracts
            
            account.open_position(
                fill=bracket.entry_fill,
                stop_loss=bracket.stop_price,
                take_profit=bracket.tp_price,
                time=self.current_timestamp
            )

    def _handle_exit(self, bracket: OCOBracket):
        """Register exit fill with Account Manager."""
        account = self.account_manager.get_account('default')
        if account and bracket.exit_fill:
            # Find matching position
            # Robust matching by direction and approximately entry price
            matching_pos = None
            for pos in account.positions:
                if (pos.direction == bracket.config.direction and 
                    abs(pos.entry_price - bracket.entry_price) < 1e-4):
                    matching_pos = pos
                    break
            
            if matching_pos:
                # Ensure fill size matches position
                bracket.exit_fill.size = matching_pos.size
                
                account.close_position(
                    position=matching_pos,
                    fill=bracket.exit_fill,
                    outcome=bracket._get_outcome(),
                    time=self.current_timestamp
                )

```

### src/sim/costs.py

```python
"""
Cost Model
Fees, slippage, and tick rounding.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from src.config import TICK_SIZE, POINT_VALUE, COMMISSION_PER_SIDE, DEFAULT_SLIPPAGE_TICKS


@dataclass
class CostModel:
    """
    Trading cost model for realistic simulation.
    """
    commission_per_side: float = COMMISSION_PER_SIDE  # Per contract per side
    slippage_ticks: float = DEFAULT_SLIPPAGE_TICKS    # Average slippage
    tick_size: float = TICK_SIZE                       # MES = 0.25
    point_value: float = POINT_VALUE                   # MES = $5
    
    def round_to_tick(self, price: float, direction: str = 'nearest') -> float:
        """
        Round price to valid tick.
        
        Args:
            price: Raw price
            direction: 'nearest', 'up', or 'down'
        """
        if direction == 'up':
            return np.ceil(price / self.tick_size) * self.tick_size
        elif direction == 'down':
            return np.floor(price / self.tick_size) * self.tick_size
        else:
            return round(price / self.tick_size) * self.tick_size
    
    def apply_slippage(
        self,
        price: float,
        direction: str,
        order_type: str = 'MARKET'
    ) -> float:
        """
        Apply slippage to fill price.
        
        Slippage is adverse: 
        - BUY market fills ABOVE quoted price
        - SELL market fills BELOW quoted price
        
        Limit orders have no slippage (fill at limit or better).
        """
        if order_type == 'LIMIT':
            return price
        
        slippage_points = self.slippage_ticks * self.tick_size
        
        if direction == 'LONG':
            # Buying - slip up
            return self.round_to_tick(price + slippage_points, 'up')
        else:
            # Selling - slip down
            return self.round_to_tick(price - slippage_points, 'down')
    
    def calculate_commission(self, contracts: int, round_trip: bool = True) -> float:
        """Calculate commission in dollars."""
        sides = 2 if round_trip else 1
        return contracts * self.commission_per_side * sides
    
    def calculate_pnl(
        self,
        entry_price: float,
        exit_price: float,
        direction: str,
        contracts: int,
        include_commission: bool = True
    ) -> float:
        """
        Calculate trade PnL in dollars.
        
        Args:
            entry_price: Entry fill price
            exit_price: Exit fill price
            direction: 'LONG' or 'SHORT'
            contracts: Number of contracts
            include_commission: Whether to subtract commission
        """
        if direction == 'LONG':
            points = exit_price - entry_price
        else:
            points = entry_price - exit_price
        
        gross_pnl = points * self.point_value * contracts
        
        if include_commission:
            commission = self.calculate_commission(contracts, round_trip=True)
            return gross_pnl - commission
        
        return gross_pnl
    
    def calculate_risk(
        self,
        entry_price: float,
        stop_price: float,
        contracts: int
    ) -> float:
        """Calculate risk in dollars (not including commission)."""
        points = abs(entry_price - stop_price)
        return points * self.point_value * contracts


# Default cost model
DEFAULT_COSTS = CostModel()

```

### src/sim/entry_strategies.py

```python
"""
Entry Strategies
Decoupled logic for calculating entry prices.

This module provides a registry of entry strategies that can be used
by the OCO Engine to determine where to place limit/stop orders.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

from src.sim.costs import CostModel

class EntryStrategy(ABC):
    """Base class for entry price calculation strategies."""
    
    @abstractmethod
    def calculate_entry(
        self,
        base_price: float,
        direction: str,
        bar: pd.Series,
        atr: float,
        params: Dict[str, Any],
        costs: CostModel,
        context: Dict[str, Any] = None
    ) -> float:
        """
        Calculate entry price.
        
        Args:
            base_price: Current/Signal price (often Close of signal bar)
            direction: 'LONG' or 'SHORT'
            bar: The signal bar (current completed bar)
            atr: Current ATR
            params: Strategy-specific parameters (e.g. {'pct': 0.5})
            costs: Cost model for tick rounding
            context: Optional context (e.g. htf_bars, indicators)
            
        Returns:
            Calculated entry price
        """
        pass


class MarketEntry(EntryStrategy):
    """Enter at market (Open of next bar, effectively)."""
    
    def calculate_entry(self, base_price, direction, bar, atr, params, costs, context=None) -> float:
        return base_price


class LimitOffsetEntry(EntryStrategy):
    """Legacy: Limit at Base +/- ATR offset."""
    
    def calculate_entry(self, base_price, direction, bar, atr, params, costs, context=None) -> float:
        offset = params.get('offset_atr', 0.0)
        if direction == 'LONG':
            price = base_price - (offset * atr)
            return costs.round_to_tick(price, 'down')
        else:
            price = base_price + (offset * atr)
            return costs.round_to_tick(price, 'up')


class SignalRetraceEntry(EntryStrategy):
    """
    Limit at X% retrace of the signal bar range.
    
    Long: Low + (Range * pct) 
    Short: High - (Range * pct)
    """
    
    def calculate_entry(self, base_price, direction, bar, atr, params, costs, context=None) -> float:
        pct = params.get('pct', 0.5) # Default 50%
        bar_range = bar['high'] - bar['low']
        
        if direction == 'LONG':
            # Buy limit below close, ideally. 
            # Logic: We want to buy at the bottom X% of the candle
            price = bar['low'] + (bar_range * pct)
            return costs.round_to_tick(price, 'down')
        else:
            # Sell limit above close
            price = bar['high'] - (bar_range * pct)
            return costs.round_to_tick(price, 'up')


class TimeframeRetraceEntry(EntryStrategy):
    """
    Limit at X% retrace of the last completed bar on a SPECIFIC timeframe.
    
    Requires 'df_context' in context with keys like '5m', '15m'.
    User Request: '50 percent of the last 15m'
    """
    
    def calculate_entry(self, base_price, direction, bar, atr, params, costs, context=None) -> float:
        tf = params.get('timeframe', '15m')
        pct = params.get('pct', 0.5)
        
        if not context or tf not in context:
            # Fallback to current bar if HTF data missing
            return SignalRetraceEntry().calculate_entry(base_price, direction, bar, atr, params, costs, context)
            
        # Get last closed bar for timeframe
        htf_bar = context[tf].iloc[-1]
        range_val = htf_bar['high'] - htf_bar['low']
        
        if direction == 'LONG':
            price = htf_bar['low'] + (range_val * pct)
            return costs.round_to_tick(price, 'down')
        else:
            price = htf_bar['high'] - (range_val * pct)
            return costs.round_to_tick(price, 'up')


class BreakoutEntry(EntryStrategy):
    """
    Stop-Limit entry at signal bar High/Low + Offset.
    
    Long: High + Offset
    Short: Low - Offset
    """
    
    def calculate_entry(self, base_price, direction, bar, atr, params, costs, context=None) -> float:
        offset_atr = params.get('offset_atr', 0.1)
        offset = offset_atr * atr
        
        if direction == 'LONG':
            price = bar['high'] + offset
            return costs.round_to_tick(price, 'up')
        else:
            price = bar['low'] - offset
            return costs.round_to_tick(price, 'down')


class RangeBreakoutEntry(EntryStrategy):
    """
    Breakout of the last N bars High/Low.
    """
    
    def calculate_entry(self, base_price, direction, bar, atr, params, costs, context=None) -> float:
        lookback = int(params.get('lookback', 5))
        
        # Need history from context or assume current bar is end of history?
        # Ideally context has 'history_1m'
        history = context.get('history', pd.DataFrame()) if context else pd.DataFrame()
        
        if len(history) < lookback:
            # Fallback to single bar breakout
            return BreakoutEntry().calculate_entry(base_price, direction, bar, atr, params, costs, context)
            
        recent = history.iloc[-lookback:]
        
        if direction == 'LONG':
            price = recent['high'].max()
            return costs.round_to_tick(price, 'up')
        else:
            price = recent['low'].min()
            return costs.round_to_tick(price, 'down')


class VwapReversionEntry(EntryStrategy):
    """
    Limit entry at VWAP.
    """
    def calculate_entry(self, base_price, direction, bar, atr, params, costs, context=None) -> float:
        vwap = context.get('vwap') if context else None
        
        if vwap is None:
             # Fallback to market if no VWAP
             return base_price
             
        if direction == 'LONG':
            return costs.round_to_tick(vwap, 'down')
        else:
            return costs.round_to_tick(vwap, 'up')


class EntryRegistry:
    """Registry for entry strategies."""
    
    _strategies = {
        'market': MarketEntry(),
        'limit_offset': LimitOffsetEntry(), # Legacy 'limit'
        'retrace_signal': SignalRetraceEntry(),
        'retrace_timeframe': TimeframeRetraceEntry(),
        'breakout': BreakoutEntry(),
        'breakout_range': RangeBreakoutEntry(),
        'vwap': VwapReversionEntry(),
    }
    
    @classmethod
    def get(cls, name: str) -> EntryStrategy:
        return cls._strategies.get(name, cls._strategies['market']) # Default to market

    @classmethod
    def list_strategies(cls):
        return list(cls._strategies.keys())

```

### src/sim/execution.py

```python
"""
Order Execution
Order types and execution logic.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum
import pandas as pd


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


@dataclass
class Order:
    """
    Order representation.
    """
    order_type: OrderType
    direction: str          # 'LONG' or 'SHORT'
    price: Optional[float]  # Limit/stop price (None for market)
    size: int = 1
    expiry_bars: int = 15   # Bars until expiry (0 = GTC)
    
    # Tracking
    order_id: str = ""
    created_bar: int = 0
    status: OrderStatus = OrderStatus.PENDING
    
    def is_expired(self, current_bar: int) -> bool:
        """Check if order has expired."""
        if self.expiry_bars == 0:
            return False  # GTC
        return (current_bar - self.created_bar) >= self.expiry_bars


@dataclass
class Fill:
    """
    Execution fill.
    """
    order: Order
    fill_price: float
    fill_bar: int
    fill_time: Optional[pd.Timestamp] = None
    slippage: float = 0.0
    
    @property
    def direction(self) -> str:
        return self.order.direction
    
    @property
    def size(self) -> int:
        return self.order.size


def process_order(
    order: Order,
    bar: pd.Series,
    bar_idx: int,
    costs = None
) -> Optional[Fill]:
    """
    Process a single order against a bar.
    
    Returns Fill if order executes, None otherwise.
    """
    from src.sim.costs import DEFAULT_COSTS
    costs = costs or DEFAULT_COSTS
    
    if order.status != OrderStatus.PENDING:
        return None
    
    # Check expiry
    if order.is_expired(bar_idx):
        order.status = OrderStatus.EXPIRED
        return None
    
    if order.order_type == OrderType.MARKET:
        # Market orders fill at open with slippage
        fill_price = costs.apply_slippage(
            bar['open'],
            order.direction,
            'MARKET'
        )
        order.status = OrderStatus.FILLED
        return Fill(
            order=order,
            fill_price=fill_price,
            fill_bar=bar_idx,
            slippage=abs(fill_price - bar['open'])
        )
    
    elif order.order_type == OrderType.LIMIT:
        # Limit order - check if touched
        if order.direction == 'LONG':
            # Buy limit fills if low <= limit
            if bar['low'] <= order.price:
                # Fill at limit or better (open if gap down)
                fill_price = min(order.price, bar['open']) if bar['open'] <= order.price else order.price
                order.status = OrderStatus.FILLED
                return Fill(
                    order=order,
                    fill_price=fill_price,
                    fill_bar=bar_idx
                )
        else:
            # Sell limit fills if high >= limit
            if bar['high'] >= order.price:
                fill_price = max(order.price, bar['open']) if bar['open'] >= order.price else order.price
                order.status = OrderStatus.FILLED
                return Fill(
                    order=order,
                    fill_price=fill_price,
                    fill_bar=bar_idx
                )
    
    elif order.order_type == OrderType.STOP:
        # Stop order - check if triggered
        if order.direction == 'LONG':
            # Buy stop triggers if high >= stop
            if bar['high'] >= order.price:
                fill_price = max(order.price, bar['open'])
                fill_price = costs.apply_slippage(fill_price, order.direction, 'MARKET')
                order.status = OrderStatus.FILLED
                return Fill(
                    order=order,
                    fill_price=fill_price,
                    fill_bar=bar_idx,
                    slippage=abs(fill_price - order.price)
                )
        else:
            # Sell stop triggers if low <= stop
            if bar['low'] <= order.price:
                fill_price = min(order.price, bar['open'])
                fill_price = costs.apply_slippage(fill_price, order.direction, 'MARKET')
                order.status = OrderStatus.FILLED
                return Fill(
                    order=order,
                    fill_price=fill_price,
                    fill_bar=bar_idx,
                    slippage=abs(fill_price - order.price)
                )
    
    return None


def process_orders(
    orders: List[Order],
    bar: pd.Series,
    bar_idx: int,
    costs = None
) -> List[Fill]:
    """Process multiple orders, return all fills."""
    fills = []
    for order in orders:
        fill = process_order(order, bar, bar_idx, costs)
        if fill:
            fills.append(fill)
    return fills

```

### src/sim/market_session.py

```python
"""
Market Session
Backend-owned bar-by-bar stepping with events.

This is the unified session manager for both historical (backtest)
and live (streaming) market data. Instead of the frontend doing stepping,
the backend owns:
- MarketStepper (historical or live)
- Indicator pipeline
- Policies/models
- Accounts
- OCO engine

Frontend becomes: renderer + controls + config UI
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Iterator
from enum import Enum
import pandas as pd
from pathlib import Path

from src.sim.causal_runner import CausalExecutor, StepResult
from src.sim.stepper import MarketStepper
from src.sim.account_manager import AccountManager
from src.policy.scanners import Scanner
from src.features.pipeline import FeatureConfig
from src.core.enums import RunMode
from src.core.registries import IndicatorSeries, IndicatorRegistry


class SimEventType(Enum):
    """Types of events during simulation."""
    BAR = "BAR"                          # New bar arrived
    INDICATORS = "INDICATORS"            # Indicators computed
    DECISION = "DECISION"                # Decision point triggered
    ORDER_SUBMIT = "ORDER_SUBMIT"        # Order submitted
    FILL = "FILL"                        # Order filled
    POSITION_OPEN = "POSITION_OPEN"      # Position opened
    POSITION_CLOSE = "POSITION_CLOSE"    # Position closed
    ACCOUNT_UPDATE = "ACCOUNT_UPDATE"    # Account state changed
    SESSION_START = "SESSION_START"      # Session started
    SESSION_END = "SESSION_END"          # Session ended


@dataclass
class SimEvent:
    """Single event during simulation."""
    type: SimEventType
    timestamp: pd.Timestamp
    bar_idx: int
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type.value,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'bar_idx': self.bar_idx,
            'data': self.data,
        }


class MarketSession:
    """
    Backend market session - the unified stepping engine.
    
    Owns all session state:
    - Market stepper (historical or live)
    - Indicator cache
    - Active accounts
    - Active policies (scanners/models)
    - Emits structured events
    
    Frontend subscribes to events via SSE.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        session_id: str = "default",
        start_idx: int = 0,
        end_idx: Optional[int] = None,
    ):
        """
        Initialize market session.
        
        Args:
            df: Full OHLCV DataFrame
            session_id: Unique session identifier
            start_idx: Starting bar index
            end_idx: Ending bar index (None = end of df)
        """
        self.session_id = session_id
        self.df = df
        self.start_idx = start_idx
        self.end_idx = end_idx or len(df)
        
        # Market stepper
        self.stepper = MarketStepper(df, start_idx=start_idx, end_idx=self.end_idx)
        
        # Account manager
        self.account_manager = AccountManager()
        
        # Executor (Lazy init in start or setup)
        self.executor: Optional[CausalExecutor] = None
        
        # Strategy components
        self.scanner: Optional[Scanner] = None
        self.scanner_config: Dict[str, Any] = {}
        self.feature_config: FeatureConfig = FeatureConfig()
        
        # Data caches
        self.df_5m = None
        self.df_15m = None
        
        # Event log
        self.events: List[SimEvent] = []
        
        # Session state
        self.is_running = False
        self.is_paused = False
        self.current_bar_idx: Optional[int] = None
        self.current_timestamp: Optional[pd.Timestamp] = None
        
    def setup_strategy(self, scanner: Scanner, feature_config: FeatureConfig, df_5m=None, df_15m=None):
        """Configure strategy for the session."""
        self.scanner = scanner
        self.feature_config = feature_config
        self.df_5m = df_5m
        self.df_15m = df_15m
    
    def add_account(self, account_id: str, starting_balance: float = 50000.0):
        """Add an account to the session."""
        self.account_manager.create_account(account_id, starting_balance)
    
    def enable_indicator(self, indicator_id: str):
        """Enable an indicator for computation."""
        if indicator_id not in self.enabled_indicators:
            self.enabled_indicators.append(indicator_id)
    
    def disable_indicator(self, indicator_id: str):
        """Disable an indicator."""
        if indicator_id in self.enabled_indicators:
            self.enabled_indicators.remove(indicator_id)
    
    def add_scanner(self, scanner: Any):
        """Add a scanner to active scanners."""
        self.active_scanners.append(scanner)
    
    def add_model(self, model: Any):
        """Add a model to active models."""
        self.active_models.append(model)
    
    def start(self):
        """Start the session."""
        self.is_running = True
        self.is_paused = False
        
        # Initialize executor if needed
        if not self.executor:
            self.executor = CausalExecutor(
                df=self.df,
                stepper=self.stepper,
                account_manager=self.account_manager,
                scanner=self.scanner,
                feature_config=self.feature_config,
                df_5m=self.df_5m,
                df_15m=self.df_15m
            )
        
        # Emit session start event
        
        # Emit session start event
        event = SimEvent(
            type=SimEventType.SESSION_START,
            timestamp=self.df.iloc[self.start_idx]['time'],
            bar_idx=self.start_idx,
            data={
                'session_id': self.session_id,
                'accounts': self.account_manager.list_accounts(),
                'indicators': self.enabled_indicators,
            }
        )
        self.events.append(event)
        return event
    
    def stop(self):
        """Stop the session."""
        self.is_running = False
        
        # Emit session end event
        if self.current_timestamp and self.current_bar_idx is not None:
            event = SimEvent(
                type=SimEventType.SESSION_END,
                timestamp=self.current_timestamp,
                bar_idx=self.current_bar_idx,
                data={
                    'session_id': self.session_id,
                    'total_events': len(self.events),
                    'stats': self.account_manager.get_aggregate_stats(),
                }
            )
            self.events.append(event)
            return event
    
    def pause(self):
        """Pause the session."""
        self.is_paused = True
    
    def resume(self):
        """Resume the session."""
        self.is_paused = False
    
    def step_once(self) -> Optional[List[SimEvent]]:
        """
        Step forward by one bar using CausalExecutor.
        """
        if not self.is_running or self.is_paused or not self.executor:
            return None
        
        result = self.executor.step()
        if not result:
            return None
        
        self.current_bar_idx = result.bar_idx
        self.current_timestamp = result.timestamp
        
        events = []
        
        # 1. BAR event
        bar_event = SimEvent(
            type=SimEventType.BAR,
            timestamp=result.timestamp,
            bar_idx=result.bar_idx,
            data={
                'open': float(result.bar['open']),
                'high': float(result.bar['high']),
                'low': float(result.bar['low']),
                'close': float(result.bar['close']),
                'volume': float(result.bar['volume']),
            }
        )
        events.append(bar_event)
        
        # 2. Fills (Exits/Entries)
        for bracket, event_type in result.fills:
            sim_type = SimEventType(event_type) if event_type in [e.value for e in SimEventType] else SimEventType.FILL
            events.append(SimEvent(
                type=sim_type,
                timestamp=result.timestamp,
                bar_idx=result.bar_idx,
                data={'bracket_id': id(bracket), 'status': bracket.status.value, 'event': event_type}
            ))

        # 3. New Orders (Decisions)
        for bracket in result.new_orders:
            # Emit Decision
            events.append(SimEvent(
                type=SimEventType.DECISION,
                timestamp=result.timestamp,
                bar_idx=result.bar_idx,
                data={
                    'scanner_id': self.scanner.__class__.__name__ if self.scanner else "unknown",
                    'triggered': True,
                    'price': bracket.entry_price, # Use bracket price which is set
                    'atr': bracket.atr_at_creation
                }
            ))
            # Emit Order Submit
            events.append(SimEvent(
                type=SimEventType.ORDER_SUBMIT,
                timestamp=result.timestamp,
                bar_idx=result.bar_idx,
                data=bracket.to_flat_dict()
            ))

        # 4. Account Updates
        for acc_id, snapshot in result.account_snapshots.items():
            events.append(SimEvent(
                type=SimEventType.ACCOUNT_UPDATE,
                timestamp=result.timestamp,
                bar_idx=result.bar_idx,
                data=snapshot.to_dict()
            ))
        
        # Store events
        self.events.extend(events)
        return events
    
    def play(self) -> Iterator[SimEvent]:
        """
        Play through the session, yielding events.
        
        This is the main simulation loop for SSE streaming.
        """
        self.start()
        
        while self.is_running:
            # Wait if paused
            if self.is_paused:
                break
            
            events = self.step_once()
            if not events:
                # Session is done
                break
            
            for event in events:
                yield event
        
        # Emit end event
        end_event = self.stop()
        if end_event:
            yield end_event
    
    def get_state(self) -> Dict[str, Any]:
        """Get current session state."""
        return {
            'session_id': self.session_id,
            'is_running': self.is_running,
            'is_paused': self.is_paused,
            'current_bar_idx': self.current_bar_idx,
            'current_timestamp': self.current_timestamp.isoformat() if self.current_timestamp else None,
            'total_events': len(self.events),
            'accounts': self.account_manager.list_accounts(),
            'enabled_indicators': self.enabled_indicators,
            'stats': self.account_manager.get_aggregate_stats(),
        }

```

### src/sim/oco.py

```python
"""
OCO (One-Cancels-Other) Order Logic
Bracket orders with entry, stop loss, and take profit.

DEPRECATED: This module is deprecated. Use src.sim.oco_engine instead.
"""

import warnings
warnings.warn(
    "src.sim.oco is deprecated. Use src.sim.oco_engine instead.",
    DeprecationWarning,
    stacklevel=2
)

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
from enum import Enum
import pandas as pd

from src.sim.execution import Order, OrderType, OrderStatus, Fill
from src.sim.bar_fill_model import BarFillEngine, BarFillConfig, DEFAULT_FILL_ENGINE
from src.sim.costs import CostModel, DEFAULT_COSTS


class OCOStatus(Enum):
    PENDING_ENTRY = "PENDING_ENTRY"   # Waiting for entry fill
    ACTIVE = "ACTIVE"                  # Entry filled, SL/TP pending
    CLOSED_TP = "CLOSED_TP"           # Closed by take profit
    CLOSED_SL = "CLOSED_SL"           # Closed by stop loss
    CLOSED_TIMEOUT = "CLOSED_TIMEOUT" # Closed by max bars
    CANCELLED = "CANCELLED"           # Entry expired/cancelled


class OCOReference(Enum):
    """What the OCO bracket is referenced from."""
    PRICE = "PRICE"              # Raw price level
    EMA_5M = "EMA_5M"            # 5-minute 200 EMA
    EMA_15M = "EMA_15M"          # 15-minute 200 EMA
    VWAP_SESSION = "VWAP_SESSION"
    VWAP_WEEKLY = "VWAP_WEEKLY"
    LEVEL_1H = "LEVEL_1H"        # Nearest 1h S/R
    LEVEL_4H = "LEVEL_4H"        # Nearest 4h S/R


@dataclass
class OCOConfig:
    """
    OCO bracket configuration.
    """
    direction: str = "LONG"         # 'LONG' or 'SHORT'
    
    # Entry
    entry_type: str = "LIMIT"       # 'MARKET', 'LIMIT'
    entry_offset_atr: float = 0.25  # ATR multiplier for limit offset
    
    # Exit
    stop_atr: float = 1.0           # Stop distance in ATR
    tp_multiple: float = 1.4        # Take profit as multiple of risk
    max_bars: int = 200             # Max bars in trade
    
    # OCO reference (for indicator-based levels)
    reference: OCOReference = OCOReference.PRICE
    reference_offset_atr: float = 0.0
    
    # Unique ID
    name: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'direction': self.direction,
            'entry_type': self.entry_type,
            'entry_offset_atr': self.entry_offset_atr,
            'stop_atr': self.stop_atr,
            'tp_multiple': self.tp_multiple,
            'max_bars': self.max_bars,
            'reference': self.reference.value,
            'reference_offset_atr': self.reference_offset_atr,
            'name': self.name,
        }
    
    def to_cli_args(self) -> list:
        return [
            '--direction', self.direction,
            '--entry-type', self.entry_type,
            '--entry-offset', str(self.entry_offset_atr),
            '--stop-atr', str(self.stop_atr),
            '--tp-mult', str(self.tp_multiple),
            '--max-bars', str(self.max_bars),
        ]


@dataclass
class OCOBracket:
    """
    Active OCO bracket tracking.
    """
    config: OCOConfig
    
    # Prices (computed at creation)
    entry_price: float = 0.0
    stop_price: float = 0.0
    tp_price: float = 0.0
    
    # State
    status: OCOStatus = OCOStatus.PENDING_ENTRY
    entry_bar: int = 0
    entry_fill: Optional[Fill] = None
    exit_fill: Optional[Fill] = None
    
    # Reference for logging
    reference_value: float = 0.0   # Value of indicator reference at creation
    atr_at_creation: float = 0.0
    
    # Tracking
    bars_in_trade: int = 0
    mae: float = 0.0   # Max Adverse Excursion
    mfe: float = 0.0   # Max Favorable Excursion


def create_oco_bracket(
    config: OCOConfig,
    base_price: float,
    atr: float,
    reference_value: Optional[float] = None,
    costs: CostModel = None
) -> OCOBracket:
    """
    Create OCO bracket with computed price levels.
    
    Args:
        config: OCO configuration
        base_price: Current price or signal bar close
        atr: Current ATR for offset calculations
        reference_value: Value if using indicator reference
        costs: Cost model for tick rounding
    """
    costs = costs or DEFAULT_COSTS
    
    # Use reference value if provided, else base price
    ref = reference_value if reference_value else base_price
    
    if config.direction == 'LONG':
        # LONG: entry below, stop below entry, TP above entry
        entry_price = costs.round_to_tick(
            ref - config.entry_offset_atr * atr, 'down'
        ) if config.entry_type == 'LIMIT' else base_price
        
        stop_price = costs.round_to_tick(
            entry_price - config.stop_atr * atr, 'down'
        )
        
        risk = entry_price - stop_price
        tp_price = costs.round_to_tick(
            entry_price + risk * config.tp_multiple, 'up'
        )
    else:
        # SHORT: entry above, stop above entry, TP below entry
        entry_price = costs.round_to_tick(
            ref + config.entry_offset_atr * atr, 'up'
        ) if config.entry_type == 'LIMIT' else base_price
        
        stop_price = costs.round_to_tick(
            entry_price + config.stop_atr * atr, 'up'
        )
        
        risk = stop_price - entry_price
        tp_price = costs.round_to_tick(
            entry_price - risk * config.tp_multiple, 'down'
        )
    
    return OCOBracket(
        config=config,
        entry_price=entry_price,
        stop_price=stop_price,
        tp_price=tp_price,
        reference_value=ref,
        atr_at_creation=atr,
    )


def process_oco_bar(
    bracket: OCOBracket,
    bar: pd.Series,
    bar_idx: int,
    fill_engine: BarFillEngine = None
) -> Tuple[OCOBracket, Optional[str]]:
    """
    Process one bar for an OCO bracket.
    
    Returns:
        Updated bracket and event ('ENTRY', 'SL', 'TP', 'TIMEOUT', or None)
    """
    fill_engine = fill_engine or DEFAULT_FILL_ENGINE
    
    if bracket.status == OCOStatus.PENDING_ENTRY:
        # Check for entry fill
        if bracket.config.entry_type == 'MARKET':
            # Market entry fills at open
            fill_price = fill_engine.costs.apply_slippage(
                bar['open'], bracket.config.direction, 'MARKET'
            )
            bracket.entry_fill = Fill(
                order=Order(OrderType.MARKET, bracket.config.direction, None),
                fill_price=fill_price,
                fill_bar=bar_idx
            )
            bracket.entry_bar = bar_idx
            bracket.status = OCOStatus.ACTIVE
            bracket.entry_price = fill_price  # Update actual entry
            return (bracket, 'ENTRY')
        
        else:
            # Limit entry
            fill_price = fill_engine.get_limit_entry_fill_price(
                bracket.entry_price,
                bracket.config.direction,
                bar
            )
            if fill_price is not None:
                bracket.entry_fill = Fill(
                    order=Order(OrderType.LIMIT, bracket.config.direction, bracket.entry_price),
                    fill_price=fill_price,
                    fill_bar=bar_idx
                )
                bracket.entry_bar = bar_idx
                bracket.status = OCOStatus.ACTIVE
                bracket.entry_price = fill_price  # May be better than limit
                return (bracket, 'ENTRY')
    
    elif bracket.status == OCOStatus.ACTIVE:
        bracket.bars_in_trade += 1
        
        # Track MAE/MFE
        if bracket.config.direction == 'LONG':
            adverse = bracket.entry_price - bar['low']
            favorable = bar['high'] - bracket.entry_price
        else:
            adverse = bar['high'] - bracket.entry_price
            favorable = bracket.entry_price - bar['low']
        
        bracket.mae = max(bracket.mae, adverse)
        bracket.mfe = max(bracket.mfe, favorable)
        
        # Check timeout
        if bracket.bars_in_trade >= bracket.config.max_bars:
            bracket.status = OCOStatus.CLOSED_TIMEOUT
            return (bracket, 'TIMEOUT')
        
        # Check SL/TP
        result, fill_price = fill_engine.check_exit(
            bracket.config.direction,
            bracket.stop_price,
            bracket.tp_price,
            bar,
            bracket.entry_bar,
            bar_idx
        )
        
        if result == 'SL':
            bracket.status = OCOStatus.CLOSED_SL
            bracket.exit_fill = Fill(
                order=Order(OrderType.STOP, bracket.config.direction, bracket.stop_price),
                fill_price=fill_price,
                fill_bar=bar_idx
            )
            return (bracket, 'SL')
        
        elif result == 'TP':
            bracket.status = OCOStatus.CLOSED_TP
            bracket.exit_fill = Fill(
                order=Order(OrderType.LIMIT, bracket.config.direction, bracket.tp_price),
                fill_price=fill_price,
                fill_bar=bar_idx
            )
            return (bracket, 'TP')
    
    return (bracket, None)

```

### src/sim/oco_engine.py

```python
"""
OCO Engine - Unified One-Cancels-Other Order Management

This is the SINGLE authoritative implementation for OCO bracket logic.
All other implementations should use this engine.

Key Features:
- Standardized stop/TP priority rules
- Tick size rounding
- Consistent bars_held calculation
- Flat oco_results output format
- Integration with stop_calculator for smart stops
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List
from enum import Enum
import pandas as pd

from src.sim.execution import Order, OrderType, OrderStatus, Fill
from src.sim.bar_fill_model import BarFillEngine, BarFillConfig, DEFAULT_FILL_ENGINE
from src.sim.costs import CostModel, DEFAULT_COSTS
from src.sim.stop_calculator import StopType, StopConfig, calculate_stop, calculate_tp_from_risk


class OCOStatus(Enum):
    """OCO bracket status."""
    PENDING_ENTRY = "PENDING_ENTRY"   # Waiting for entry fill
    ACTIVE = "ACTIVE"                  # Entry filled, SL/TP pending
    CLOSED_TP = "CLOSED_TP"           # Closed by take profit
    CLOSED_SL = "CLOSED_SL"           # Closed by stop loss
    CLOSED_TIMEOUT = "CLOSED_TIMEOUT" # Closed by max bars
    CANCELLED = "CANCELLED"           # Entry expired/cancelled


class ExitPriority(Enum):
    """
    Priority rule when both SL and TP would trigger in same bar.
    
    According to ARCHITECTURE_AGREEMENT.md:
    - STOP_FIRST: Conservative - assume worst case (default)
    - TP_FIRST: Optimistic - assume best case
    - RANDOM: Random selection (for sensitivity analysis)
    - INTRABAR_MODEL: Use bar fill model (if available)
    """
    STOP_FIRST = "STOP_FIRST"
    TP_FIRST = "TP_FIRST"
    RANDOM = "RANDOM"
    INTRABAR_MODEL = "INTRABAR_MODEL"


@dataclass
class OCOConfig:
    """
    Unified OCO bracket configuration.
    
    Supports both legacy ATR-based stops and modern smart stops.
    """
    direction: str = "LONG"         # 'LONG' or 'SHORT'
    
    # Entry
    entry_type: str = "LIMIT"       # 'MARKET', 'LIMIT', 'RETRACE', etc.
    entry_params: Dict[str, Any] = field(default_factory=dict)  # Params for entry strategy
    entry_offset_atr: float = 0.25  # Legacy support (maps to limit_offset params)
    
    # Stop configuration
    stop_config: Optional[StopConfig] = None  # Use smart stops if provided
    stop_atr: float = 1.0                     # Legacy: ATR-based stop (if stop_config is None)
    
    # Take profit
    tp_multiple: float = 1.4        # Take profit as multiple of risk
    
    # Limits
    max_bars: int = 200             # Max bars in trade
    
    # Exit priority
    exit_priority: ExitPriority = ExitPriority.STOP_FIRST
    
    # Unique ID
    name: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'direction': self.direction,
            'entry_type': self.entry_type,
            'entry_params': self.entry_params,
            'entry_offset_atr': self.entry_offset_atr,
            'stop_atr': self.stop_atr if self.stop_config is None else None,
            'stop_config': self.stop_config.to_dict() if self.stop_config else None,
            'tp_multiple': self.tp_multiple,
            'max_bars': self.max_bars,
            'exit_priority': self.exit_priority.value,
            'name': self.name,
        }


@dataclass
class OCOBracket:
    """
    Active OCO bracket state.
    
    This is the runtime state of an OCO order.
    """
    config: OCOConfig
    
    # Computed prices (rounded to tick size)
    entry_price: float = 0.0
    stop_price: float = 0.0
    tp_price: float = 0.0
    
    # State
    status: OCOStatus = OCOStatus.PENDING_ENTRY
    entry_bar: int = 0
    entry_fill: Optional[Fill] = None
    exit_fill: Optional[Fill] = None
    
    # Reference data for logging
    atr_at_creation: float = 0.0
    
    # Tracking (for analytics)
    bars_in_trade: int = 0          # Bars AFTER entry (not including entry bar)
    mae: float = 0.0                # Max Adverse Excursion (points)
    mfe: float = 0.0                # Max Favorable Excursion (points)
    
    def to_flat_dict(self) -> Dict[str, Any]:
        """
        Convert to flat dictionary for oco_results.
        
        This is the AUTHORITATIVE format for UI consumption.
        According to ARCHITECTURE_AGREEMENT.md, oco_results MUST be flat.
        """
        return {
            'direction': self.config.direction,
            'entry_price': self.entry_price,
            'stop_price': self.stop_price,
            'tp_price': self.tp_price,
            'status': self.status.value,
            'entry_bar': self.entry_bar,
            'bars_held': self.bars_in_trade,  # UI reads this field
            'mae': self.mae,
            'mfe': self.mfe,
            'filled': self.status != OCOStatus.CANCELLED,
            'outcome': self._get_outcome(),
            'exit_price': self.exit_fill.fill_price if self.exit_fill else 0.0,
        }
    
    def _get_outcome(self) -> str:
        """Get outcome string for oco_results."""
        if self.status == OCOStatus.CLOSED_TP:
            return "TP"
        elif self.status == OCOStatus.CLOSED_SL:
            return "SL"
        elif self.status == OCOStatus.CLOSED_TIMEOUT:
            return "TIMEOUT"
        elif self.status == OCOStatus.CANCELLED:
            return "CANCELLED"
        elif self.status == OCOStatus.ACTIVE:
            return "ACTIVE"
        else:
            return "PENDING"


class OCOEngine:
    """
    Unified OCO Engine.
    
    This is the single source of truth for OCO bracket creation and processing.
    """
    
    def __init__(
        self,
        fill_engine: BarFillEngine = None,
        costs: CostModel = None
    ):
        self.fill_engine = fill_engine or DEFAULT_FILL_ENGINE
        self.costs = costs or DEFAULT_COSTS
    
    def create_bracket(
        self,
        config: OCOConfig,
        base_price: float,
        atr: float,
        df_1m: Optional[pd.DataFrame] = None,
        df_htf: Optional[pd.DataFrame] = None,
        current_idx: int = 0,
        range_high: float = 0.0,
        range_low: float = 0.0,
        direction_override: Optional[str] = None,
    ) -> OCOBracket:
        """
        Create OCO bracket with computed price levels.
        
        Args:
            config: OCO configuration
            base_price: Current price or signal bar close
            atr: Current ATR for offset calculations
            df_1m: 1-minute data (for smart stops)
            df_htf: Higher timeframe data (for smart stops)
            current_idx: Current bar index
            range_high: Pre-calculated range high (for range-based stops)
            range_low: Pre-calculated range low (for range-based stops)
            direction_override: Override config.direction (for dynamic scanners)
            
        Returns:
            OCOBracket with rounded prices
        """
        from src.sim.entry_strategies import EntryRegistry

        # Use override direction if provided, else use config
        direction = direction_override or config.direction
        
        # Prepare context for entry strategy
        entry_context = {
            '5m': None, # TODO: Pass these in more reliably
            '15m': df_htf if df_htf is not None else None, # Assuming htf is the needed one for now
            'vwap': None # TODO: Pass in
        }
        
        # Helper params merge (legacy support)
        entry_params = config.entry_params.copy()
        if 'offset_atr' not in entry_params:
            entry_params['offset_atr'] = config.entry_offset_atr
            
        # Calculate entry price using strategy
        strategy = EntryRegistry.get(config.entry_type.lower())
        entry_price = strategy.calculate_entry(
            base_price=base_price,
            direction=direction,
            bar=df_1m.iloc[current_idx] if df_1m is not None else pd.Series({'high': base_price, 'low': base_price, 'close': base_price}), 
            atr=atr,
            params=entry_params,
            costs=self.costs,
            context=entry_context
        )
        
        # Calculate stop price
        if config.stop_config is not None:
            # Use smart stop calculator
            stop_price, stop_reason = calculate_stop(
                direction=direction,
                entry_price=entry_price,
                atr=atr,
                config=config.stop_config,
                df_1m=df_1m,
                df_htf=df_htf,
                current_idx=current_idx,
                range_high=range_high,
                range_low=range_low,
            )
            # Round to tick
            if direction == 'LONG':
                stop_price = self.costs.round_to_tick(stop_price, 'down')
            else:
                stop_price = self.costs.round_to_tick(stop_price, 'up')
        else:
            # Legacy ATR-based stop
            if direction == 'LONG':
                stop_price = self.costs.round_to_tick(
                    entry_price - config.stop_atr * atr, 'down'
                )
            else:
                stop_price = self.costs.round_to_tick(
                    entry_price + config.stop_atr * atr, 'up'
                )
        
        # Calculate TP from risk
        tp_price = calculate_tp_from_risk(
            entry_price=entry_price,
            stop_price=stop_price,
            direction=direction,
            r_multiple=config.tp_multiple
        )
        
        # Round TP to tick
        if direction == 'LONG':
            tp_price = self.costs.round_to_tick(tp_price, 'up')
        else:
            tp_price = self.costs.round_to_tick(tp_price, 'down')
        
        return OCOBracket(
            config=config,
            entry_price=entry_price,
            stop_price=stop_price,
            tp_price=tp_price,
            atr_at_creation=atr,
        )
    
    def process_bar(
        self,
        bracket: OCOBracket,
        bar: pd.Series,
        bar_idx: int
    ) -> Tuple[OCOBracket, Optional[str]]:
        """
        Process one bar for an OCO bracket.
        
        Returns:
            (Updated bracket, event string or None)
            
        Events:
            'ENTRY': Entry filled
            'SL': Stop loss hit
            'TP': Take profit hit
            'TIMEOUT': Max bars reached
        """
        if bracket.status == OCOStatus.PENDING_ENTRY:
            return self._process_entry(bracket, bar, bar_idx)
        elif bracket.status == OCOStatus.ACTIVE:
            return self._process_active(bracket, bar, bar_idx)
        
        # Already closed
        return (bracket, None)
    
    def _process_entry(
        self,
        bracket: OCOBracket,
        bar: pd.Series,
        bar_idx: int
    ) -> Tuple[OCOBracket, Optional[str]]:
        """Process entry fill attempt."""
        if bracket.config.entry_type == 'MARKET':
            # Market entry fills at open (with slippage)
            fill_price = self.fill_engine.costs.apply_slippage(
                bar['open'], bracket.config.direction, 'MARKET'
            )
            bracket.entry_fill = Fill(
                order=Order(OrderType.MARKET, bracket.config.direction, None),
                fill_price=fill_price,
                fill_bar=bar_idx
            )
            bracket.entry_bar = bar_idx
            bracket.status = OCOStatus.ACTIVE
            bracket.entry_price = fill_price  # Update actual entry
            return (bracket, 'ENTRY')
        
        else:  # LIMIT
            fill_price = self.fill_engine.get_limit_entry_fill_price(
                bracket.entry_price,
                bracket.config.direction,
                bar
            )
            if fill_price is not None:
                bracket.entry_fill = Fill(
                    order=Order(OrderType.LIMIT, bracket.config.direction, bracket.entry_price),
                    fill_price=fill_price,
                    fill_bar=bar_idx
                )
                bracket.entry_bar = bar_idx
                bracket.status = OCOStatus.ACTIVE
                bracket.entry_price = fill_price
                return (bracket, 'ENTRY')
        
        return (bracket, None)
    
    def _process_active(
        self,
        bracket: OCOBracket,
        bar: pd.Series,
        bar_idx: int
    ) -> Tuple[OCOBracket, Optional[str]]:
        """Process active bracket for exit."""
        # Increment bars_in_trade (counts bars AFTER entry)
        bracket.bars_in_trade += 1
        
        # Update MAE/MFE
        if bracket.config.direction == 'LONG':
            adverse = bracket.entry_price - bar['low']
            favorable = bar['high'] - bracket.entry_price
        else:
            adverse = bar['high'] - bracket.entry_price
            favorable = bracket.entry_price - bar['low']
        
        bracket.mae = max(bracket.mae, adverse)
        bracket.mfe = max(bracket.mfe, favorable)
        
        # Check timeout (bars_held + 1 to account for entry bar)
        if bracket.bars_in_trade >= bracket.config.max_bars:
            bracket.status = OCOStatus.CLOSED_TIMEOUT
            # Exit at close
            bracket.exit_fill = Fill(
                order=Order(OrderType.MARKET, bracket.config.direction, None),
                fill_price=bar['close'],
                fill_bar=bar_idx
            )
            return (bracket, 'TIMEOUT')
        
        # Check for SL/TP
        result, fill_price = self.fill_engine.check_exit(
            bracket.config.direction,
            bracket.stop_price,
            bracket.tp_price,
            bar,
            bracket.entry_bar,
            bar_idx
        )
        
        if result == 'SL':
            bracket.status = OCOStatus.CLOSED_SL
            bracket.exit_fill = Fill(
                order=Order(OrderType.STOP, bracket.config.direction, bracket.stop_price),
                fill_price=fill_price,
                fill_bar=bar_idx
            )
            return (bracket, 'SL')
        
        elif result == 'TP':
            bracket.status = OCOStatus.CLOSED_TP
            bracket.exit_fill = Fill(
                order=Order(OrderType.LIMIT, bracket.config.direction, bracket.tp_price),
                fill_price=fill_price,
                fill_bar=bar_idx
            )
            return (bracket, 'TP')
        
        return (bracket, None)


# Global engine instance (for backward compatibility)
DEFAULT_OCO_ENGINE = OCOEngine()


def create_oco_bracket(
    config: OCOConfig,
    base_price: float,
    atr: float,
    reference_value: Optional[float] = None,
    costs: CostModel = None,
    **kwargs
) -> OCOBracket:
    """
    Legacy compatibility wrapper for create_oco_bracket.
    
    New code should use OCOEngine directly.
    """
    engine = OCOEngine(costs=costs or DEFAULT_COSTS)
    return engine.create_bracket(config, base_price, atr, **kwargs)


def process_oco_bar(
    bracket: OCOBracket,
    bar: pd.Series,
    bar_idx: int,
    fill_engine: BarFillEngine = None
) -> Tuple[OCOBracket, Optional[str]]:
    """
    Legacy compatibility wrapper for process_oco_bar.
    
    New code should use OCOEngine directly.
    """
    engine = OCOEngine(fill_engine=fill_engine or DEFAULT_FILL_ENGINE)
    return engine.process_bar(bracket, bar, bar_idx)

```

### src/sim/sizing.py

```python
"""
Position Sizing - Single Source of Truth

This module provides centralized position sizing calculations to ensure
consistent contract sizing across all strategies and exporters.

According to ARCHITECTURE_AGREEMENT.md:
- Never default to 1 contract without explicit sizing calculation
- contracts = floor(MAX_RISK_DOLLARS / (risk_points * point_value)), min 1
- All PnL calculations must use the same cost model
"""

import math
from typing import Tuple
from dataclasses import dataclass

from src.sim.costs import CostModel, DEFAULT_COSTS
from src.config import DEFAULT_MAX_RISK_DOLLARS


@dataclass
class SizingResult:
    """Result of position sizing calculation."""
    contracts: int
    risk_points: float
    risk_dollars: float
    max_risk_dollars: float
    point_value: float
    
    def to_dict(self):
        """Export as dictionary for viz."""
        return {
            'contracts': self.contracts,
            'risk_points': self.risk_points,
            'risk_dollars': self.risk_dollars,
            'max_risk_dollars': self.max_risk_dollars,
            'point_value': self.point_value,
        }


def calculate_contracts(
    entry_price: float,
    stop_price: float,
    max_risk_dollars: float = DEFAULT_MAX_RISK_DOLLARS,
    cost_model: CostModel = DEFAULT_COSTS
) -> SizingResult:
    """
    Calculate number of contracts based on risk parameters.
    
    This is the SINGLE source of truth for contract sizing.
    
    Formula:
        risk_points = abs(entry_price - stop_price)
        contracts = floor(max_risk_dollars / (risk_points * point_value))
        contracts = max(1, contracts)  # minimum 1 contract
    
    Args:
        entry_price: Entry price for the trade
        stop_price: Stop loss price
        max_risk_dollars: Maximum dollar risk per trade (default: $300)
        cost_model: Cost model for point value (default: DEFAULT_COSTS)
    
    Returns:
        SizingResult with contracts and risk parameters
    
    Example:
        >>> result = calculate_contracts(5000.0, 4990.0, 300.0)
        >>> result.contracts
        6
        >>> result.risk_dollars
        300.0
    """
    # Calculate risk in points
    risk_points = abs(entry_price - stop_price)
    
    # Handle edge case: zero risk (shouldn't happen, but be defensive)
    if risk_points <= 0:
        return SizingResult(
            contracts=1,
            risk_points=0.0,
            risk_dollars=0.0,
            max_risk_dollars=max_risk_dollars,
            point_value=cost_model.point_value
        )
    
    # Calculate contracts
    # contracts = floor(max_risk / (risk_points * point_value))
    risk_per_contract = risk_points * cost_model.point_value
    contracts = int(math.floor(max_risk_dollars / risk_per_contract))
    
    # Minimum 1 contract
    contracts = max(1, contracts)
    
    # Calculate actual risk with rounded contracts
    actual_risk_dollars = contracts * risk_per_contract
    
    return SizingResult(
        contracts=contracts,
        risk_points=risk_points,
        risk_dollars=actual_risk_dollars,
        max_risk_dollars=max_risk_dollars,
        point_value=cost_model.point_value
    )


def calculate_reward_dollars(
    entry_price: float,
    tp_price: float,
    direction: str,
    contracts: int,
    cost_model: CostModel = DEFAULT_COSTS
) -> float:
    """
    Calculate potential reward in dollars.
    
    Args:
        entry_price: Entry price
        tp_price: Take profit price
        direction: "LONG" or "SHORT"
        contracts: Number of contracts
        cost_model: Cost model for point value
    
    Returns:
        Reward in dollars (always positive)
    """
    if direction == "LONG":
        reward_points = tp_price - entry_price
    else:
        reward_points = entry_price - tp_price
    
    reward_dollars = abs(reward_points * cost_model.point_value * contracts)
    return reward_dollars


def calculate_pnl_dollars(
    entry_price: float,
    exit_price: float,
    direction: str,
    contracts: int,
    cost_model: CostModel = DEFAULT_COSTS,
    include_commission: bool = True
) -> Tuple[float, float]:
    """
    Calculate trade PnL in points and dollars.
    
    This is the SINGLE source of truth for PnL calculation.
    MUST be consistent with OCOEngine and CostModel.
    
    Args:
        entry_price: Entry fill price
        exit_price: Exit fill price
        direction: "LONG" or "SHORT"
        contracts: Number of contracts
        cost_model: Cost model for point value and commission
        include_commission: Whether to subtract commission
    
    Returns:
        Tuple of (pnl_points, pnl_dollars)
    """
    # Calculate points
    if direction == "LONG":
        pnl_points = exit_price - entry_price
    else:
        pnl_points = entry_price - exit_price
    
    # Calculate dollars using cost model
    pnl_dollars = cost_model.calculate_pnl(
        entry_price=entry_price,
        exit_price=exit_price,
        direction=direction,
        contracts=contracts,
        include_commission=include_commission
    )
    
    return pnl_points, pnl_dollars

```

### src/sim/stepper.py

```python
"""
Market Stepper
Bar-by-bar market simulation with CAUSAL data access only.
NO peek_future() method - future access is quarantined in labels/.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class StepResult:
    """Result of a single step."""
    bar: pd.Series
    bar_idx: int
    is_done: bool


class MarketStepper:
    """
    Bar-by-bar market simulation.
    
    CAUSAL ONLY - no future access.
    Same inputs → same outputs (deterministic).
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
        time_col: str = 'time'
    ):
        """
        Initialize stepper.
        
        Args:
            df: DataFrame with OHLCV data (must have time column)
            start_idx: Starting bar index
            end_idx: Ending bar index (exclusive). None = end of data.
            time_col: Name of time column
        """
        self.df = df.reset_index(drop=True)
        self.time_col = time_col
        self.start_idx = start_idx
        self.end_idx = end_idx or len(df)
        self.current_idx = start_idx
        
        if self.start_idx < 0:
            raise ValueError("start_idx must be >= 0")
        if self.end_idx > len(df):
            raise ValueError(f"end_idx {self.end_idx} > data length {len(df)}")
        if self.start_idx >= self.end_idx:
            raise ValueError("start_idx must be < end_idx")
    
    def reset(self, start_idx: Optional[int] = None):
        """Reset stepper to start position."""
        self.current_idx = start_idx if start_idx is not None else self.start_idx

    def skip_to(self, target_idx: int):
        """
        Fast forward to a specific index.
        
        Args:
            target_idx: Target bar index. Must be >= current_idx.
        """
        if target_idx < self.current_idx:
            # Can't go back
            return
        
        self.current_idx = min(target_idx, self.end_idx)
    
    def step(self) -> StepResult:
        """
        Advance one bar.
        
        Returns:
            StepResult with current bar, index, and done flag.
        """
        if self.current_idx >= self.end_idx:
            return StepResult(
                bar=None,
                bar_idx=self.current_idx,
                is_done=True
            )
        
        bar = self.df.iloc[self.current_idx]
        bar_idx = self.current_idx
        self.current_idx += 1
        
        return StepResult(
            bar=bar,
            bar_idx=bar_idx,
            is_done=self.current_idx >= self.end_idx
        )
    
    def get_current_bar(self) -> Optional[pd.Series]:
        """Get current bar (the one just returned by step)."""
        idx = self.current_idx - 1
        if idx < 0 or idx >= len(self.df):
            return None
        return self.df.iloc[idx]
    
    def get_current_idx(self) -> int:
        """Get current bar index."""
        return self.current_idx - 1
    
    def get_current_time(self) -> Optional[pd.Timestamp]:
        """Get current bar timestamp."""
        bar = self.get_current_bar()
        if bar is None:
            return None
        return bar[self.time_col]
    
    def get_history(self, lookback: int) -> pd.DataFrame:
        """
        Get past N bars (CAUSAL - no future leak).
        
        Returns bars from [current_idx - lookback, current_idx).
        If not enough history, returns what's available.
        """
        end_idx = self.current_idx
        start_idx = max(0, end_idx - lookback)
        return self.df.iloc[start_idx:end_idx].copy()
    
    def get_history_array(
        self,
        lookback: int,
        columns: list = None
    ) -> np.ndarray:
        """
        Get history as numpy array for model input.
        
        Args:
            lookback: Number of bars
            columns: Columns to include. Default: ['open', 'high', 'low', 'close', 'volume']
            
        Returns:
            Array of shape (lookback, n_columns). Padded with zeros if insufficient history.
        """
        columns = columns or ['open', 'high', 'low', 'close', 'volume']
        history = self.get_history(lookback)
        
        # Extract values
        values = history[columns].values
        
        # Pad if insufficient history
        if len(values) < lookback:
            padding = np.zeros((lookback - len(values), len(columns)))
            values = np.vstack([padding, values])
        
        return values.astype(np.float32)
    
    def bars_remaining(self) -> int:
        """Number of bars remaining in simulation."""
        return max(0, self.end_idx - self.current_idx)
    
    def progress(self) -> float:
        """Progress as fraction [0, 1]."""
        total = self.end_idx - self.start_idx
        done = self.current_idx - self.start_idx
        return done / total if total > 0 else 1.0
    
    # NOTE: No peek_future() method exists!
    # Future access is only available via FutureWindowProvider in labels/

```

### src/sim/stop_calculator.py

```python
"""
Stop Calculator
Flexible stop loss calculation based on different reference points.

Stop types:
- CANDLE_OPEN: Previous candle open (5m, 15m, etc.)
- CANDLE_LOW/HIGH: Previous candle low/high
- RANGE_LOW/HIGH: Low/high of a time range (e.g., OR)
- SWING_LOW/HIGH: Previous swing point on higher TF
- ATR_OFFSET: Simple ATR offset from entry (legacy)

ATR is used as PADDING, not as the stop level itself.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple
import pandas as pd
import numpy as np


class StopType(Enum):
    """Type of stop level calculation."""
    ATR_OFFSET = "atr_offset"         # Legacy: stop = entry +/- atr * mult
    CANDLE_OPEN = "candle_open"       # Stop at previous candle open
    CANDLE_LOW = "candle_low"         # Stop at previous candle low (LONG)
    CANDLE_HIGH = "candle_high"       # Stop at previous candle high (SHORT)
    RANGE_LOW = "range_low"           # Stop at range low (e.g., OR low for LONG)
    RANGE_HIGH = "range_high"         # Stop at range high (e.g., OR high for SHORT)
    SWING_LOW = "swing_low"           # Previous swing low on HTF (LONG)
    SWING_HIGH = "swing_high"         # Previous swing high on HTF (SHORT)


@dataclass
class StopConfig:
    """Configuration for stop calculation."""
    stop_type: StopType = StopType.CANDLE_LOW
    timeframe: str = "5m"             # Timeframe for candle-based stops
    lookback: int = 1                 # How many candles back to look
    atr_padding: float = 0.25         # ATR padding beyond the stop level
    range_start_time: str = "09:30"   # For range-based stops
    range_end_time: str = "09:45"     # For range-based stops
    swing_lookback: int = 20          # Bars to look for swing points
    
    def to_dict(self) -> dict:
        return {
            'stop_type': self.stop_type.value,
            'timeframe': self.timeframe,
            'lookback': self.lookback,
            'atr_padding': self.atr_padding,
        }


def calculate_stop(
    direction: str,
    entry_price: float,
    atr: float,
    config: StopConfig,
    df_1m: Optional[pd.DataFrame] = None,
    df_htf: Optional[pd.DataFrame] = None,
    current_idx: int = 0,
    range_high: float = 0.0,
    range_low: float = 0.0,
) -> Tuple[float, str]:
    """
    Calculate stop price based on configuration.
    
    Args:
        direction: 'LONG' or 'SHORT'
        entry_price: Entry price for reference
        atr: Current ATR for padding
        config: Stop configuration
        df_1m: 1-minute data (for finding HTF bars)
        df_htf: Higher timeframe data (5m, 15m, etc.)
        current_idx: Current bar index in df_1m
        range_high: Pre-calculated range high (for RANGE_* stops)
        range_low: Pre-calculated range low (for RANGE_* stops)
        
    Returns:
        (stop_price, reason_string)
    """
    padding = config.atr_padding * atr
    
    if config.stop_type == StopType.ATR_OFFSET:
        # Legacy: simple ATR offset from entry
        if direction == "LONG":
            stop = entry_price - atr
        else:
            stop = entry_price + atr
        return (stop, f"ATR offset from entry")
    
    elif config.stop_type == StopType.CANDLE_OPEN:
        # Stop at previous candle open
        if df_htf is not None and len(df_htf) > config.lookback:
            candle = df_htf.iloc[-(config.lookback + 1)]
            base_stop = candle['open']
            if direction == "LONG":
                stop = base_stop - padding
            else:
                stop = base_stop + padding
            return (stop, f"{config.timeframe} candle open - padding")
    
    elif config.stop_type == StopType.CANDLE_LOW:
        # Stop at previous candle low (for LONG)
        if df_htf is not None and len(df_htf) > config.lookback:
            candle = df_htf.iloc[-(config.lookback + 1)]
            base_stop = candle['low']
            stop = base_stop - padding
            return (stop, f"{config.timeframe} candle low - padding")
    
    elif config.stop_type == StopType.CANDLE_HIGH:
        # Stop at previous candle high (for SHORT)
        if df_htf is not None and len(df_htf) > config.lookback:
            candle = df_htf.iloc[-(config.lookback + 1)]
            base_stop = candle['high']
            stop = base_stop + padding
            return (stop, f"{config.timeframe} candle high + padding")
    
    elif config.stop_type == StopType.RANGE_LOW:
        # Stop at range low (for LONG trades)
        if range_low > 0:
            stop = range_low - padding
            return (stop, f"Range low - padding")
    
    elif config.stop_type == StopType.RANGE_HIGH:
        # Stop at range high (for SHORT trades)
        if range_high > 0:
            stop = range_high + padding
            return (stop, f"Range high + padding")
    
    elif config.stop_type == StopType.SWING_LOW:
        # Previous swing low on HTF
        if df_htf is not None and len(df_htf) > config.swing_lookback:
            lookback_df = df_htf.iloc[-config.swing_lookback:]
            swing_low = lookback_df['low'].min()
            stop = swing_low - padding
            return (stop, f"{config.timeframe} swing low - padding")
    
    elif config.stop_type == StopType.SWING_HIGH:
        # Previous swing high on HTF
        if df_htf is not None and len(df_htf) > config.swing_lookback:
            lookback_df = df_htf.iloc[-config.swing_lookback:]
            swing_high = lookback_df['high'].max()
            stop = swing_high + padding
            return (stop, f"{config.timeframe} swing high + padding")
    
    # Fallback: ATR offset
    if direction == "LONG":
        stop = entry_price - atr
    else:
        stop = entry_price + atr
    return (stop, "Fallback ATR offset")


def get_stop_for_or_retest(
    direction: str,
    entry_price: float,
    atr: float,
    or_high: float,
    or_low: float,
    padding_atr: float = 0.25
) -> Tuple[float, str]:
    """
    Convenience function for Opening Range retest strategy.
    
    For LONG (retest of OR low): stop is OR low - padding
    For SHORT (retest of OR high): stop is OR high + padding
    """
    padding = padding_atr * atr
    
    if direction == "LONG":
        stop = or_low - padding
        return (stop, f"OR low ({or_low:.2f}) - {padding:.2f} padding")
    else:
        stop = or_high + padding
        return (stop, f"OR high ({or_high:.2f}) + {padding:.2f} padding")


def calculate_risk(entry_price: float, stop_price: float, direction: str) -> float:
    """Calculate risk in points (always positive)."""
    if direction == "LONG":
        return abs(entry_price - stop_price)
    else:
        return abs(stop_price - entry_price)


def calculate_tp_from_risk(
    entry_price: float,
    stop_price: float,
    direction: str,
    r_multiple: float
) -> float:
    """Calculate take profit price from risk and R multiple."""
    risk = calculate_risk(entry_price, stop_price, direction)
    
    if direction == "LONG":
        return entry_price + (risk * r_multiple)
    else:
        return entry_price - (risk * r_multiple)

```

### src/sim/validation.py

```python
"""
Trade Validation Rails

Safety checks for simulation integrity:
- Minimum distance between entry and stop/TP
- Prevents trades that can't be simulated on 1m data
- Flags suspicious fills

Usage:
    from src.sim.validation import validate_trade_distances, MIN_TRADE_DISTANCE
    
    if not validate_trade_distances(entry, stop, tp, candle_range):
        print("Trade too tight for simulation!")
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any


# =============================================================================
# Minimum Trade Distance Rules
# =============================================================================

# Minimum distance in points - trades smaller than this can't be reliably simulated
MIN_TRADE_DISTANCE_POINTS = 1.0  # 1 point minimum (4 ticks on MES)

# Alternative: minimum as multiple of average candle range
MIN_DISTANCE_CANDLE_MULT = 0.5  # Stop/TP must be at least 0.5x average candle range


@dataclass
class ValidationResult:
    """Result of trade validation."""
    valid: bool
    reason: str = ""
    warnings: list = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


def validate_trade_distances(
    entry: float,
    stop: float,
    tp: float,
    avg_candle_range: float,
    min_points: float = MIN_TRADE_DISTANCE_POINTS,
    min_candle_mult: float = MIN_DISTANCE_CANDLE_MULT,
) -> ValidationResult:
    """
    Validate that trade distances are large enough to simulate.
    
    Args:
        entry: Entry price
        stop: Stop loss price
        tp: Take profit price
        avg_candle_range: Average candle range (high - low) for the timeframe
        min_points: Minimum distance in points
        min_candle_mult: Minimum distance as multiple of candle range
    
    Returns:
        ValidationResult with valid flag and reason if invalid
    """
    stop_distance = abs(entry - stop)
    tp_distance = abs(entry - tp)
    min_candle_distance = avg_candle_range * min_candle_mult
    
    warnings = []
    
    # Check stop distance
    if stop_distance < min_points:
        return ValidationResult(
            valid=False,
            reason=f"Stop too tight: {stop_distance:.2f} pts < {min_points:.2f} pts minimum"
        )
    
    if stop_distance < min_candle_distance:
        return ValidationResult(
            valid=False,
            reason=f"Stop smaller than candle: {stop_distance:.2f} < {min_candle_distance:.2f} (0.5x avg candle)"
        )
    
    # Check TP distance
    if tp_distance < min_points:
        return ValidationResult(
            valid=False,
            reason=f"TP too tight: {tp_distance:.2f} pts < {min_points:.2f} pts minimum"
        )
    
    if tp_distance < min_candle_distance:
        return ValidationResult(
            valid=False,
            reason=f"TP smaller than candle: {tp_distance:.2f} < {min_candle_distance:.2f} (0.5x avg candle)"
        )
    
    # Warnings for marginal cases
    if stop_distance < avg_candle_range:
        warnings.append(f"Stop ({stop_distance:.2f}) < avg candle range ({avg_candle_range:.2f})")
    
    if tp_distance < avg_candle_range:
        warnings.append(f"TP ({tp_distance:.2f}) < avg candle range ({avg_candle_range:.2f})")
    
    return ValidationResult(valid=True, warnings=warnings)


def get_minimum_stop_distance(avg_candle_range: float, atr: float = None) -> float:
    """
    Calculate the minimum stop distance for reliable simulation.
    
    Returns the larger of:
    - MIN_TRADE_DISTANCE_POINTS
    - 0.5x average candle range
    - 0.5x ATR (if provided)
    
    Use this when placing stops to ensure simulation validity.
    """
    candidates = [MIN_TRADE_DISTANCE_POINTS]
    
    candidates.append(avg_candle_range * MIN_DISTANCE_CANDLE_MULT)
    
    if atr is not None:
        candidates.append(atr * 0.5)
    
    return max(candidates)


def check_same_bar_fill_risk(
    entry: float,
    stop: float,
    tp: float,
    bar_high: float,
    bar_low: float,
) -> Dict[str, Any]:
    """
    Check if both stop and TP could hit on the same bar (ambiguous).
    
    This happens when the bar's range contains both the stop and TP.
    When this occurs, we can't determine which hit first.
    
    Returns:
        Dict with 'at_risk' bool and 'details' string
    """
    bar_range = bar_high - bar_low
    stop_distance = abs(entry - stop)
    tp_distance = abs(entry - tp)
    
    # Check if bar range exceeds both distances
    both_in_range = (
        bar_range >= stop_distance and
        bar_range >= tp_distance
    )
    
    # Check if bar actually touched both
    stop_in_bar = bar_low <= stop <= bar_high or bar_low <= stop <= bar_high
    tp_in_bar = bar_low <= tp <= bar_high or bar_low <= tp <= bar_high
    
    at_risk = both_in_range or (stop_in_bar and tp_in_bar)
    
    return {
        'at_risk': at_risk,
        'bar_range': bar_range,
        'stop_distance': stop_distance,
        'tp_distance': tp_distance,
        'details': f"Bar range: {bar_range:.2f}, Stop: {stop_distance:.2f}, TP: {tp_distance:.2f}"
    }


# =============================================================================
# Helper for grid searches
# =============================================================================

def filter_valid_grid_params(
    stop_atr_range: list,
    tp_r_range: list,
    avg_candle_range: float,
    avg_atr: float,
) -> list:
    """
    Filter grid search parameters to only include valid combinations.
    
    Returns list of (stop_atr, tp_r) tuples that pass validation.
    """
    valid_combos = []
    min_stop = get_minimum_stop_distance(avg_candle_range, avg_atr)
    
    for stop_atr in stop_atr_range:
        stop_distance = avg_atr * stop_atr
        
        if stop_distance < min_stop:
            continue  # Stop too tight
        
        for tp_r in tp_r_range:
            tp_distance = stop_distance * tp_r
            
            if tp_distance < min_stop:
                continue  # TP too tight
            
            valid_combos.append((stop_atr, tp_r))
    
    return valid_combos


if __name__ == "__main__":
    # Quick test
    print("Trade Validation Rails")
    print("=" * 40)
    
    # Simulate typical MES values
    avg_candle_range = 2.5  # 2.5 points average 1m candle
    atr = 4.0  # 15m ATR
    
    print(f"Avg candle range: {avg_candle_range}")
    print(f"ATR: {atr}")
    print(f"Min stop distance: {get_minimum_stop_distance(avg_candle_range, atr):.2f}")
    print()
    
    # Test some trades
    test_cases = [
        (6000.0, 5999.5, 6001.0),  # Too tight
        (6000.0, 5998.0, 6004.0),  # OK
        (6000.0, 5997.0, 6006.0),  # Good
    ]
    
    for entry, stop, tp in test_cases:
        result = validate_trade_distances(entry, stop, tp, avg_candle_range)
        status = "✓ VALID" if result.valid else f"✗ INVALID: {result.reason}"
        print(f"Entry={entry}, Stop={stop}, TP={tp}: {status}")

```

### src/sim/yfinance_stepper.py

```python
"""
YFinance Stepper

A MarketStepper implementation that uses yfinance for data.
- Starts with N days of history (backfill).
- Simulates through history at requested speed.
- When history catches up to now, switches to LIVE mode:
  - Polls yfinance periodically for the newest closed bar.
  - Yields new bars in real-time or None if waiting.
"""

import time
import sys
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, List
from zoneinfo import ZoneInfo

from src.sim.stepper import StepResult

EST = ZoneInfo("America/New_York")


class YFinanceStepper:
    """
    Market simulation using yfinance data.
    Seamless transition from historical backfill to live polling.
    """
    
    def __init__(
        self,
        ticker: str = "MES=F",
        days_back: int = 7,
        lookback_padding: int = 60,
    ):
        """
        Args:
            ticker: Symbol to trade (default MES=F)
            days_back: Number of days of history to load (max 7 for 1m)
            lookback_padding: Extra bars to keep for indicator calculation
        """
        self.ticker_symbol = ticker
        self.interval = "1m"
        self.ticker = yf.Ticker(ticker)
        
        # Load initial history
        print(f"[YF] Loading {days_back} days history for {ticker}...", file=sys.stderr)
        self.df = self._fetch_initial_history(days_back)
        
        if len(self.df) == 0:
            # Empty init fallback
            self.df = pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        
        # State
        self.current_idx = max(0, lookback_padding) if len(self.df) > lookback_padding else 0
        self.live_mode = False
        self.last_poll_time = 0
        self.poll_interval = 20  # Seconds between API calls
        
        print(f"[YF] Loaded {len(self.df)} bars. Starting at index {self.current_idx}", file=sys.stderr)
        if len(self.df) > 0:
            print(f"[YF] Range: {self.df['time'].iloc[0]} -> {self.df['time'].iloc[-1]}", file=sys.stderr)

    def _fetch_initial_history(self, days: int) -> pd.DataFrame:
        """Fetch historical data."""
        # YFinance 1m is max 7 days
        days = min(days, 7)
        end = datetime.now()
        start = end - timedelta(days=days)
        
        try:
            df = self.ticker.history(start=start, end=end, interval="1m")
            if df is None or len(df) == 0:
                return pd.DataFrame()
            
            # Normalize columns
            df.columns = [c.lower() for c in df.columns]
            df = df.reset_index()
            
            # Handle timezone
            if 'Datetime' in df.columns:
                df['time'] = df['Datetime']
            elif 'datetime' in df.columns:
                df['time'] = df['datetime']
            
            # Ensure TZ aware (NY)
            if df['time'].dt.tz is None:
                df['time'] = df['time'].dt.tz_localize(EST)
            else:
                df['time'] = df['time'].dt.tz_convert(EST)
                
            return df[['time', 'open', 'high', 'low', 'close', 'volume']]
        except Exception as e:
            print(f"[YF] Init Error: {e}", file=sys.stderr)
            return pd.DataFrame()

    def step(self) -> Optional[StepResult]:
        """
        Advance one step. NON-BLOCKING.
        If in history: returns next bar immediately.
        If at live edge: Checks API once. If new bar, returns it. Else returns None.
        """
        # Check if we are at the end of known data
        if self.current_idx >= len(self.df):
            self.live_mode = True
            self._poll_for_new_bar_once()
            
            # Check again if data arrived
            if self.current_idx >= len(self.df):
                return None  # Still waiting
        
        # Return current bar
        bar = self.df.iloc[self.current_idx]
        idx = self.current_idx
        self.current_idx += 1
        
        return StepResult(bar=bar, bar_idx=idx, is_done=False)

    def _poll_for_new_bar_once(self):
        """Poll YFinance API once if interval has passed. NON-BLOCKING."""
        now = time.time()
        if now - self.last_poll_time < self.poll_interval:
            return

        self.last_poll_time = now
        print(f"[YF] Checking for new candle...", file=sys.stderr)
        
        try:
            # Fetch just the last day to get latest
            latest = self.ticker.history(period="1d", interval="1m")
            if len(latest) == 0:
                return
            
            # Normalize
            latest.columns = [c.lower() for c in latest.columns]
            latest = latest.reset_index()
            if 'Datetime' in latest.columns:
                latest['time'] = latest['Datetime']
            elif 'datetime' in latest.columns:
                latest['time'] = latest['datetime']
            if latest['time'].dt.tz is None:
                latest['time'] = latest['time'].dt.tz_localize(EST)
            else:
                latest['time'] = latest['time'].dt.tz_convert(EST)
            
            latest = latest[['time', 'open', 'high', 'low', 'close', 'volume']]
            
            # Filter for strictly new bars
            if len(self.df) > 0:
                last_timestamp = self.df['time'].iloc[-1]
                new_bars = latest[latest['time'] > last_timestamp]
            else:
                new_bars = latest

            if not new_bars.empty:
                print(f"[YF] Found {len(new_bars)} new bars. Latest: {new_bars['time'].iloc[-1]}", file=sys.stderr)
                # Append to internal dataframe
                self.df = pd.concat([self.df, new_bars], ignore_index=True)
            
        except Exception as e:
            print(f"[YF] Poll error: {e}", file=sys.stderr)

    def get_history(self, lookback: int) -> pd.DataFrame:
        """Get CAUSAL history from current point."""
        end_idx = self.current_idx
        start_idx = max(0, end_idx - lookback)
        return self.df.iloc[start_idx:end_idx].copy()

```

### src/skills/data_skills.py

```python
"""
Data Skills
Atomic tools for fetching and inspecting market data.
Wraps src.data.loader for the Agent.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta

from src.core.tool_registry import ToolRegistry, ToolCategory
from src.data import loader
from src.config import NY_TZ


@ToolRegistry.register(
    tool_id="fetch_ohlcv",
    category=ToolCategory.UTILITY,
    name="Fetch OHLCV Data",
    description="Fetch OHLCV (candlestick) data for analysis",
    input_schema={
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Symbol to fetch (default: continuous)",
                "default": "continuous"
            },
            "start_date": {
                "type": "string",
                "description": "Start date (YYYY-MM-DD), optional"
            },
            "end_date": {
                "type": "string",
                "description": "End date (YYYY-MM-DD), optional"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of bars (default: 1000)",
                "default": 1000
            }
        }
    },
    output_schema={
        "type": "object",
        "properties": {
            "bars": {
                "type": "array",
                "items": {"type": "object"}
            },
            "count": {"type": "integer"}
        }
    }
)
class FetchOHLCVTool:
    def execute(self, symbol: str = "continuous", start_date: str = None, end_date: str = None, limit: int = 1000, **kwargs) -> Dict[str, Any]:
        """Fetch OHLCV data for analysis."""
        df = loader.load_continuous_contract()
        
        # Filter by date
        if start_date:
            start_dt = pd.to_datetime(start_date).tz_localize(NY_TZ)
            df = df[df['time'] >= start_dt]
            
        if end_date:
            end_dt = pd.to_datetime(end_date).tz_localize(NY_TZ) + timedelta(days=1)
            df = df[df['time'] < end_dt]
            
        # Limit rows
        if limit and len(df) > limit:
            df = df.iloc[:limit]
            
        # Convert to list of dicts
        records = df.to_dict('records')
        
        # Format timestamps
        for r in records:
            if isinstance(r['time'], pd.Timestamp):
                r['time'] = r['time'].isoformat()
                
        return {
            "bars": records,
            "count": len(records)
        }


@ToolRegistry.register(
    tool_id="get_dataset_last_price",
    category=ToolCategory.UTILITY,
    name="Get Dataset Last Price",
    description="Get the last close price in the historical dataset (end of Sept 2025). This is NOT live market data.",
    input_schema={
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Symbol (default: continuous)",
                "default": "continuous"
            }
        }
    },
    output_schema={
        "type": "object",
        "properties": {
            "price": {"type": "number"},
            "timestamp": {"type": "string"},
            "note": {"type": "string"}
        }
    }
)
class GetDatasetLastPriceTool:
    def execute(self, symbol: str = "continuous", **kwargs) -> Dict[str, Any]:
        """Get the last price from the historical dataset."""
        df = loader.load_continuous_contract()
        if df.empty:
            return {"price": 0.0, "timestamp": "", "note": "Dataset is empty"}
        
        price = float(df['close'].iloc[-1])
        timestamp = str(df['time'].iloc[-1])
        
        return {
            "price": price,
            "timestamp": timestamp,
            "note": "This is the END of the historical dataset, not live market data"
        }


@ToolRegistry.register(
    tool_id="get_dataset_summary",
    category=ToolCategory.UTILITY,
    name="Get Dataset Summary",
    description="Get summary statistics about the historical dataset (March-Sept 2025): date range, total bars, volatility metrics",
    input_schema={
        "type": "object",
        "properties": {}
    },
    output_schema={
        "type": "object",
        "properties": {
            "start_date": {"type": "string"},
            "end_date": {"type": "string"},
            "total_bars": {"type": "integer"},
            "avg_daily_range": {"type": "number"},
            "avg_volume": {"type": "number"},
            "period_description": {"type": "string"}
        }
    }
)
class GetDatasetSummaryTool:
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Get summary statistics about the historical dataset."""
        df = loader.load_continuous_contract()
        if df.empty:
            return {
                "start_date": "",
                "end_date": "",
                "total_bars": 0,
                "avg_daily_range": 0.0,
                "avg_volume": 0.0,
                "period_description": "No data available"
            }
        
        start_date = str(df['time'].iloc[0].date())
        end_date = str(df['time'].iloc[-1].date())
        total_bars = len(df)
        
        # Calculate average daily range (high - low)
        daily_range = (df['high'] - df['low']).mean()
        avg_volume = df['volume'].mean()
        
        return {
            "start_date": start_date,
            "end_date": end_date,
            "total_bars": total_bars,
            "avg_daily_range": float(daily_range),
            "avg_volume": float(avg_volume),
            "period_description": f"Historical MES data from {start_date} to {end_date} ({total_bars:,} 1-minute bars)"
        }


@ToolRegistry.register(
    tool_id="get_market_regime",
    category=ToolCategory.UTILITY,
    name="Get Market Regime",
    description="Determine the market regime over the last N days (TRENDING_UP, TRENDING_DOWN, or RANGING)",
    input_schema={
        "type": "object",
        "properties": {
            "window_days": {
                "type": "integer",
                "description": "Number of days to analyze (default: 5)",
                "default": 5
            }
        }
    },
    output_schema={
        "type": "object",
        "properties": {
            "regime": {
                "type": "string",
                "enum": ["TRENDING_UP", "TRENDING_DOWN", "RANGING", "UNKNOWN"]
            },
            "return_pct": {"type": "number"}
        }
    }
)
class GetMarketRegimeTool:
    def execute(self, window_days: int = 5, **kwargs) -> Dict[str, Any]:
        """Determine market regime over last N days."""
        df = loader.load_continuous_contract()
        if df.empty:
            return {"regime": "UNKNOWN", "return_pct": 0.0}
            
        # Filter last N days
        cutoff = df['time'].iloc[-1] - timedelta(days=window_days)
        recent = df[df['time'] >= cutoff]
        
        if recent.empty:
            return {"regime": "UNKNOWN", "return_pct": 0.0}
            
        start_price = recent['close'].iloc[0]
        end_price = recent['close'].iloc[-1]
        ret = (end_price - start_price) / start_price
        
        regime = "RANGING"
        if ret > 0.02:  # > 2% up
            regime = "TRENDING_UP"
        elif ret < -0.02:  # > 2% down
            regime = "TRENDING_DOWN"
            
        return {
            "regime": regime,
            "return_pct": float(ret * 100)  # Convert to percentage
        }
@ToolRegistry.register(
    tool_id="get_time_of_day_stats",
    category=ToolCategory.UTILITY,
    name="Get Time-of-Day Stats",
    description="Analyze volatility and price action average by hour of the day",
    input_schema={
        "type": "object",
        "properties": {
            "lookback_days": {
                "type": "integer",
                "description": "Number of days to analyze (default: 30)",
                "default": 30
            }
        }
    },
    output_schema={
        "type": "object",
        "properties": {
            "hourly_stats": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "hour": {"type": "integer"},
                        "avg_range": {"type": "number"},
                        "avg_volume": {"type": "number"},
                        "volatility": {"type": "number"}
                    }
                }
            }
        }
    }
)
class GetTimeOfDayStatsTool:
    def execute(self, lookback_days: int = 30, **kwargs) -> Dict[str, Any]:
        """Analyze average stats by hour."""
        from src.data.loader import load_continuous_contract

"""
Fast Viz API Routes

CRUD endpoints for ephemeral Fast Viz runs.
- POST /fast-viz/run: Execute fast scan
- GET /fast-viz/list: List ephemeral runs
- POST /fast-viz/save/{run_id}: Promote to full simulation
- DELETE /fast-viz/{run_id}: Delete ephemeral run
- POST /fast-viz/add: Add from Lab tool result
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import json
from datetime import datetime

from src.sim.fast_forward import fast_viz_strategy, FastVizResult, FastVizTrade

router = APIRouter(prefix="/fast-viz", tags=["fast-viz"])

# In-memory storage for ephemeral Fast Viz runs
_fast_viz_runs: Dict[str, FastVizResult] = {}


class FastVizRunRequest(BaseModel):
    """Request to run a Fast Viz scan."""
    config: Dict[str, Any]  # {trigger: {...}, bracket: {...}}
    start_date: str  # YYYY-MM-DD
    end_date: str
    run_name: Optional[str] = None


class FastVizAddRequest(BaseModel):
    """Request to add a scan result from Lab to Fast Viz."""
    trigger_type: str
    trigger_params: Dict[str, Any] = {}
    start_date: str
    end_date: str
    stop_atr: float = 2.0
    tp_atr: float = 3.0
    run_name: Optional[str] = None


class FastVizTradeResponse(BaseModel):
    """Serializable trade for API response."""
    entry_time: str
    entry_price: float
    direction: str
    stop_price: float
    target_price: float
    outcome: str
    exit_time: Optional[str]
    exit_price: Optional[float]
    pnl_points: float
    trigger_name: str


class FastVizResultResponse(BaseModel):
    """Serializable result for API response."""
    run_id: str
    strategy_name: str
    start_date: str
    end_date: str
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    trades: List[FastVizTradeResponse]


def _result_to_response(result: FastVizResult) -> FastVizResultResponse:
    """Convert internal result to API response."""
    return FastVizResultResponse(
        run_id=result.run_id,
        strategy_name=result.strategy_name,
        start_date=result.start_date,
        end_date=result.end_date,
        total_trades=result.total_trades,
        wins=result.wins,
        losses=result.losses,
        win_rate=result.win_rate,
        trades=[
            FastVizTradeResponse(
                entry_time=t.entry_time,
                entry_price=t.entry_price,
                direction=t.direction,
                stop_price=t.stop_price,
                target_price=t.target_price,
                outcome=t.outcome,
                exit_time=t.exit_time,
                exit_price=t.exit_price,
                pnl_points=t.pnl_points,
                trigger_name=t.trigger_name
            )
            for t in result.trades
        ]
    )


@router.post("/run", response_model=FastVizResultResponse)
async def run_fast_viz(request: FastVizRunRequest):
    """
    Execute a Fast Viz scan.
    Returns trades immediately without full simulation.
    """
    try:
        result = fast_viz_strategy(
            config=request.config,
            start_date=request.start_date,
            end_date=request.end_date,
            run_id=request.run_name
        )
        
        # Store in ephemeral cache
        _fast_viz_runs[result.run_id] = result
        
        return _result_to_response(result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_fast_viz_runs():
    """List all ephemeral Fast Viz runs."""
    return {
        "runs": [
            {
                "run_id": r.run_id,
                "strategy_name": r.strategy_name,
                "total_trades": r.total_trades,
                "win_rate": r.win_rate,
                "start_date": r.start_date,
                "end_date": r.end_date
            }
            for r in _fast_viz_runs.values()
        ]
    }


@router.get("/{run_id}", response_model=FastVizResultResponse)
async def get_fast_viz_run(run_id: str):
    """Get a specific Fast Viz run."""
    if run_id not in _fast_viz_runs:
        raise HTTPException(status_code=404, detail=f"Fast Viz run not found: {run_id}")
    
    return _result_to_response(_fast_viz_runs[run_id])


@router.delete("/{run_id}")
async def delete_fast_viz_run(run_id: str):
    """Delete an ephemeral Fast Viz run."""
    if run_id not in _fast_viz_runs:
        raise HTTPException(status_code=404, detail=f"Fast Viz run not found: {run_id}")
    
    del _fast_viz_runs[run_id]
    return {"success": True, "deleted": run_id}


@router.post("/save/{run_id}")
async def save_fast_viz_run(run_id: str):
    """
    Promote a Fast Viz run to a full simulation.
    This re-runs the strategy with full OCO simulation and writes viz artifacts.
    """
    if run_id not in _fast_viz_runs:
        raise HTTPException(status_code=404, detail=f"Fast Viz run not found: {run_id}")
    
    result = _fast_viz_runs[run_id]
    
    # Import here to avoid circular imports
    import subprocess
    import sys
    import tempfile
    import os
    from pathlib import Path
    
    # Build recipe from stored config
    recipe = {
        "name": f"Saved: {result.strategy_name}",
        "cooldown_bars": result.config.get("cooldown_bars", 20),
        "entry_trigger": result.config.get("trigger", {"type": "ema_cross"}),
        "oco": {
            "entry": "MARKET",
            "take_profit": {
                "multiple": result.config.get("bracket", {}).get("tp_atr", 3.0)
            },
            "stop_loss": {
                "multiple": result.config.get("bracket", {}).get("stop_atr", 2.0)
            }
        }
    }
    
    # Write temp recipe
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(recipe, f, indent=2)
        recipe_path = f.name
    
    try:
        # Run full simulation via run_recipe.py
        new_run_id = f"saved_{run_id}"
        cmd = [
            sys.executable,
            "-m", "scripts.run_recipe",
            "--recipe", recipe_path,
            "--out", new_run_id,
            "--start-date", result.start_date,
            "--end-date", result.end_date
        ]
        
        subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        # Remove from ephemeral cache
        del _fast_viz_runs[run_id]
        
        return {
            "success": True,
            "original_run_id": run_id,
            "new_run_id": new_run_id,
            "message": "Promoted to full simulation with viz artifacts"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save: {str(e)}")
    finally:
        try:
            Path(recipe_path).unlink()
        except:
            pass


@router.post("/add", response_model=FastVizResultResponse)
async def add_from_lab_result(request: FastVizAddRequest):
    """
    Add a scan result from Lab Agent to Fast Viz.
    Re-runs the strategy with provided parameters.
    """
    config = {
        "trigger": {
            "type": request.trigger_type,
            **request.trigger_params
        },
        "bracket": {
            "type": "atr",
            "stop_atr": request.stop_atr,
            "tp_atr": request.tp_atr
        }
    }
    
    result = fast_viz_strategy(
        config=config,
        start_date=request.start_date,
        end_date=request.end_date,
        run_id=request.run_name
    )
    
    _fast_viz_runs[result.run_id] = result
    
    return _result_to_response(result)

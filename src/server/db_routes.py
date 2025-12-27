"""
Experiments DB Routes

API endpoints for interacting with the ExperimentDB.
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel
import shutil
import json

from src.storage import ExperimentDB
from src.config import RESULTS_DIR

router = APIRouter(prefix="/experiments", tags=["experiments"])

class ExperimentResponse(BaseModel):
    run_id: str
    created_at: str
    strategy: str
    config: Dict[str, Any]
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl: float
    avg_pnl_per_trade: float
    sharpe: Optional[float] = None
    max_drawdown: Optional[float] = None
    profit_factor: Optional[float] = None

    # Computed fields
    has_viz: bool = False

@router.get("", response_model=Dict[str, Any])
async def list_experiments(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    sort_by: str = Query("created_at", pattern="^(created_at|win_rate|total_pnl|total_trades)$"),
    sort_desc: bool = Query(True),
    strategy: Optional[str] = Query(None)
):
    """
    List all experiments with pagination and sorting.
    """
    db = ExperimentDB()

    results = db.query_best(
        metric=sort_by,
        strategy=strategy,
        min_trades=0,
        top_k=1000 # Fetch more to support pagination loosely
    )

    # Apply python sorting if needed (though query_best does DESC)
    if not sort_desc:
        results.reverse()

    # Pagination
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    paginated_results = results[start_idx:end_idx]

    # Add has_viz field
    response_items = []
    for r in paginated_results:
        # Check if viz folder exists
        viz_path = RESULTS_DIR / "viz" / r['run_id']
        has_viz = viz_path.exists() and (viz_path / "trades.jsonl").exists()

        item = r.copy()
        item['has_viz'] = has_viz
        response_items.append(item)

    return {
        "items": response_items,
        "total": len(results), # Approximate total since we capped at 1000
        "page": page,
        "limit": limit
    }

@router.get("/{run_id}", response_model=ExperimentResponse)
async def get_experiment(run_id: str):
    """Get details of a specific experiment."""
    db = ExperimentDB()
    record = db.get_run(run_id)
    if not record:
        raise HTTPException(status_code=404, detail="Experiment not found")

    # Check viz
    viz_path = RESULTS_DIR / "viz" / run_id
    has_viz = viz_path.exists() and (viz_path / "trades.jsonl").exists()

    record['has_viz'] = has_viz
    return record

@router.delete("/{run_id}")
async def delete_experiment(run_id: str):
    """Delete an experiment and its artifacts."""
    db = ExperimentDB()

    # Delete from DB
    deleted = db.delete_run(run_id)

    # Delete artifacts
    viz_path = RESULTS_DIR / "viz" / run_id
    if viz_path.exists():
        shutil.rmtree(viz_path)

    # Also check base results dir
    base_path = RESULTS_DIR / run_id
    if base_path.exists():
        shutil.rmtree(base_path)

    if not deleted and not viz_path.exists() and not base_path.exists():
        raise HTTPException(status_code=404, detail="Experiment not found")

    return {"success": True, "message": f"Deleted run {run_id}"}

class VisualizeRequest(BaseModel):
    pass # No params needed, just re-run using stored config

@router.post("/{run_id}/visualize")
async def visualize_experiment(run_id: str):
    """
    Re-run an experiment with visualization enabled.
    This effectively converts a 'Light' run to a 'Viz' run.
    """
    import subprocess
    import sys
    import tempfile

    db = ExperimentDB()
    record = db.get_run(run_id)
    if not record:
        raise HTTPException(status_code=404, detail="Experiment not found")

    config = record.get('config', {})
    recipe = config.get('recipe')
    start_date = config.get('start_date')
    end_date = config.get('end_date')

    if not recipe:
        raise HTTPException(status_code=400, detail="Experiment config missing recipe")

    # Create temp recipe file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(recipe, f, indent=2)
        recipe_path = f.name

    try:
        # Run run_recipe.py WITHOUT --light
        cmd = [
            sys.executable, "-m", "scripts.run_recipe",
            "--recipe", recipe_path,
            "--out", run_id, # Overwrite same ID
            "--start-date", start_date,
            "--end-date", end_date
        ]

        print(f"Re-running for viz: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(RESULTS_DIR.parent) # Assuming cwd is repo root
        )

        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Re-run failed: {result.stderr}")

        return {"success": True, "message": f"Visualization generated for {run_id}"}

    finally:
        # Cleanup
        try:
            from pathlib import Path
            Path(recipe_path).unlink()
        except:
            pass

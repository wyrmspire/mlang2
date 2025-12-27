"""
Experiments API Router
Exposes ExperimentDB functionality to the frontend.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from src.storage.experiments_db import ExperimentDB

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

@router.get("/", response_model=List[ExperimentResponse])
async def list_experiments(
    strategy: Optional[str] = None,
    sort_by: str = Query("created_at", enum=["created_at", "total_pnl", "win_rate"]),
    limit: int = 100
):
    """List experiments from the database."""
    try:
        db = ExperimentDB()
        # db.query_best handles filtering and sorting (descending)
        # Map sort_by: db expects column names.
        # "created_at" isn't a direct metric in query_best usually, but let's check db implementation.
        # query_best allows custom metric.
        
        results = db.query_best(
            metric=sort_by,
            strategy=strategy,
            min_trades=0, # Show all
            top_k=limit
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{run_id}")
async def delete_experiment(run_id: str):
    """Delete an experiment record."""
    try:
        db = ExperimentDB()
        success = db.delete_run(run_id)
        if not success:
             raise HTTPException(status_code=404, detail="Run not found")
        return {"status": "success", "run_id": run_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/summary")
async def experiments_summary():
    """Get aggregate stats by strategy."""
    try:
        db = ExperimentDB()
        return db.list_strategies()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

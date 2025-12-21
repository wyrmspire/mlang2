"""
Simulation API Routes

Interactive simulation endpoints for:
- Starting/stopping sessions
- Stepping through bars
- Updating parameters mid-stream
- Getting session state
"""

from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
import uuid

from src.sim.engine import SimulationEngine
from src.data.loader import load_continuous_contract

router = APIRouter(prefix="/sim", tags=["simulation"])

# Active simulation sessions (in-memory)
_sessions: Dict[str, SimulationEngine] = {}


# =============================================================================
# Request/Response Models
# =============================================================================

class SimStartRequest(BaseModel):
    """Request to start a simulation session."""
    strategy_name: str = "random"  # Strategy name (e.g., 'ifvg_cnn', 'always_long', 'random')
    config: Dict[str, Any] = {}    # Strategy configuration
    start_date: Optional[str] = None  # ISO date string (e.g., '2025-03-17')
    end_date: Optional[str] = None    # ISO date string
    start_idx: Optional[int] = None   # Alternative: use index instead of date
    end_idx: Optional[int] = None


class SimStepRequest(BaseModel):
    """Request to step forward in simulation."""
    n_bars: int = 1  # Number of bars to step


class SimUpdateParamsRequest(BaseModel):
    """Request to update simulation parameters."""
    params: Dict[str, Any]  # Parameters to update


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/start")
async def start_simulation(request: SimStartRequest) -> Dict[str, Any]:
    """
    Start a new simulation session.
    
    Returns:
        session_id: Unique session identifier
        state: Initial session state
    """
    try:
        # Load market data
        df = load_continuous_contract()
        
        # Determine start/end indices
        start_idx = request.start_idx
        end_idx = request.end_idx
        
        if request.start_date:
            # Convert date to index
            start_time = pd.Timestamp(request.start_date)
            if start_time.tzinfo is None:
                start_time = start_time.tz_localize('America/New_York')
            
            # Find first bar >= start_date
            matching_bars = df[df['time'] >= start_time]
            if len(matching_bars) > 0:
                start_idx = matching_bars.index[0]
            else:
                start_idx = 0
        
        if request.end_date:
            # Convert date to index
            end_time = pd.Timestamp(request.end_date)
            if end_time.tzinfo is None:
                end_time = end_time.tz_localize('America/New_York')
            
            # Find last bar <= end_date
            matching_bars = df[df['time'] <= end_time]
            if len(matching_bars) > 0:
                end_idx = matching_bars.index[-1] + 1  # +1 because end_idx is exclusive
            else:
                end_idx = len(df)
        
        # Default to last 1000 bars if no range specified
        if start_idx is None and end_idx is None:
            start_idx = max(0, len(df) - 1000)
            end_idx = len(df)
        elif start_idx is None:
            start_idx = 0
        elif end_idx is None:
            end_idx = len(df)
        
        # Create simulation engine
        engine = SimulationEngine(
            df=df,
            strategy_name=request.strategy_name,
            config=request.config,
            start_idx=start_idx,
            end_idx=end_idx
        )
        
        # Generate session ID
        session_id = str(uuid.uuid4())[:8]
        _sessions[session_id] = engine
        
        return {
            "session_id": session_id,
            "status": "started",
            "strategy": request.strategy_name,
            "config": request.config,
            "data_range": {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "total_bars": end_idx - start_idx
            },
            "state": engine.get_state()
        }
    
    except Exception as e:
        raise HTTPException(500, f"Failed to start simulation: {str(e)}")


@router.post("/{session_id}/step")
async def step_simulation(session_id: str, request: SimStepRequest) -> Dict[str, Any]:
    """
    Step forward n bars in the simulation.
    
    Returns:
        bars: List of bar data
        events: List of all events that occurred
        state: Current session state after stepping
    """
    if session_id not in _sessions:
        raise HTTPException(404, f"Session {session_id} not found")
    
    engine = _sessions[session_id]
    
    results = []
    all_events = []
    
    for _ in range(request.n_bars):
        result = engine.step()
        
        if result.get('done'):
            break
        
        results.append(result)
        all_events.extend(result.get('events', []))
    
    if len(results) == 0:
        return {
            "done": True,
            "bars": [],
            "events": [],
            "state": engine.get_state()
        }
    
    return {
        "done": results[-1].get('done', False),
        "bars": [r['bar'] for r in results],
        "events": all_events,
        "signals": [r['signal'] for r in results],
        "state": engine.get_state()
    }


@router.post("/{session_id}/update_params")
async def update_params(session_id: str, request: SimUpdateParamsRequest) -> Dict[str, Any]:
    """
    Update simulation parameters mid-stream.
    
    Allows changing:
    - entry_type: 'MARKET' or 'LIMIT'
    - stop_atr: Stop distance in ATR
    - tp_multiple: Take profit multiple
    - auto_submit_ocos: Auto-submit OCOs on signals
    """
    if session_id not in _sessions:
        raise HTTPException(404, f"Session {session_id} not found")
    
    engine = _sessions[session_id]
    engine.update_params(request.params)
    
    return {
        "status": "updated",
        "params": request.params,
        "state": engine.get_state()
    }


@router.get("/{session_id}/state")
async def get_state(session_id: str) -> Dict[str, Any]:
    """Get current simulation state."""
    if session_id not in _sessions:
        raise HTTPException(404, f"Session {session_id} not found")
    
    engine = _sessions[session_id]
    return engine.get_state()


@router.post("/{session_id}/stop")
async def stop_simulation(session_id: str) -> Dict[str, Any]:
    """Stop and remove a simulation session."""
    if session_id not in _sessions:
        raise HTTPException(404, f"Session {session_id} not found")
    
    engine = _sessions[session_id]
    final_state = engine.get_state()
    
    # Remove session
    del _sessions[session_id]
    
    return {
        "status": "stopped",
        "session_id": session_id,
        "final_state": final_state
    }


@router.get("/sessions")
async def list_sessions() -> Dict[str, Any]:
    """List all active simulation sessions."""
    return {
        "sessions": [
            {
                "session_id": sid,
                "strategy": engine.strategy.strategy_name,
                "progress": engine.stream.progress,
                "stats": engine.get_state().get('stats', {})
            }
            for sid, engine in _sessions.items()
        ]
    }


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str) -> Dict[str, Any]:
    """Delete a simulation session (alias for stop)."""
    return await stop_simulation(session_id)

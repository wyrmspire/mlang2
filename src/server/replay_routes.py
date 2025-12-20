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


class ReplayControlRequest(BaseModel):
    """Request to control a replay session."""
    action: str  # "pause", "resume", "stop", "speed"
    value: Optional[float] = None  # For speed control


@router.post("/start")
async def start_replay(request: ReplayStartRequest) -> Dict[str, Any]:
    """
    Start a new replay session.
    
    Returns session_id to use for streaming.
    """
    session_id = str(uuid.uuid4())[:8]
    
    # Build command
    cmd = [
        "python", "scripts/run_replay.py",
        "--model", request.model_path,
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

#!/bin/bash
# start.sh - Start mlang2 backend and frontend servers
# Usage: ./start.sh

set -e

echo "=============================================="
echo "  MLang2 - Starting Services"
echo "=============================================="

# Load environment variables
if [ -f src/.env.local ]; then
    echo "[1] Loading src/.env.local..."
    export $(cat src/.env.local | grep -v '^#' | xargs)
elif [ -f .env.local ]; then
    echo "[1] Loading .env.local..."
    export $(cat .env.local | grep -v '^#' | xargs)
else
    echo "[!] Warning: No .env.local found. GEMINI_API_KEY may not be set."
fi

# Kill existing processes aggressively
echo "[2] Killing all existing servers..."

# Detect OS and use appropriate kill commands
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    taskkill //F //IM python.exe 2>/dev/null || true
    taskkill //F //IM pythonw.exe 2>/dev/null || true
    taskkill //F //IM node.exe 2>/dev/null || true
    
    # Also find and kill whatever is holding port 8000/8001 specifically
    for port in 8000 8001; do
        pid=$(netstat -ano | grep ":$port" | grep "LISTEN" | awk '{print $5}' | head -n 1)
        if [ -n "$pid" ]; then
            echo "    Killing process $pid on port $port..."
            taskkill //F //PID $pid 2>/dev/null || true
        fi
    done
else
    # Unix/Linux/Mac
    # Kill uvicorn/FastAPI processes
    pkill -f "uvicorn.*src.server.main" || true
    pkill -f "python.*src.server.main" || true
    
    # Kill vite/npm dev server processes
    pkill -f "vite" || true
    pkill -f "npm.*run.*dev" || true
    
    # Kill any process listening on ports 8000, 8001, 3000
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    lsof -ti:8001 | xargs kill -9 2>/dev/null || true
    lsof -ti:3000 | xargs kill -9 2>/dev/null || true
fi

# Wait for ports to clear
echo "    Waiting for ports to clear..."
sleep 2

# Check if port 8000 is available, fallback to 8001
BACKEND_PORT=8000
if command -v lsof &> /dev/null; then
    # Unix/Linux/Mac with lsof
    if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "    Port 8000 busy, using 8001..."
        BACKEND_PORT=8001
    fi
elif command -v netstat &> /dev/null; then
    # Fallback to netstat
    if netstat -ano 2>/dev/null | grep -q ":8000.*LISTEN\|:8000.*ESTABLISHED\|:8000.*FIN_WAIT"; then
        echo "    Port 8000 busy, using 8001..."
        BACKEND_PORT=8001
    fi
fi

# Start backend
echo "[3] Starting FastAPI backend on port $BACKEND_PORT..."
python -m uvicorn src.server.main:app --host 0.0.0.0 --port $BACKEND_PORT &
BACKEND_PID=$!

# Wait for backend to be ready
echo "    Waiting for backend..."
for i in {1..30}; do
    if curl -s http://localhost:$BACKEND_PORT/health > /dev/null 2>&1; then
        echo "    Backend is ready on port $BACKEND_PORT!"
        break
    fi
    sleep 1
done

# Start frontend
echo "[4] Starting Vite frontend on port 3000..."
npm run dev &
FRONTEND_PID=$!

# Wait for frontend
sleep 2

echo ""
echo "=============================================="
echo "  Services Started!"
echo "  Backend:  http://localhost:$BACKEND_PORT"
echo "  Frontend: http://localhost:3000"
echo "=============================================="
echo ""
echo "Press Ctrl+C to stop all services."
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo "Stopping services..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    
    # Detect OS and use appropriate kill commands
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        taskkill //F //IM python.exe 2>/dev/null || true
        taskkill //F //IM node.exe 2>/dev/null || true
    else
        pkill -f "uvicorn.*src.server.main" || true
        pkill -f "vite" || true
    fi
    exit 0
}

trap cleanup INT TERM

# Wait for Ctrl+C
wait

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

# Kill Python/uvicorn processes
taskkill //F //IM python.exe 2>/dev/null || true
taskkill //F //IM pythonw.exe 2>/dev/null || true

# Kill Node/npm/vite processes  
taskkill //F //IM node.exe 2>/dev/null || true

# Wait for ports to clear
echo "    Waiting for ports to clear..."
sleep 2

# Check if port 8000 is available, fallback to 8001
BACKEND_PORT=8000
if netstat -ano | grep -q ":8000.*LISTEN\|:8000.*ESTABLISHED\|:8000.*FIN_WAIT"; then
    echo "    Port 8000 busy, using 8001..."
    BACKEND_PORT=8001
fi

# Start backend
echo "[3] Starting FastAPI backend on port $BACKEND_PORT..."
python -m uvicorn src.server.main:app --host 0.0.0.0 --port $BACKEND_PORT &
BACKEND_PID=$!

# Wait for backend to be ready
echo "    Waiting for backend..."
for i in {1..10}; do
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
    taskkill //F //IM python.exe 2>/dev/null || true
    taskkill //F //IM node.exe 2>/dev/null || true
    exit 0
}

trap cleanup INT TERM

# Wait for Ctrl+C
wait

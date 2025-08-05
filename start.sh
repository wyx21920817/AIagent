#!/bin/bash

# Default port
PORT=${1:-8888}

# Function to check and kill process using the port
kill_port_process() {
    # Try to kill any process using the port
    if command -v lsof >/dev/null 2>&1; then
        pid=$(lsof -ti:$1)
    else
        # Fallback to ps and grep for uvicorn
        pid=$(ps aux | grep "uvicorn.*:$1" | grep -v grep | awk '{print $2}')
    fi
    
    if [ ! -z "$pid" ]; then
        echo "Port $1 is in use. Killing process $pid..."
        kill -9 $pid 2>/dev/null || true
        echo "Process killed successfully"
    fi
}

echo "Using port: $PORT"

# Kill any process using our port
kill_port_process $PORT

# Kill any remaining uvicorn processes
pkill -f uvicorn

# Wait a moment for the process to be fully killed
sleep 2

# Make script executable
chmod +x start.sh

# Activate virtual environment if exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Check if virtual environment exists and has packages installed
if [ ! -f "venv/lib/python*/site-packages/fastapi" ]; then
    echo "Installing dependencies (first time setup)..."
    pip install -r requirements.txt
fi

# Start the FastAPI server

uvicorn app.main:app --host 0.0.0.0 --port $PORT --reload

# Deactivate virtual environment
if [ -f "venv/bin/deactivate" ]; then
    deactivate
fi

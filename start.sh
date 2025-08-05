#!/bin/bash

# Default port
PORT=8888

# Function to check and kill process on specific port
kill_port_process() {
    # Only look for process on the specified port
    if command -v netstat >/dev/null 2>&1; then
        pid=$(netstat -ano | grep "LISTENING" | grep ":$1" | awk '{print $5}')
    else
        # Fallback to ps and grep
        pid=$(ps aux | grep ":$1" | grep "LISTEN" | grep -v grep | awk '{print $2}')
    fi
    
    if [ ! -z "$pid" ]; then
        echo "Found process $pid using port $1"
        kill -9 $pid 2>/dev/null || true
        echo "Process killed successfully"
    else
        echo "No process found on port $1"
    fi
}

echo "Using port: $PORT"

# Only kill process using our specific port
kill_port_process $PORT

# Wait a moment for the process to be fully killed
sleep 1

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

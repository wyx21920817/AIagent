#!/bin/bash

# Default port
PORT=8888

# Function to check and kill process on specific port
kill_port_process() {
    local port=$1
    local pids=$(lsof -t -i:$port)
    if [ -n "$pids" ]; then
        echo "Killing processes using port $port: $pids"
        for pid in $pids; do
            echo "Killing process $pid"
            kill -9 $pid 2>/dev/null
        done
    else
        echo "No process found using port $port"
    fi
    
    # Double-check if port is still in use
    sleep 1
    if netstat -tuln | grep -q ":$port "; then
        echo "Port $port is still in use. Trying alternative method..."
        # Try alternative method to kill processes
        fuser -k -n tcp $port 2>/dev/null
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

# Create logs and reports directories if not exists
mkdir -p logs
mkdir -p reports

# Start the FastAPI server in background
echo "Starting FastAPI server on port $PORT..."
uvicorn app.main:app --host 0.0.0.0 --port $PORT --reload >> logs/server.log 2>&1 &

# Save PID for later use
echo $! > .server.pid
echo "Server started in background. PID: $(cat .server.pid)"
echo "Logs are being written to logs/server.log"
echo "Use 'tail -f logs/server.log' to follow logs"

# Deactivate virtual environment
if [ -f "venv/bin/deactivate" ]; then
    deactivate
fi

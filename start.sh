#!/bin/bash

# Default port
PORT=${1:-8888}

# Function to get current running port
get_current_port() {
    CURRENT_PORT=$(netstat -tuln | grep "uvicorn" | grep -o ":[0-9]\+" | cut -d':' -f2)
    if [ ! -z "$CURRENT_PORT" ]; then
        echo "Found running server on port $CURRENT_PORT"
        PORT=$CURRENT_PORT
    fi
}

# Function to check and kill process using the port
kill_port_process() {
    if netstat -tuln | grep -q ":$1 "; then
        echo "Port $1 is in use. Killing the process..."
        # Get PID of process using the port and kill it
        pid=$(netstat -tuln | grep ":$1 " | awk '{print $7}' | cut -d'/' -f1)
        if [ ! -z "$pid" ]; then
            kill -9 $pid
            echo "Process killed successfully"
        fi
    fi
}

# Get current running port if no port specified
if [ -z "$1" ]; then
    get_current_port
fi

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

# Install requirements if needed
pip install -r requirements.txt

# Start the FastAPI server
cd app
uvicorn app.main:app --host 0.0.0.0 --port $PORT --reload

# Deactivate virtual environment
if [ -f "venv/bin/deactivate" ]; then
    deactivate
fi

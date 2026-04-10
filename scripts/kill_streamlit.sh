#!/bin/bash

# Kill any running Streamlit processes

echo "Looking for Streamlit processes..."

# Find Streamlit processes
PIDS=$(ps aux | grep '[s]treamlit run' | awk '{print $2}')

if [ -z "$PIDS" ]; then
    echo "No Streamlit processes found"
else
    echo "Found Streamlit processes: $PIDS"
    echo "Killing processes..."
    for PID in $PIDS; do
        kill -9 $PID 2>/dev/null && echo "  Killed process $PID" || echo "  Failed to kill $PID"
    done
fi

# Also check for processes on common Streamlit ports
for PORT in 8501 8502 8503; do
    PID=$(lsof -ti:$PORT 2>/dev/null)
    if [ -n "$PID" ]; then
        echo "Killing process on port $PORT (PID: $PID)"
        kill -9 $PID 2>/dev/null
    fi
done

echo "Done"

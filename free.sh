#!/bin/bash

# Define the port range
START_PORT=8000
END_PORT=8010

# Loop through each port and kill associated processes
for PORT in $(seq $START_PORT $END_PORT); do
    # Find the process ID (PID) using the port
    PID=$(lsof -ti tcp:$PORT)

    # Check if there's a process to kill
    if [ -n "$PID" ]; then
        sudo kill -9 $PID
        echo "✅ Killed process on port $PORT (PID: $PID)"
    else
        echo "ℹ️  No process found on port $PORT"
    fi
done

#!/bin/bash

echo "Stopping all uvicorn processes..."

# Check if .server_pid file exists and use it to kill the specific server process
if [ -f .server_pid ]; then
  SERVER_PID=$(cat .server_pid)
  echo "Found server PID: $SERVER_PID"
  
  if ps -p $SERVER_PID > /dev/null; then
    echo "Killing server process with PID: $SERVER_PID"
    kill $SERVER_PID
    rm .server_pid
  else
    echo "Server process with PID $SERVER_PID is not running"
    rm .server_pid
  fi
fi

# Kill any remaining uvicorn processes
pkill -f uvicorn || echo "No uvicorn processes found"

echo "All uvicorn processes stopped" 
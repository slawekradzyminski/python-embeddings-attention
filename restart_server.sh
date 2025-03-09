#!/bin/bash

# Kill all running uvicorn processes
echo "Stopping any running uvicorn processes..."
pkill -f uvicorn || echo "No uvicorn processes found"

# Start the server in the background
echo "Starting server on port 5000..."
uvicorn app.main:app --host 0.0.0.0 --port 5000 --reload &
SERVER_PID=$!

# Wait for server to start (max 30 seconds)
echo "Waiting for server to start (max 30 seconds)..."
MAX_WAIT=30
WAIT_COUNT=0
SERVER_UP=false

while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
  # Check if server is up by making a request to the health endpoint
  if curl -s http://localhost:5000/health > /dev/null; then
    SERVER_UP=true
    break
  fi
  
  # Check if the server process is still running
  if ! ps -p $SERVER_PID > /dev/null; then
    echo "Server process died unexpectedly"
    exit 1
  fi
  
  echo "Waiting for server to start... ($WAIT_COUNT/$MAX_WAIT)"
  sleep 1
  WAIT_COUNT=$((WAIT_COUNT + 1))
done

if [ "$SERVER_UP" = false ]; then
  echo "Server failed to start within $MAX_WAIT seconds"
  kill $SERVER_PID 2>/dev/null || true
  exit 1
fi

echo "Server started successfully!"

# Verify health endpoint
echo "Verifying health endpoint..."
HEALTH_RESPONSE=$(curl -s http://localhost:5000/health)
if [[ $HEALTH_RESPONSE == *"healthy"* ]]; then
  echo "Health check passed: $HEALTH_RESPONSE"
else
  echo "Health check failed: $HEALTH_RESPONSE"
  kill $SERVER_PID 2>/dev/null || true
  exit 1
fi

# Verify models endpoint
echo "Verifying models endpoint..."
MODELS_RESPONSE=$(curl -s http://localhost:5000/models)
if [[ $MODELS_RESPONSE == *"models"* ]]; then
  echo "Models check passed: $MODELS_RESPONSE"
else
  echo "Models check failed: $MODELS_RESPONSE"
  kill $SERVER_PID 2>/dev/null || true
  exit 1
fi

echo "Server is up and running correctly!"
echo "Server is running with PID: $SERVER_PID"
echo "To stop the server, run: kill $SERVER_PID"

# Create a file with the PID for future reference
echo $SERVER_PID > .server_pid

# Detach the server process from this script
disown $SERVER_PID 
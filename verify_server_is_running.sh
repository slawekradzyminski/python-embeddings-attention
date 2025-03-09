#!/bin/bash

# Script to verify that the server is running
echo "Waiting for server to start (max 60 seconds)..."
MAX_WAIT=60
WAIT_COUNT=0
SERVER_UP=false

while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
  if curl -s http://localhost:5000/health > /dev/null; then
    SERVER_UP=true
    break
  fi
  
  echo "Waiting for server to start... ($WAIT_COUNT/$MAX_WAIT)"
  sleep 1
  WAIT_COUNT=$((WAIT_COUNT + 1))
done

if [ "$SERVER_UP" = false ]; then
  echo "Server failed to start within $MAX_WAIT seconds"
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
  exit 1
fi

# Verify models endpoint
echo "Verifying models endpoint..."
MODELS_RESPONSE=$(curl -s http://localhost:5000/models)
if [[ $MODELS_RESPONSE == *"models"* ]]; then
  echo "Models check passed: $MODELS_RESPONSE (Note: models list may be empty until models are used)"
else
  echo "Models check failed: $MODELS_RESPONSE"
  exit 1
fi

echo "Server verification completed successfully!" 
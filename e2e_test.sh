#!/bin/bash

# Set up exit on error
set -e

# Function to check if a command succeeded
check_status() {
  if [ $? -ne 0 ]; then
    echo "Error: $1 failed"
    if [ -z "$CI" ]; then
      ./kill_server.sh
    fi
    exit 1
  fi
}

# Check if running in CI environment
if [ -z "$CI" ]; then
  # Start the server (only if not in CI)
  echo "Starting server..."
  ./restart_server.sh
  check_status "Server startup"

  # Wait a moment to ensure server is fully ready
  sleep 2
fi

echo "Running end-to-end tests..."

# Test embeddings endpoint
echo "Testing embeddings endpoint..."
EMBEDDINGS_RESPONSE=$(curl -s -X POST http://localhost:5000/embeddings \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a test sentence for embeddings.", "model_name": "gpt2"}')

if [[ $EMBEDDINGS_RESPONSE == *"tokens"* ]] && [[ $EMBEDDINGS_RESPONSE == *"embeddings"* ]]; then
  echo "✅ Embeddings endpoint test passed"
else
  echo "❌ Embeddings endpoint test failed: $EMBEDDINGS_RESPONSE"
  if [ -z "$CI" ]; then
    ./kill_server.sh
  fi
  exit 1
fi

# Test attention endpoint
echo "Testing attention endpoint..."
ATTENTION_RESPONSE=$(curl -s -X POST http://localhost:5000/attention \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a test sentence for attention.", "model_name": "gpt2"}')

if [[ $ATTENTION_RESPONSE == *"tokens"* ]] && [[ $ATTENTION_RESPONSE == *"attention"* ]]; then
  echo "✅ Attention endpoint test passed"
else
  echo "❌ Attention endpoint test failed: $ATTENTION_RESPONSE"
  if [ -z "$CI" ]; then
    ./kill_server.sh
  fi
  exit 1
fi

# Test reduce endpoint
echo "Testing reduce endpoint..."
REDUCE_RESPONSE=$(curl -s -X POST http://localhost:5000/reduce \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a test sentence for dimensionality reduction.", "model_name": "gpt2", "reduction_method": "pca", "n_components": 2}')

if [[ $REDUCE_RESPONSE == *"tokens"* ]] && [[ $REDUCE_RESPONSE == *"reduced_embeddings"* ]]; then
  echo "✅ Reduce endpoint test passed"
else
  echo "❌ Reduce endpoint test failed: $REDUCE_RESPONSE"
  if [ -z "$CI" ]; then
    ./kill_server.sh
  fi
  exit 1
fi

# Check logs to verify everything was processed correctly
echo "Checking logs..."
LOGS_RESPONSE=$(curl -s "http://localhost:5000/logs?lines=50")

if [[ $LOGS_RESPONSE == *"embeddings"* ]] && [[ $LOGS_RESPONSE == *"attention"* ]] && [[ $LOGS_RESPONSE == *"dimensionality reduction"* ]]; then
  echo "✅ Logs verification passed"
else
  echo "❌ Logs verification failed"
  if [ -z "$CI" ]; then
    ./kill_server.sh
  fi
  exit 1
fi

# All tests passed
echo "All end-to-end tests passed successfully!"

# Stop the server (only if not in CI)
if [ -z "$CI" ]; then
  echo "Stopping server..."
  ./kill_server.sh
fi

echo "End-to-end testing completed successfully!" 
# Use Python 3.12 slim as the base image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    TRANSFORMERS_CACHE=/app/.cache/huggingface

# Set the working directory
WORKDIR /app

# Install system dependencies required for some Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app

# Copy utility scripts
COPY restart_server.sh kill_server.sh e2e_test.sh verify_server_is_running.sh ./
RUN chmod +x restart_server.sh kill_server.sh e2e_test.sh verify_server_is_running.sh

# Expose the application port
EXPOSE 5000

# Run FastAPI with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]

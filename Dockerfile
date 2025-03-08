FROM python:3.12-slim

# Create a working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY app/ .
COPY tests/ /app/tests/

# Expose default port for FastAPI
EXPOSE 5000

# Run app with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"] 
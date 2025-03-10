# Python Sidecar for Token Embeddings and Attention Visualization

This service provides a REST API to extract per-token embeddings and multi-head attention data from transformer models for visualization and demonstration purposes.

## Features

- Extract token-level embeddings from transformer models
- Get multi-head attention weights for visualization
- Support for dimensionality reduction (PCA, UMAP)
- Support for both 2D and 3D dimensionality reduction
- Model caching for improved performance
- Simple REST API interface

## Prerequisites

- Python 3.12+
- Docker (for containerized deployment)

## Installation

### Using Docker

1. Build the Docker image:
   ```bash
   docker build -t python-sidecar:latest .
   ```

2. Run the container:
   ```bash
   docker run -p 5000:5000 python-sidecar:latest
   ```

### Using Docker Compose

1. Start the service:
   ```bash
   docker-compose up -d
   ```

2. Stop the service:
   ```bash
   docker-compose down
   ```

### Local Development

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 5000 --reload
   ```
   
   Alternatively, use the provided utility script:
   ```bash
   ./restart_server.sh
   ```

4. To stop the server:
   ```bash
   ./kill_server.sh
   ```

5. To run end-to-end tests:
   ```bash
   ./e2e_test.sh
   ```

## API Endpoints

### POST /embeddings

Process text through a transformer model and return tokens and embeddings.

**Request Body:**
```json
{
  "text": "Hello world, this is a test",
  "model_name": "gpt2"
}
```

**Response:**
```json
{
  "tokens": ["Hello", "world", "this", "is", "a", "test"],
  "embeddings": [[...], [...], ...],
  "model_name": "gpt2"
}
```

### POST /tokenize

Split input text into tokens using the specified tokenizer.

**Request Body:**
```json
{
  "text": "Hello world, this is a test!",
  "model_name": "gpt2"
}
```

**Response:**
```json
{
  "tokens": ["Hello", "world", "this", "is", "a", "test", "!"],
  "model_name": "gpt2"
}
```

### POST /attention

Process text through a transformer model and return tokens and attention weights.

**Request Body:**
```json
{
  "text": "Hello world, this is a test",
  "model_name": "gpt2"
}
```

**Response:**
```json
{
  "tokens": ["Hello", "world", "this", "is", "a", "test"],
  "attention": [[...], [...], ...],
  "model_name": "gpt2"
}
```

### POST /reduce

Process text through a transformer model, get embeddings, and reduce their dimensionality.

**Request Body:**
```json
{
  "text": "Hello world, this is a test",
  "model_name": "gpt2",
  "reduction_method": "pca",
  "n_components": 2
}
```

**Response:**
```json
{
  "tokens": ["Hello", "world", "this", "is", "a", "test"],
  "reduced_embeddings": [[x1, y1], [x2, y2], ...],
  "model_name": "gpt2"
}
```

#### 3D Dimensionality Reduction

For 3D visualization, set `n_components` to 3:

```json
{
  "text": "Hello world, this is a test",
  "model_name": "gpt2",
  "reduction_method": "pca",
  "n_components": 3
}
```

**Response with 3D embeddings:**
```json
{
  "tokens": ["Hello", "world", "this", "is", "a", "test"],
  "reduced_embeddings": [[x1, y1, z1], [x2, y2, z2], ...],
  "model_name": "gpt2"
}
```

### GET /models

List available pre-loaded models.

**Response:**
```json
{
  "models": ["gpt2", "bert-base-uncased", ...]
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```

## API Documentation with Swagger UI

The API is documented using Swagger UI, which provides an interactive interface to explore and test the API endpoints.

### Accessing Swagger UI

Once the service is running, you can access the Swagger UI at:

```
http://localhost:5000/docs
```

This interface allows you to:
- View all available endpoints
- See request and response schemas
- Test endpoints directly from the browser
- Understand parameter requirements

### Example Swagger UI

The Swagger UI provides a clean interface to explore and test the API:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Python Sidecar for Token Embeddings and Attention 1.0.0                │
│                                                                         │
│  API for extracting token-level embeddings and attention from           │
│  transformer models                                                     │
│                                                                         │
│  [ embeddings ]                                                         │
│    POST /embeddings Get token embeddings                                │
│                                                                         │
│  [ tokenize ]                                                             │
│    POST /tokenize Split input text into tokens                           │
│                                                                         │
│  [ attention ]                                                          │
│    POST /attention Get attention weights                                │
│                                                                         │
│  [ reduce ]                                                             │
│    POST /reduce Get dimensionally reduced embeddings                    │
│                                                                         │
│  [ models ]                                                             │
│    GET /models List available models                                    │
│                                                                         │
│  [ health ]                                                             │
│    GET /health Health check                                             │
│                                                                         │
│  [ logs ]                                                               │
│    GET /logs Get recent logs                                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### OpenAPI Specification

The raw OpenAPI specification is available at:

```
http://localhost:5000/openapi.json
```

This JSON file can be imported into API tools like Postman or used for client code generation.

### ReDoc Alternative

An alternative documentation UI is available at:

```
http://localhost:5000/redoc
```

## Testing

Run tests with pytest:

```bash
python -m pytest
```

The test suite includes:
- Health check endpoint testing
- Model listing endpoint testing
- Tests for each endpoint (/embeddings, /attention, /reduce)
- 2D and 3D dimensionality reduction testing

### Utility Scripts

The project includes several utility scripts to help with development and testing:

#### Server Management

- **restart_server.sh**: Starts (or restarts) the server on port 5000
  ```bash
  ./restart_server.sh
  ```
  This script:
  - Kills any running uvicorn processes
  - Starts the server on port 5000
  - Waits until the server is up (max 30 seconds)
  - Verifies the health and models endpoints are working
  - Stores the server PID in `.server_pid` for future reference

- **kill_server.sh**: Stops all uvicorn processes
  ```bash
  ./kill_server.sh
  ```
  This script:
  - Checks for a `.server_pid` file and kills that specific process
  - Kills any remaining uvicorn processes

#### End-to-End Testing

- **e2e_test.sh**: Runs end-to-end tests on all endpoints
  ```bash
  ./e2e_test.sh
  ```
  This script:
  - Starts the server using `restart_server.sh`
  - Tests all domain endpoints (embeddings, attention, reduce) with realistic test data
  - Verifies that responses are successful and make sense
  - Checks the logs to ensure everything was processed correctly
  - Stops the server using `kill_server.sh`

## CI/CD with GitHub Actions

This project includes a GitHub Actions workflow that:
1. Runs tests on every push
2. Builds and tests the Docker image on every push

The workflow configuration is in `.github/workflows/python-app.yml`.

## Docker Compose Integration

Add to your existing docker-compose.yml:

```yaml
services:
  python-sidecar:
    build:
      context: ./python-sidecar
      dockerfile: Dockerfile
    container_name: python-sidecar
    restart: unless-stopped
    ports:
      - "5000:5000"
    networks:
      - my-private-ntwk
```


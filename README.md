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

### POST /process (Deprecated)

The original endpoint that returns tokens, embeddings, attention, and optionally reduced embeddings. This endpoint is deprecated and will be removed in a future version. Please use the new endpoints instead.

**Request Body:**
```json
{
  "text": "Hello world, this is a test",
  "model_name": "gpt2",
  "dimensionality_reduction": true,
  "reduction_method": "pca",
  "n_components": 2
}
```

**Response:**
```json
{
  "tokens": ["Hello", "world", "this", "is", "a", "test"],
  "embeddings": [[...], [...], ...],
  "attention": [[...], [...], ...],
  "reduced_embeddings": [[x1, y1], [x2, y2], ...],
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

## Testing

Run tests with pytest:

```bash
python -m pytest
```

The test suite includes:
- Health check endpoint testing
- Model listing endpoint testing
- Text processing with and without dimensionality reduction
- 2D and 3D dimensionality reduction testing
- Tests for each new endpoint (/embeddings, /attention, /reduce)

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

## Java Client Example

```java
WebClient client = WebClient.builder()
    .baseUrl("http://python-sidecar:5000")
    .build();

// For embeddings only
Map<String, Object> embeddingsRequest = Map.of(
    "text", "Hello world, this is a test",
    "model_name", "gpt2"
);

Map embeddingsResponse = client.post()
    .uri("/embeddings")
    .bodyValue(embeddingsRequest)
    .retrieve()
    .bodyToMono(Map.class)
    .block();

// For attention only
Map<String, Object> attentionRequest = Map.of(
    "text", "Hello world, this is a test",
    "model_name", "gpt2"
);

Map attentionResponse = client.post()
    .uri("/attention")
    .bodyValue(attentionRequest)
    .retrieve()
    .bodyToMono(Map.class)
    .block();

// For dimensionality reduction
Map<String, Object> reduceRequest = Map.of(
    "text", "Hello world, this is a test",
    "model_name", "gpt2",
    "reduction_method", "pca",
    "n_components", 3  // For 3D visualization
);

Map reduceResponse = client.post()
    .uri("/reduce")
    .bodyValue(reduceRequest)
    .retrieve()
    .bodyToMono(Map.class)
    .block();
``` 
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

### POST /process

Process text through a transformer model and return tokens, embeddings, and attention weights.

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

#### 3D Dimensionality Reduction

For 3D visualization, set `n_components` to 3:

```json
{
  "text": "Hello world, this is a test",
  "model_name": "gpt2",
  "dimensionality_reduction": true,
  "reduction_method": "pca",
  "n_components": 3
}
```

**Response with 3D embeddings:**
```json
{
  "reduced_embeddings": [[x1, y1, z1], [x2, y2, z2], ...],
  ...
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

Map<String, Object> requestBody = Map.of(
    "text", "Hello world, this is a test",
    "dimensionality_reduction", true,
    "n_components", 3  // For 3D visualization
);

Map response = client.post()
    .uri("/process")
    .bodyValue(requestBody)
    .retrieve()
    .bodyToMono(Map.class)
    .block();
``` 
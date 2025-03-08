# Python Sidecar for Token Embeddings and Attention Visualization

Below is a step-by-step plan for creating a **Python-based microservice ("sidecar")** that exposes a REST API to return per-token embeddings and multi-head attention data for demonstration or visualization. We'll use **Hugging Face Transformers** plus a small web framework (FastAPI or Flask) inside a Docker container.

---

## 1. Overview

You want a Python service that can:
1. **Accept JSON**: containing input text and optional model configurations.  
2. **Tokenize** the text.  
3. **Run a forward pass** through a pre-trained transformer model (e.g. GPT-2 or DistilBERT) with `output_attentions=True`.  
4. **Collect**:
   - Token strings
   - Hidden states (per-token embeddings)
   - Attention weights
5. **Return** this data as JSON to the caller (your Java backend or any other service).

**Why**: This enables you to incorporate advanced introspection (e.g., attention maps, token embeddings) in your main stack without rewriting everything in Python.

---

## 2. Prerequisites

- **Docker** (to containerize your sidecar)
- **Python 3.12+** recommended
- **PyTorch** (version matching your hardware; CPU is fine for small demos)
- **Transformers library** (from Hugging Face)
- **FastAPI** or **Flask** (for the REST API)
- (Optional) **scikit-learn** or **umap-learn** if you want dimensionality reduction in the sidecar

---

## 3. Directory Structure Example

```
python-sidecar/
├── Dockerfile
├── requirements.txt
└── app/
    ├── main.py
    └── model_service.py
```

Where:
- `Dockerfile`: Docker instructions
- `requirements.txt`: pinned Python dependencies
- `app/main.py`: web framework setup (FastAPI or Flask)
- `app/model_service.py`: logic for loading the model, tokenizing, running forward passes, returning data

---

## 4. Implementation Steps

### 4.1 Create `requirements.txt`
This file lists dependencies. For example:

```
fastapi==0.110.0
uvicorn==0.27.1
transformers==4.38.0
torch==2.2.0
scikit-learn==1.4.0  # if you want PCA
umap-learn==0.5.5    # for UMAP
```

(Adjust versions as needed.)

### 4.2 Write `Dockerfile`
Here's a minimal example:

```dockerfile
FROM python:3.12-slim

# Create a working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY app/ /app

# Expose default port for FastAPI
EXPOSE 5000

# Run app with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
```

Adjust to suit your environment.

### 4.3 Write `app/model_service.py`
This is your internal logic module that loads the Hugging Face model and provides a function to run it. For example:

```python
import torch
from transformers import AutoModel, AutoTokenizer

class ModelService:
    def __init__(self, model_name="gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_attentions=True)
        self.model.eval()

    def get_embeddings_and_attention(self, text: str):
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extract per-token hidden states
        # outputs.last_hidden_state: shape [batch_size, seq_len, hidden_dim]
        hidden_states = outputs.last_hidden_state[0].cpu().numpy()

        # Extract attentions (list of shape [batch_size, num_heads, seq_len, seq_len])
        attentions = [att[0].cpu().numpy().tolist() for att in outputs.attentions]

        # Convert token IDs back to actual tokens
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        return tokens, hidden_states, attentions
```

Note: `outputs.attentions` is a list (one entry per layer). Each entry is shape `[batch_size, n_heads, seq_len, seq_len]`. The example above returns them as nested Python lists for JSON serialization.

### 4.4 Write `app/main.py`
Use FastAPI (similar for Flask). A minimal example:

```python
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import numpy as np

from model_service import ModelService

app = FastAPI()
model_service = ModelService("gpt2")  # or any other model name

class RequestData(BaseModel):
    text: str
    model_name: Optional[str] = None
    do_pca: bool = False  # optional param if you want PCA

@app.post("/process")
def process_text(data: RequestData):
    """Return tokens, embeddings, and attention from model forward pass."""
    # Re-initialize model if user specified a different model_name
    if data.model_name and data.model_name != "gpt2":
        # Possibly switch model or have a dict of loaded models
        pass

    tokens, hidden_states, attentions = model_service.get_embeddings_and_attention(data.text)

    # Optional: do PCA to reduce hidden_states dimension
    # shape: [seq_len, hidden_dim]
    coords_2d = None
    if data.do_pca:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(hidden_states).tolist()  # shape => [seq_len, 2]

    response = {
        "tokens": tokens,
        "embeddings": hidden_states.tolist(),  # [seq_len, hidden_dim]
        "attention": attentions,               # list of shape [num_layers, num_heads, seq_len, seq_len]
        "pca_2d": coords_2d                    # optional
    }
    return response
```

### 4.5 Test Locally

```bash
docker build -t python-sidecar:latest .
docker run -p 5000:5000 python-sidecar:latest
```

Open or curl:

```bash
curl -X POST localhost:5000/process \
     -H "Content-Type: application/json" \
     -d '{"text":"Hello world!", "do_pca":true}'
```

Inspect JSON response.

## 5. Integrate with Docker Compose
In your existing docker-compose.yml, add:

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

Ensure the python-sidecar directory is placed alongside your main compose root.

## 6. Calling the Service from Java
Inside your Java/Spring Boot code:

Construct an HTTP request to `http://python-sidecar:5000/process` (note the Docker Compose service name).
POST JSON with the text you want. For example:

```java
// Pseudocode
WebClient client = WebClient.builder()
    .baseUrl("http://python-sidecar:5000")
    .build();

Map<String, Object> requestBody = Map.of(
    "text", "Hello world, this is a test",
    "do_pca", true
);

Map response = client.post()
    .uri("/process")
    .bodyValue(requestBody)
    .retrieve()
    .bodyToMono(Map.class)
    .block();
```

The response will contain "tokens", "embeddings", "attention", "pca_2d" etc.

## 7. Front-End or Visualization
- If you want to visualize the attention matrix, you can parse `attention[layer][head][i][j]` to highlight how token i attends to j.
- For embeddings (2D from PCA or UMAP), you can create an interactive scatter plot. Each point corresponds to a token, and you can label them with the tokens array.

## 8. Optional Enhancements

### Multiple Models
Caching or switching between gpt2, bert-base-uncased, etc., based on user request.

### GPU Support
If you have a GPU, configure your Dockerfile's base image to be nvidia/cuda.

### Auto-Reload
Use watchfiles or other dev tool for local dev.

### Auth
Add token-based authentication if you want to control usage.

## 9. Summary
By following this plan:

- You containerize a small Python service that can produce token-level embeddings and attention from a standard Hugging Face model.
- Your Java application can make simple REST calls to retrieve the data (tokens, hidden states, attention).
- You can visualize it any way you like in your front-end or in Java itself.

This approach keeps your main stack (Java + Docker) intact while leveraging the Python ecosystem's robust tooling for neural model introspection.
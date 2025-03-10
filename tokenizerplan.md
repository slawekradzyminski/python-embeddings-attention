# Tokenize Endpoint Implementation Plan

This document outlines a step-by-step plan to add a new `/tokenize` endpoint to the Python sidecar. The plan is detailed to ensure reliable implementation.

## 1. Create New Request & Response Models

### Request Model
In `app/models/request.py`, add:

```python
from pydantic import BaseModel, Field
from typing import Optional

class TokenizeRequest(BaseModel):
    text: str = Field(..., description="The input text to split into tokens")
    model_name: Optional[str] = Field("gpt2", description="The name of the transformer model's tokenizer to use (default: gpt2)")
```

### Response Model
In `app/models/response.py`, add:

```python
from pydantic import BaseModel, Field
from typing import List

class TokenizeResponse(BaseModel):
    tokens: List[str] = Field(..., description="List of tokens derived from the input text")
    model_name: str = Field(..., description="Name of the transformer model/tokenizer used")
```

## 2. Update Model Manager and Service

### Model Service Updates
In `app/services/model_service.py`, add to the `ModelService` class:

```python
def tokenize_text(self, text: str) -> List[str]:
    inputs = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")
    token_ids = inputs.input_ids[0].tolist()
    tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
    return tokens
```

> Note: `add_special_tokens=False` prevents insertion of [CLS] or <|endoftext|> tokens. Adjust if special tokens are needed.

### Model Manager Updates
In `app/services/model_manager.py`, add to the `ModelManager` class:

```python
def tokenize_only(self, text: str, model_name: str) -> List[str]:
    model_service = self.get_model(model_name)
    tokens = model_service.tokenize_text(text)
    return tokens
```

## 3. Create the /tokenize Endpoint

Create `app/api/endpoints/tokenize.py`:

```python
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
import uuid
import json

from app.models.request import TokenizeRequest
from app.models.response import TokenizeResponse
from app.api.dependencies import get_model_manager, get_logger
from app.services.model_manager import ModelManager

router = APIRouter()

@router.post("/tokenize", 
             response_model=TokenizeResponse,
             summary="Split text into tokens",
             description="Accepts text and returns the tokenized output from the specified transformer tokenizer.",
             response_description="Tokens from the text and the model name used",
             status_code=200,
             responses={
                 200: {"description": "Successful response with tokens"},
                 400: {"description": "Bad request, invalid model name or parameters"},
                 500: {"description": "Internal server error during processing"}
             })
async def tokenize_text(
    data: TokenizeRequest,
    model_manager: ModelManager = Depends(get_model_manager),
    logger = Depends(get_logger)
) -> TokenizeResponse:
    """
    Split text into tokens using the specified model's tokenizer.
    
    This endpoint tokenizes the input text and returns the tokens, without computing
    embeddings or attention. It relies on the same tokenizer used by the other
    endpoints.
    
    Args:
        data (TokenizeRequest): Request data containing text and model name
        
    Returns:
        TokenizeResponse: Tokens and the model name
    
    Raises:
        HTTPException: If the model name is invalid or there's an error during tokenization
    """
    request_id = str(uuid.uuid4())
    logger.info(f"Request {request_id}: tokenizing text with model {data.model_name}")
    
    try:
        tokens = model_manager.tokenize_only(data.text, data.model_name)
    except ValueError as e:
        error_msg = f"Failed to tokenize with model {data.model_name}: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        error_msg = f"Error tokenizing text: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    
    response = TokenizeResponse(
        tokens=tokens,
        model_name=data.model_name
    )
    
    # Log a summary
    log_response = {
        "tokens_count": len(tokens),
        "model_name": data.model_name
    }
    logger.info(f"Tokenize response summary: {json.dumps(log_response)}")
    
    return response
```

### Register the Router
In `app/api/router.py`, add:

```python
from app.api.endpoints import tokenize

api_router.include_router(tokenize.router, tags=["tokenize"])
```

## 4. Add Tests

Create `tests/test_endpoint_tokenize.py`:

```python
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app
from tests.mock_model_service import MockModelService

client = TestClient(app)

@pytest.fixture(autouse=True)
def mock_model_service(monkeypatch):
    """
    Use the MockModelService to avoid loading real Hugging Face models.
    """
    def mock_init(*args, **kwargs):
        return MockModelService(*args, **kwargs)
    
    monkeypatch.setattr("app.services.model_service.ModelService", mock_init)

def test_tokenize_endpoint_basic():
    """
    Test basic tokenization with the /tokenize endpoint.
    """
    test_data = {
        "text": "Hello world, this is a test.",
        "model_name": "gpt2"
    }
    response = client.post("/tokenize", json=test_data)
    assert response.status_code == 200, f"Status code is {response.status_code}, expected 200"
    
    data = response.json()
    assert "tokens" in data, "Response should contain 'tokens'"
    assert "model_name" in data, "Response should contain 'model_name'"
    assert isinstance(data["tokens"], list), "Expected a list of tokens"
    assert data["model_name"] == "gpt2", "Model name should match 'gpt2'"

@patch("app.services.model_manager.ModelManager.get_model")
def test_tokenize_endpoint_error_handling(mock_get_model):
    """
    Test error handling by forcing an exception in model retrieval.
    """
    mock_get_model.side_effect = ValueError("Failed to load model")

    test_data = {
        "text": "Hello world!",
        "model_name": "broken-model"
    }
    response = client.post("/tokenize", json=test_data)
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "Failed to load model" in data["detail"]
```

### Update Mock Model Service
In `tests/mock_model_service.py`, add:

```python
def tokenize_text(self, text: str):
    # Return a simple token split for demonstration
    return text.split()
```

## 5. Swagger Documentation

The FastAPI Swagger UI will automatically generate documentation from the endpoint decorators and docstrings. No additional work needed.

Access at: `http://localhost:5000/docs`

## 6. Update README.md

Add to the API Endpoints section:

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

## 7. Verify End-to-End

1. **Local Tests**
   ```bash
   pytest
   ```
   Verify all tests pass, including new `test_endpoint_tokenize.py`

2. **Docker Build**
   ```bash
   docker build -t python-sidecar:latest .
   docker run -p 5000:5000 python-sidecar:latest
   ```
   Check `http://localhost:5000/docs` for `/tokenize` endpoint

3. **Optional: End-to-End Test**
   Add to `e2e_test.sh`:
   ```bash
   echo "Testing tokenize endpoint..."
   TOKENIZE_RESPONSE=$(curl -s -X POST http://localhost:5000/tokenize \
     -H "Content-Type: application/json" \
     -d '{"text": "This is a test", "model_name": "gpt2"}')
   echo "Tokenize response: $TOKENIZE_RESPONSE"
   ```

## 8. Summary

This implementation adds a `/tokenize` endpoint with:
- Request/Response models
- Model service updates
- New endpoint implementation
- Comprehensive tests
- Swagger documentation
- README updates

The endpoint aligns with existing project patterns and provides a clean, testable interface for token extraction.
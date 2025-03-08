from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union
import numpy as np

from model_service import ModelService

app = FastAPI(
    title="Embeddings and Attention API",
    description="API for retrieving token embeddings and attention weights from transformer models",
    version="1.0.0"
)

# Initialize with default model
model_service = ModelService("gpt2")

# Cache for models to avoid reloading
model_cache = {"gpt2": model_service}


class RequestData(BaseModel):
    text: str
    model_name: Optional[str] = "gpt2"
    dimensionality_reduction: Optional[bool] = False
    reduction_method: Optional[str] = "pca"
    n_components: Optional[int] = 2  # Default is 2D, but can be set to 3 for 3D visualization


class ResponseData(BaseModel):
    tokens: List[str]
    embeddings: List[List[float]]
    attention: List[List[List[List[float]]]]
    reduced_embeddings: Optional[List[List[float]]] = None
    model_name: str


@app.post("/process", response_model=ResponseData)
def process_text(data: RequestData) -> Dict[str, Any]:
    """
    Process text through a transformer model and return tokens, embeddings, and attention weights.
    
    Args:
        data: Request data containing text and optional parameters
        
    Returns:
        Dictionary with tokens, embeddings, attention weights, and optional reduced embeddings
    """
    # Get or initialize model
    if data.model_name not in model_cache:
        try:
            model_cache[data.model_name] = ModelService(data.model_name)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load model {data.model_name}: {str(e)}")
    
    service = model_cache[data.model_name]
    
    # Process text
    try:
        tokens, hidden_states, attentions = service.get_embeddings_and_attention(data.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")
    
    # Optional dimensionality reduction
    reduced_embeddings = None
    if data.dimensionality_reduction:
        try:
            reduced = service.dimensionality_reduction(
                hidden_states, 
                method=data.reduction_method,
                n_components=data.n_components
            )
            reduced_embeddings = reduced.tolist()
        except Exception as e:
            # Don't fail the whole request if reduction fails
            pass
    
    # Prepare response
    response = {
        "tokens": tokens,
        "embeddings": hidden_states.tolist(),
        "attention": attentions,
        "reduced_embeddings": reduced_embeddings,
        "model_name": data.model_name
    }
    
    return response


@app.get("/models")
def list_available_models() -> Dict[str, List[str]]:
    """
    List available pre-loaded models.
    
    Returns:
        Dictionary with list of model names
    """
    return {"models": list(model_cache.keys())}


@app.get("/health")
def health_check() -> Dict[str, str]:
    """
    Simple health check endpoint.
    
    Returns:
        Status message
    """
    return {"status": "healthy"} 
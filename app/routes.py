from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union
import numpy as np
import json
import uuid
import os

from .model_service import ModelService
from .reduction_service import DimensionalityReducer
from .logging_config import setup_logger

# Get logger
logger = setup_logger()

# Model cache
model_cache: Dict[str, ModelService] = {}

# Create router
router = APIRouter()

# Request and response models
class EmbeddingsRequestData(BaseModel):
    text: str
    model_name: Optional[str] = "gpt2"

class EmbeddingsResponseData(BaseModel):
    tokens: List[str]
    embeddings: List[List[float]]
    model_name: str

class AttentionRequestData(BaseModel):
    text: str
    model_name: Optional[str] = "gpt2"

class AttentionResponseData(BaseModel):
    tokens: List[str]
    attention: List[List[List[List[float]]]]
    model_name: str

class ReduceRequestData(BaseModel):
    text: str
    model_name: Optional[str] = "gpt2"
    reduction_method: Optional[str] = "pca"
    n_components: Optional[int] = 2

class ReduceResponseData(BaseModel):
    tokens: List[str]
    reduced_embeddings: List[List[float]]
    model_name: str

@router.post("/embeddings", response_model=EmbeddingsResponseData)
async def get_embeddings(data: EmbeddingsRequestData) -> Dict[str, Any]:
    """
    Process text through a transformer model and return tokens and embeddings.
    
    Args:
        data: Request data containing text and model name
        
    Returns:
        Dictionary with tokens and embeddings
    """
    request_id = str(uuid.uuid4())
    logger.info(f"Processing text for embeddings with model {data.model_name}")
    
    # Get or initialize model
    if data.model_name not in model_cache:
        try:
            model_cache[data.model_name] = ModelService(data.model_name)
        except Exception as e:
            error_msg = f"Failed to load model {data.model_name}: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
    
    service = model_cache[data.model_name]
    
    # Process text
    try:
        tokens, hidden_states, _ = service.get_embeddings_and_attention(data.text)
    except Exception as e:
        error_msg = f"Error processing text: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    
    # Prepare response
    response = {
        "tokens": tokens,
        "embeddings": hidden_states.tolist(),
        "model_name": data.model_name
    }
    
    # Log a summary of the response
    log_response = {
        "tokens_count": len(tokens),
        "embeddings_shape": list(hidden_states.shape),
        "model_name": data.model_name
    }
    logger.info(f"Embeddings response summary: {json.dumps(log_response)}")
    
    return response

@router.post("/attention", response_model=AttentionResponseData)
async def get_attention(data: AttentionRequestData) -> Dict[str, Any]:
    """
    Process text through a transformer model and return tokens and attention weights.
    
    Args:
        data: Request data containing text and model name
        
    Returns:
        Dictionary with tokens and attention weights
    """
    request_id = str(uuid.uuid4())
    logger.info(f"Processing text for attention with model {data.model_name}")
    
    # Get or initialize model
    if data.model_name not in model_cache:
        try:
            model_cache[data.model_name] = ModelService(data.model_name)
        except Exception as e:
            error_msg = f"Failed to load model {data.model_name}: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
    
    service = model_cache[data.model_name]
    
    # Process text
    try:
        tokens, _, attentions = service.get_embeddings_and_attention(data.text)
    except Exception as e:
        error_msg = f"Error processing text: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    
    # Prepare response
    response = {
        "tokens": tokens,
        "attention": attentions,
        "model_name": data.model_name
    }
    
    # Log a summary of the response
    log_response = {
        "tokens_count": len(tokens),
        "attention_layers": len(attentions),
        "model_name": data.model_name
    }
    logger.info(f"Attention response summary: {json.dumps(log_response)}")
    
    return response

@router.post("/reduce", response_model=ReduceResponseData)
async def reduce_embeddings(data: ReduceRequestData) -> Dict[str, Any]:
    """
    Process text through a transformer model, get embeddings, and reduce their dimensionality.
    
    Args:
        data: Request data containing text, model name, and reduction parameters
        
    Returns:
        Dictionary with tokens and reduced embeddings
    """
    request_id = str(uuid.uuid4())
    logger.info(f"Processing text for dimensionality reduction with model {data.model_name}")
    
    # Get or initialize model
    if data.model_name not in model_cache:
        try:
            model_cache[data.model_name] = ModelService(data.model_name)
        except Exception as e:
            error_msg = f"Failed to load model {data.model_name}: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
    
    service = model_cache[data.model_name]
    
    # Process text
    try:
        tokens, hidden_states, _ = service.get_embeddings_and_attention(data.text)
    except Exception as e:
        error_msg = f"Error processing text: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    
    # Perform dimensionality reduction
    try:
        reducer = DimensionalityReducer(
            method=data.reduction_method,
            n_components=data.n_components
        )
        reduced = reducer.reduce(hidden_states)
        reduced_embeddings = reduced.tolist()
        logger.info(f"Dimensionality reduction successful, shape: {np.array(reduced_embeddings).shape}")
    except Exception as e:
        error_msg = f"Dimensionality reduction failed: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    
    # Prepare response
    response = {
        "tokens": tokens,
        "reduced_embeddings": reduced_embeddings,
        "model_name": data.model_name
    }
    
    # Log a summary of the response
    log_response = {
        "tokens_count": len(tokens),
        "reduced_embeddings_shape": list(np.array(reduced_embeddings).shape),
        "model_name": data.model_name
    }
    logger.info(f"Reduced embeddings response summary: {json.dumps(log_response)}")
    
    return response

@router.get("/models")
async def list_available_models() -> Dict[str, List[str]]:
    """
    List available pre-loaded models.
    
    Returns:
        Dictionary with list of model names
    """
    logger.info(f"Listing available models: {list(model_cache.keys())}")
    return {"models": list(model_cache.keys())}

@router.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Simple health check endpoint.
    
    Returns:
        Status message
    """
    logger.info("Health check requested")
    return {"status": "healthy"}

@router.get("/logs")
async def view_logs(lines: int = 100) -> Dict[str, Any]:
    """
    View the most recent log entries.
    
    Args:
        lines: Number of recent log lines to return
        
    Returns:
        Dictionary with log entries
    """
    log_file = os.path.join("logs", "api.log")
    if not os.path.exists(log_file):
        return {"logs": "No log file found"}
    
    try:
        with open(log_file, "r") as f:
            # Read the last 'lines' lines from the log file
            log_lines = f.readlines()
            log_lines = log_lines[-lines:] if len(log_lines) > lines else log_lines
            return {"logs": "".join(log_lines)}
    except Exception as e:
        logger.error(f"Error reading log file: {str(e)}")
        return {"logs": f"Error reading log file: {str(e)}"} 
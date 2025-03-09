from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
import uuid
import json
import numpy as np

from app.models.request import EmbeddingsRequest
from app.models.response import EmbeddingsResponse
from app.api.dependencies import get_model_manager, get_logger
from app.services.model_manager import ModelManager

router = APIRouter()

@router.post("/embeddings", response_model=EmbeddingsResponse)
async def get_embeddings(
    data: EmbeddingsRequest,
    model_manager: ModelManager = Depends(get_model_manager),
    logger = Depends(get_logger)
) -> Dict[str, Any]:
    """
    Process text through a transformer model and return tokens and embeddings.
    
    Args:
        data: Request data containing text and model name
        
    Returns:
        Dictionary with tokens and embeddings
    """
    request_id = str(uuid.uuid4())
    logger.info(f"Processing text for embeddings with model {data.model_name}")
    
    try:
        tokens, hidden_states = model_manager.get_embeddings(data.text, data.model_name)
    except ValueError as e:
        error_msg = f"Failed to load model {data.model_name}: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)
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
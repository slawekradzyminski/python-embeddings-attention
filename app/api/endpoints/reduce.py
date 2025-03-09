from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
import uuid
import json
import numpy as np

from app.models.request import ReduceRequest
from app.models.response import ReduceResponse
from app.api.dependencies import get_model_manager, get_logger, get_reducer
from app.services.model_manager import ModelManager
from app.services.reduction_service import DimensionalityReducer

router = APIRouter()

@router.post("/reduce", response_model=ReduceResponse)
async def reduce_embeddings(
    data: ReduceRequest,
    model_manager: ModelManager = Depends(get_model_manager),
    logger = Depends(get_logger)
) -> Dict[str, Any]:
    """
    Process text through a transformer model, get embeddings, and reduce their dimensionality.
    
    Args:
        data: Request data containing text, model name, and reduction parameters
        
    Returns:
        Dictionary with tokens and reduced embeddings
    """
    request_id = str(uuid.uuid4())
    logger.info(f"Processing text for dimensionality reduction with model {data.model_name}")
    
    # Get embeddings
    try:
        tokens, hidden_states = model_manager.get_embeddings_for_reduction(data.text, data.model_name)
    except ValueError as e:
        error_msg = f"Failed to load model {data.model_name}: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        error_msg = f"Error processing text: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    
    # Perform dimensionality reduction
    try:
        reducer = get_reducer(method=data.reduction_method, n_components=data.n_components)
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
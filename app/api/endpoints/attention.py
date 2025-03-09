from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
import uuid
import json

from app.models.request import AttentionRequest
from app.models.response import AttentionResponse
from app.api.dependencies import get_model_manager, get_logger
from app.services.model_manager import ModelManager

router = APIRouter()

@router.post("/attention", response_model=AttentionResponse)
async def get_attention(
    data: AttentionRequest,
    model_manager: ModelManager = Depends(get_model_manager),
    logger = Depends(get_logger)
) -> Dict[str, Any]:
    """
    Process text through a transformer model and return tokens and attention weights.
    
    Args:
        data: Request data containing text and model name
        
    Returns:
        Dictionary with tokens and attention weights
    """
    request_id = str(uuid.uuid4())
    logger.info(f"Processing text for attention with model {data.model_name}")
    
    try:
        tokens, attentions = model_manager.get_attention(data.text, data.model_name)
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
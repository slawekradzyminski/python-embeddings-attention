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
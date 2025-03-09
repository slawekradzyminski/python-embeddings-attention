from fastapi import APIRouter, Depends
from typing import Dict, List

from app.models.response import ModelsResponse
from app.api.dependencies import get_model_manager, get_logger
from app.services.model_manager import ModelManager

router = APIRouter()

@router.get("/models", response_model=ModelsResponse,
            summary="List available models",
            description="Get a list of all available transformer models that can be used with the API.",
            response_description="List of available model names",
            status_code=200)
async def get_models(
    model_manager: ModelManager = Depends(get_model_manager),
    logger = Depends(get_logger)
) -> Dict[str, List[str]]:
    """
    List all available transformer models.
    
    Returns a list of model names that are available for use with the embeddings,
    attention, and reduce endpoints.
    
    Returns:
        Dictionary with a list of available model names
    """
    models = model_manager.list_models()
    logger.info(f"Listing available models: {models}")
    return {"models": models} 
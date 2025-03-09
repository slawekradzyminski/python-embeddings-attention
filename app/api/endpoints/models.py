from fastapi import APIRouter, Depends
from typing import Dict, List

from app.models.response import ModelsResponse
from app.api.dependencies import get_model_manager, get_logger
from app.services.model_manager import ModelManager

router = APIRouter()

@router.get("/models", response_model=ModelsResponse)
async def list_available_models(
    model_manager: ModelManager = Depends(get_model_manager),
    logger = Depends(get_logger)
) -> Dict[str, List[str]]:
    """
    List available pre-loaded models.
    
    Returns:
        Dictionary with list of model names
    """
    models = model_manager.list_models()
    logger.info(f"Listing available models: {models}")
    return {"models": models} 
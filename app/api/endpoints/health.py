from fastapi import APIRouter, Depends
from typing import Dict

from app.models.response import HealthResponse
from app.api.dependencies import get_logger

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check(
    logger = Depends(get_logger)
) -> Dict[str, str]:
    """
    Simple health check endpoint.
    
    Returns:
        Status message
    """
    logger.info("Health check requested")
    return {"status": "healthy"} 
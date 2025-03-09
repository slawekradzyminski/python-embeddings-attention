from fastapi import APIRouter, Depends
from typing import Dict

from app.models.response import HealthResponse
from app.api.dependencies import get_logger

router = APIRouter()

@router.get("/health", response_model=HealthResponse,
            summary="Health check",
            description="Check if the service is healthy and running properly.",
            response_description="Health status of the service",
            status_code=200)
async def health_check(
    logger = Depends(get_logger)
) -> Dict[str, str]:
    """
    Perform a health check on the service.
    
    This endpoint verifies that the service is up and running correctly.
    It can be used for monitoring and health checks by load balancers or orchestration systems.
    
    Returns:
        Dictionary with health status
    """
    logger.info("Health check requested")
    return {"status": "healthy"} 
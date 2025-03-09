from fastapi import APIRouter, Depends, Query
from typing import Dict, Any
import os

from app.models.response import LogsResponse
from app.api.dependencies import get_logger

router = APIRouter()

@router.get("/logs", response_model=LogsResponse,
            summary="Get recent logs",
            description="Retrieve recent log entries from the service.",
            response_description="Recent log entries",
            status_code=200,
            responses={
                200: {"description": "Successful response with log entries"},
                500: {"description": "Internal server error while retrieving logs"}
            })
async def get_logs(
    lines: int = Query(100, description="Number of log lines to retrieve", ge=1, le=1000),
    logger = Depends(get_logger)
) -> Dict[str, str]:
    """
    Retrieve recent log entries from the service.
    
    This endpoint returns the most recent log entries from the service's log file.
    It's useful for debugging and monitoring the service's operation.
    
    Args:
        lines: Number of log lines to retrieve (default: 100, max: 1000)
        
    Returns:
        Dictionary with log entries as a string
        
    Raises:
        HTTPException: If log file cannot be read or processed
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
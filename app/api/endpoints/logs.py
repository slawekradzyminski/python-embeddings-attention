from fastapi import APIRouter, Depends
from typing import Dict, Any
import os

from app.models.response import LogsResponse
from app.api.dependencies import get_logger

router = APIRouter()

@router.get("/logs", response_model=LogsResponse)
async def view_logs(
    lines: int = 100,
    logger = Depends(get_logger)
) -> Dict[str, Any]:
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
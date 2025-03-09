from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union
import numpy as np
import logging
import time
import json
from fastapi.middleware.cors import CORSMiddleware
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
import os
from logging.handlers import RotatingFileHandler

from model_service import ModelService

# Configure logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Create a logger
logger = logging.getLogger("api")
logger.setLevel(logging.INFO)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_format)

# Create file handler for detailed logs
file_handler = RotatingFileHandler(
    os.path.join(log_dir, "api.log"),
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
file_handler.setLevel(logging.INFO)
file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_format)

# Add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Custom middleware for request/response logging
class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        
        # Log request method and path
        logger.info(f"Request {request_id} - {request.method} {request.url.path}")
        
        # Log request body for POST/PUT requests
        if request.method in ["POST", "PUT"]:
            try:
                # Create a copy of the request to read the body
                body_bytes = await request.body()
                # Log the request body
                body_str = body_bytes.decode()
                logger.info(f"Request {request_id} body: {body_str}")
                
                # We don't need to modify the request._receive method
                # This was causing the TypeError
            except Exception as e:
                logger.warning(f"Failed to log request body: {str(e)}")
        
        # Process the request and get the response
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Log response status and time
        logger.info(f"Response {request_id} - Status: {response.status_code} - Time: {process_time:.4f}s")
        
        return response

app = FastAPI(
    title="Embeddings and Attention API",
    description="API for retrieving token embeddings and attention weights from transformer models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Add logging middleware
app.add_middleware(LoggingMiddleware)

# Initialize with default model
model_service = ModelService("gpt2")

# Cache for models to avoid reloading
model_cache = {"gpt2": model_service}


class RequestData(BaseModel):
    text: str
    model_name: Optional[str] = "gpt2"
    dimensionality_reduction: Optional[bool] = False
    reduction_method: Optional[str] = "pca"
    n_components: Optional[int] = 2  # Default is 2D, but can be set to 3 for 3D visualization


class ResponseData(BaseModel):
    tokens: List[str]
    embeddings: List[List[float]]
    attention: List[List[List[List[float]]]]
    reduced_embeddings: Optional[List[List[float]]] = None
    model_name: str


@app.post("/process", response_model=ResponseData)
async def process_text(data: RequestData) -> Dict[str, Any]:
    """
    Process text through a transformer model and return tokens, embeddings, and attention weights.
    
    Args:
        data: Request data containing text and optional parameters
        
    Returns:
        Dictionary with tokens, embeddings, attention weights, and optional reduced embeddings
    """
    request_id = str(uuid.uuid4())
    logger.info(f"Processing text with model {data.model_name}, dimensionality reduction: {data.dimensionality_reduction}")
    
    # Get or initialize model
    if data.model_name not in model_cache:
        try:
            model_cache[data.model_name] = ModelService(data.model_name)
        except Exception as e:
            error_msg = f"Failed to load model {data.model_name}: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
    
    service = model_cache[data.model_name]
    
    # Process text
    try:
        tokens, hidden_states, attentions = service.get_embeddings_and_attention(data.text)
    except Exception as e:
        error_msg = f"Error processing text: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    
    # Optional dimensionality reduction
    reduced_embeddings = None
    if data.dimensionality_reduction:
        try:
            reduced = service.dimensionality_reduction(
                hidden_states, 
                method=data.reduction_method,
                n_components=data.n_components
            )
            reduced_embeddings = reduced.tolist()
            logger.info(f"Dimensionality reduction successful, shape: {np.array(reduced_embeddings).shape}")
        except Exception as e:
            # Don't fail the whole request if reduction fails
            logger.warning(f"Dimensionality reduction failed: {str(e)}")
            pass
    
    # Prepare response
    response = {
        "tokens": tokens,
        "embeddings": hidden_states.tolist(),
        "attention": attentions,
        "reduced_embeddings": reduced_embeddings,
        "model_name": data.model_name
    }
    
    # Log a summary of the response (not the full embeddings which are too large)
    log_response = {
        "tokens_count": len(tokens),
        "embeddings_shape": list(hidden_states.shape),
        "attention_layers": len(attentions),
        "reduced_embeddings": reduced_embeddings if reduced_embeddings is not None else None,
        "model_name": data.model_name
    }
    logger.info(f"Response summary: {json.dumps(log_response)}")
    
    return response


@app.get("/models")
async def list_available_models() -> Dict[str, List[str]]:
    """
    List available pre-loaded models.
    
    Returns:
        Dictionary with list of model names
    """
    logger.info(f"Listing available models: {list(model_cache.keys())}")
    return {"models": list(model_cache.keys())}


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Simple health check endpoint.
    
    Returns:
        Status message
    """
    logger.info("Health check requested")
    return {"status": "healthy"}


@app.get("/logs")
async def view_logs(lines: int = 100) -> Dict[str, Any]:
    """
    View the most recent log entries.
    
    Args:
        lines: Number of recent log lines to return
        
    Returns:
        Dictionary with log entries
    """
    log_file = os.path.join(log_dir, "api.log")
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
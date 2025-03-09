from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware

from .logging_config import setup_logger
from .routes import router

# Set up logger
logger = setup_logger()

# Create FastAPI app
app = FastAPI(
    title="Python Sidecar for Token Embeddings and Attention",
    description="API for extracting token-level embeddings and attention from transformer models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom middleware for request/response logging
class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        
        # Log request method and path
        logger.info(f"Request {request_id}: {request.method} {request.url.path}")
        
        # Record request start time
        start_time = time.time()
        
        # Process the request
        try:
            response = await call_next(request)
            
            # Calculate and log processing time
            process_time = time.time() - start_time
            logger.info(f"Response {request_id}: status={response.status_code}, time={process_time:.4f}s")
            
            # Add custom header with processing time
            response.headers["X-Process-Time"] = str(process_time)
            return response
            
        except Exception as e:
            # Log any exceptions
            process_time = time.time() - start_time
            logger.error(f"Error {request_id}: {str(e)}, time={process_time:.4f}s")
            raise

# Add logging middleware
app.add_middleware(LoggingMiddleware)

# Include router
app.include_router(router) 
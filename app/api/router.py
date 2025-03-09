from fastapi import APIRouter
from app.api.endpoints import embeddings, attention, reduce, models, health, logs

api_router = APIRouter()

api_router.include_router(embeddings.router, tags=["embeddings"])
api_router.include_router(attention.router, tags=["attention"])
api_router.include_router(reduce.router, tags=["reduce"])
api_router.include_router(models.router, tags=["models"])
api_router.include_router(health.router, tags=["health"])
api_router.include_router(logs.router, tags=["logs"]) 
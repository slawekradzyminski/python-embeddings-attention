from fastapi import Depends
from app.services.model_manager import ModelManager
from app.services.reduction_service import DimensionalityReducer
from app.core.logging_config import setup_logger

# Singleton instances
model_manager = ModelManager()
logger = setup_logger()

def get_model_manager() -> ModelManager:
    """Dependency for getting the model manager instance"""
    return model_manager

def get_logger():
    """Dependency for getting the logger instance"""
    return logger

def get_reducer(method: str = "pca", n_components: int = 2) -> DimensionalityReducer:
    """Dependency for getting a dimensionality reducer with specified parameters"""
    return DimensionalityReducer(method=method, n_components=n_components) 
from typing import Dict, Any, Tuple, List
import numpy as np
from app.services.model_service import ModelService

class ModelManager:
    def __init__(self):
        self.model_cache: Dict[str, ModelService] = {}
    
    def get_model(self, model_name: str) -> ModelService:
        """Get or initialize a model by name"""
        if model_name not in self.model_cache:
            self.model_cache[model_name] = ModelService(model_name)
        return self.model_cache[model_name]
    
    def get_embeddings(self, text: str, model_name: str) -> Tuple[List[str], np.ndarray]:
        """Get tokens and embeddings for text using specified model"""
        model = self.get_model(model_name)
        tokens, embeddings, _ = model.get_embeddings_and_attention(text)
        return tokens, embeddings
    
    def get_attention(self, text: str, model_name: str) -> Tuple[List[str], List[Any]]:
        """Get tokens and attention weights for text using specified model"""
        model = self.get_model(model_name)
        tokens, _, attention = model.get_embeddings_and_attention(text)
        return tokens, attention
    
    def get_embeddings_for_reduction(self, text: str, model_name: str) -> Tuple[List[str], np.ndarray]:
        """Get tokens and embeddings for dimensionality reduction"""
        return self.get_embeddings(text, model_name)
    
    def list_models(self) -> List[str]:
        """List all available models"""
        return list(self.model_cache.keys()) 
from pydantic import BaseModel, Field
from typing import List, Optional

class EmbeddingsResponse(BaseModel):
    tokens: List[str] = Field(..., description="List of tokens from the input text after tokenization")
    embeddings: List[List[float]] = Field(..., description="Token embeddings (hidden states) from the model, shape: [num_tokens, embedding_dim]")
    model_name: str = Field(..., description="Name of the transformer model used")

class AttentionResponse(BaseModel):
    tokens: List[str] = Field(..., description="List of tokens from the input text after tokenization")
    attention: List[List[List[List[float]]]] = Field(..., description="Attention weights from all layers and heads, shape: [num_layers, num_heads, num_tokens, num_tokens]")
    model_name: str = Field(..., description="Name of the transformer model used")

class ReduceResponse(BaseModel):
    tokens: List[str] = Field(..., description="List of tokens from the input text after tokenization")
    reduced_embeddings: List[List[float]] = Field(..., description="Dimensionally reduced token embeddings, shape: [num_tokens, n_components]")
    model_name: str = Field(..., description="Name of the transformer model used")

class ModelsResponse(BaseModel):
    models: List[str] = Field(..., description="List of available transformer models")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status of the service")

class LogsResponse(BaseModel):
    logs: str = Field(..., description="Recent log entries from the service") 
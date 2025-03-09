from pydantic import BaseModel
from typing import List, Optional

class EmbeddingsResponse(BaseModel):
    tokens: List[str]
    embeddings: List[List[float]]
    model_name: str

class AttentionResponse(BaseModel):
    tokens: List[str]
    attention: List[List[List[List[float]]]]
    model_name: str

class ReduceResponse(BaseModel):
    tokens: List[str]
    reduced_embeddings: List[List[float]]
    model_name: str

class ModelsResponse(BaseModel):
    models: List[str]

class HealthResponse(BaseModel):
    status: str

class LogsResponse(BaseModel):
    logs: str 
from pydantic import BaseModel
from typing import Optional

class EmbeddingsRequest(BaseModel):
    text: str
    model_name: Optional[str] = "gpt2"

class AttentionRequest(BaseModel):
    text: str
    model_name: Optional[str] = "gpt2"

class ReduceRequest(BaseModel):
    text: str
    model_name: Optional[str] = "gpt2"
    reduction_method: Optional[str] = "pca"
    n_components: Optional[int] = 2 
from pydantic import BaseModel, Field
from typing import Optional

class EmbeddingsRequest(BaseModel):
    text: str = Field(..., description="The input text to process through the model")
    model_name: Optional[str] = Field("gpt2", description="The name of the transformer model to use (default: gpt2)")

class AttentionRequest(BaseModel):
    text: str = Field(..., description="The input text to process through the model")
    model_name: Optional[str] = Field("gpt2", description="The name of the transformer model to use (default: gpt2)")

class ReduceRequest(BaseModel):
    text: str = Field(..., description="The input text to process through the model")
    model_name: Optional[str] = Field("gpt2", description="The name of the transformer model to use (default: gpt2)")
    reduction_method: Optional[str] = Field("pca", description="Dimensionality reduction method to use: 'pca' or 'umap' (default: pca)")
    n_components: Optional[int] = Field(2, description="Number of dimensions to reduce to, typically 2 or 3 (default: 2)") 
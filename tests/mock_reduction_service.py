import numpy as np
from typing import Optional

class MockDimensionalityReducer:
    """
    Mock implementation of DimensionalityReducer for testing.
    """
    
    def __init__(self, method: str = "pca", n_components: int = 2):
        self.method = method
        self.n_components = n_components
    
    def reduce(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Mock dimensionality reduction that returns random coordinates.
        
        Args:
            embeddings: Token embeddings
            
        Returns:
            Random coordinates with the specified number of dimensions
        """
        # Ensure we have enough data points for reduction
        seq_len, hidden_dim = embeddings.shape
        if seq_len < self.n_components:
            raise ValueError(f"Cannot reduce {seq_len} tokens to {self.n_components} dimensions. Not enough tokens.")
            
        # Check for valid method
        if self.method not in ["pca", "umap"]:
            raise ValueError(f"Unsupported dimensionality reduction method: {self.method}")
            
        # Return random values in the range [-1, 1]
        return np.random.uniform(-1, 1, (seq_len, self.n_components)).astype(np.float32) 
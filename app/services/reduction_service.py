import numpy as np
from typing import Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from umap import UMAP

class DimensionalityReducer:
    def __init__(self, method: str = "pca", n_components: int = 2):
        """
        Initialize the dimensionality reducer.
        
        Args:
            method: Reduction method ('pca' or 'umap')
            n_components: Number of dimensions to reduce to
        """
        self.method = method
        self.n_components = n_components
        
    def reduce(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Reduce dimensionality of embeddings for visualization.
        
        Args:
            embeddings: Token embeddings to reduce
            
        Returns:
            Reduced embeddings as NumPy array, normalized to range [-1, 1]
        """
        # Ensure we have enough data points for reduction
        seq_len, hidden_dim = embeddings.shape
        if seq_len < self.n_components:
            raise ValueError(f"Cannot reduce {seq_len} tokens to {self.n_components} dimensions. Not enough tokens.")

        # Standardize embeddings (zero mean, unit variance)
        scaler = StandardScaler()
        embeddings_normalized = scaler.fit_transform(embeddings)

        # Apply PCA or UMAP
        if self.method == "pca":
            reducer = PCA(n_components=self.n_components)
            reduced = reducer.fit_transform(embeddings_normalized)
        elif self.method == "umap":
            reducer = UMAP(n_components=self.n_components)
            reduced = reducer.fit_transform(embeddings_normalized)
        else:
            raise ValueError(f"Unsupported dimensionality reduction method: {self.method}")
        
        # Normalize the reduced embeddings to range [-1, 1]
        normalizer = MinMaxScaler(feature_range=(-1, 1))
        reduced = normalizer.fit_transform(reduced)

        return reduced 
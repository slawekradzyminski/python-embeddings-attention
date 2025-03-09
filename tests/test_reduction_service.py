import pytest
import numpy as np
from app.reduction_service import DimensionalityReducer

class TestDimensionalityReducer:
    def test_pca_2d_reduction(self):
        # given
        embeddings = np.random.rand(10, 768)  # 10 tokens, 768 dimensions
        reducer = DimensionalityReducer(method="pca", n_components=2)
        
        # when
        reduced = reducer.reduce(embeddings)
        
        # then
        assert reduced.shape == (10, 2)
        assert np.all(reduced >= -1.001) and np.all(reduced <= 1.001)  # Check normalization to [-1, 1] with small tolerance
    
    def test_pca_3d_reduction(self):
        # given
        embeddings = np.random.rand(10, 768)  # 10 tokens, 768 dimensions
        reducer = DimensionalityReducer(method="pca", n_components=3)
        
        # when
        reduced = reducer.reduce(embeddings)
        
        # then
        assert reduced.shape == (10, 3)
        assert np.all(reduced >= -1.001) and np.all(reduced <= 1.001)  # Check normalization to [-1, 1] with small tolerance
    
    def test_umap_2d_reduction(self):
        # given
        embeddings = np.random.rand(10, 768)  # 10 tokens, 768 dimensions
        reducer = DimensionalityReducer(method="umap", n_components=2)
        
        # when
        reduced = reducer.reduce(embeddings)
        
        # then
        assert reduced.shape == (10, 2)
        assert np.all(reduced >= -1.001) and np.all(reduced <= 1.001)  # Check normalization to [-1, 1] with small tolerance
    
    def test_invalid_method(self):
        # given
        embeddings = np.random.rand(10, 768)  # 10 tokens, 768 dimensions
        reducer = DimensionalityReducer(method="invalid_method", n_components=2)
        
        # when/then
        with pytest.raises(ValueError, match="Unsupported dimensionality reduction method"):
            reducer.reduce(embeddings)
    
    def test_not_enough_tokens(self):
        # given
        embeddings = np.random.rand(2, 768)  # 2 tokens, 768 dimensions
        reducer = DimensionalityReducer(method="pca", n_components=3)
        
        # when/then
        with pytest.raises(ValueError, match="Cannot reduce 2 tokens to 3 dimensions"):
            reducer.reduce(embeddings) 
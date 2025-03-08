import pytest
import numpy as np
from tests.mock_model_service import MockModelService


class TestModelService:
    """Tests for the ModelService class using the mock implementation."""
    
    # given
    @pytest.fixture
    def model_service(self):
        return MockModelService()
    
    def test_get_embeddings_and_attention(self, model_service):
        # given
        text = "Hello world this is a test"
        
        # when
        tokens, embeddings, attentions = model_service.get_embeddings_and_attention(text)
        
        # then
        # Check tokens
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(token, str) for token in tokens)
        
        # Check embeddings
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(tokens)  # One embedding per token
        assert embeddings.shape[1] > 0  # Embedding dimension
        
        # Check attentions
        assert isinstance(attentions, list)
        assert len(attentions) > 0  # At least one layer
        
        # Check first layer's attention
        layer_attention = attentions[0]
        assert isinstance(layer_attention, list)
        assert len(layer_attention) > 0  # At least one attention head
        
        # Check first head's attention
        head_attention = layer_attention[0]
        assert isinstance(head_attention, list)
        assert len(head_attention) == len(tokens)  # Attention from each token
        assert len(head_attention[0]) == len(tokens)  # Attention to each token
    
    def test_dimensionality_reduction(self, model_service):
        # given
        # Create sample embeddings
        num_tokens = 5
        embedding_dim = 16
        embeddings = np.random.randn(num_tokens, embedding_dim).astype(np.float32)
        
        # when
        # Test PCA reduction
        reduced_pca = model_service.dimensionality_reduction(embeddings, method="pca", n_components=2)
        
        # then
        assert isinstance(reduced_pca, np.ndarray)
        assert reduced_pca.shape == (num_tokens, 2)
        
        # when
        # Test with 3 components
        reduced_3d = model_service.dimensionality_reduction(embeddings, method="pca", n_components=3)
        
        # then
        assert reduced_3d.shape == (num_tokens, 3) 
import numpy as np
from typing import List, Tuple, Any


class MockModelService:
    """
    Mock implementation of ModelService for testing without loading actual models.
    """
    
    def __init__(self, model_name: str = "mock-gpt2"):
        self.model_name = model_name
    
    def get_embeddings_and_attention(self, text: str) -> Tuple[List[str], np.ndarray, List[Any]]:
        """
        Return mock tokens, embeddings, and attention for the given text.
        
        Args:
            text: Input text
            
        Returns:
            Mock tokens, embeddings, and attention
        """
        # Split text into mock tokens
        words = text.split()
        tokens = []
        for word in words:
            # Simulate subword tokenization by splitting longer words
            if len(word) > 5:
                tokens.extend([word[:3], word[3:]])
            else:
                tokens.append(word)
        
        # Create mock embeddings (random values)
        embedding_dim = 16  # Small dimension for testing
        embeddings = np.random.randn(len(tokens), embedding_dim).astype(np.float32)
        
        # Create mock attention (random values)
        num_layers = 2
        num_heads = 4
        attentions = []
        
        for _ in range(num_layers):
            layer_attention = np.random.rand(num_heads, len(tokens), len(tokens)).tolist()
            attentions.append(layer_attention)
        
        return tokens, embeddings, attentions 

    def tokenize_text(self, text: str) -> List[str]:
        """
        Return mock tokens for the given text.
        
        Args:
            text: Input text
            
        Returns:
            Mock tokens
        """
        # Split text into mock tokens
        words = text.split()
        tokens = []
        for word in words:
            # Simulate subword tokenization by splitting longer words
            if len(word) > 5:
                tokens.extend([word[:3], word[3:]])
            else:
                tokens.append(word)
        return tokens 
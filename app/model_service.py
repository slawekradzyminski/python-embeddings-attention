import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from typing import Dict, List, Tuple, Any, Optional


class ModelService:
    def __init__(self, model_name: str = "gpt2"):
        """
        Initialize the model service with a specified model.
        
        Args:
            model_name: Name of the Hugging Face model to use
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_attentions=True)
        self.model.eval()
    
    def get_embeddings_and_attention(self, text: str) -> Tuple[List[str], np.ndarray, List[Any]]:
        """
        Process text through the model and return tokens, embeddings, and attention.
        
        Args:
            text: Input text to process
            
        Returns:
            Tuple containing:
            - List of token strings
            - NumPy array of token embeddings
            - List of attention matrices
        """
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt")
        
        # Run model inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Extract per-token hidden states
        # Shape: [batch_size, seq_len, hidden_dim]
        hidden_states = outputs.last_hidden_state[0].cpu().numpy()
        
        # Extract attentions
        # List of tensors with shape [batch_size, num_heads, seq_len, seq_len]
        attentions = [att[0].cpu().numpy().tolist() for att in outputs.attentions]
        
        # Convert token IDs back to actual tokens
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        return tokens, hidden_states, attentions
    
    def dimensionality_reduction(self, embeddings: np.ndarray, method: str = "pca", 
                                n_components: int = 2) -> np.ndarray:
        """
        Reduce dimensionality of embeddings for visualization.
        
        Args:
            embeddings: Token embeddings to reduce
            method: Reduction method ('pca' or 'umap')
            n_components: Number of dimensions to reduce to
            
        Returns:
            Reduced embeddings as NumPy array
        """
        if method == "pca":
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=n_components)
        elif method == "umap":
            from umap import UMAP
            reducer = UMAP(n_components=n_components)
        else:
            raise ValueError(f"Unsupported dimensionality reduction method: {method}")
        
        return reducer.fit_transform(embeddings) 
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from typing import List, Tuple, Any, Dict, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from umap import UMAP

# Global caches to avoid reloading models and tokenizers
_MODEL_CACHE: Dict[str, Any] = {}
_TOKENIZER_CACHE: Dict[str, Any] = {}

class ModelService:
    def __init__(self, model_name: str = "gpt2"):
        """
        Initialize the model service with a specified model.
        
        Args:
            model_name: Name of the Hugging Face model to use
        """
        self.model_name = model_name
        
        # Use cached tokenizer if available
        if model_name not in _TOKENIZER_CACHE:
            _TOKENIZER_CACHE[model_name] = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer = _TOKENIZER_CACHE[model_name]
        
        # Use cached model if available
        if model_name not in _MODEL_CACHE:
            _MODEL_CACHE[model_name] = AutoModel.from_pretrained(model_name, output_attentions=True)
            _MODEL_CACHE[model_name].eval()  # Set to evaluation mode
        self.model = _MODEL_CACHE[model_name]
    
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
        hidden_states = outputs.last_hidden_state[0].cpu().numpy()  # Shape: [seq_len, hidden_dim]
        
        # Extract attentions
        attentions = [att[0].cpu().numpy().tolist() for att in outputs.attentions]  # [layers, heads, seq_len, seq_len]

        # Convert token IDs back to actual tokens
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        return tokens, hidden_states, attentions
    
    def dimensionality_reduction(self, embeddings: np.ndarray, method: str = "pca", n_components: int = 2) -> np.ndarray:
        """
        Reduce dimensionality of embeddings for visualization.
        
        Args:
            embeddings: Token embeddings to reduce
            method: Reduction method ('pca' or 'umap')
            n_components: Number of dimensions to reduce to
            
        Returns:
            Reduced embeddings as NumPy array, normalized to range [-1, 1]
        """
        # Ensure we have enough data points for reduction
        seq_len, hidden_dim = embeddings.shape
        if seq_len < n_components:
            raise ValueError(f"Cannot reduce {seq_len} tokens to {n_components} dimensions. Not enough tokens.")

        # Standardize embeddings (zero mean, unit variance)
        scaler = StandardScaler()
        embeddings_normalized = scaler.fit_transform(embeddings)

        # Apply PCA or UMAP
        try:
            if method == "pca":
                reducer = PCA(n_components=n_components)
                reduced = reducer.fit_transform(embeddings_normalized)
            elif method == "umap":
                reducer = UMAP(n_components=n_components)
                reduced = reducer.fit_transform(embeddings_normalized)
            else:
                raise ValueError(f"Unsupported dimensionality reduction method: {method}")
            
            # Normalize the reduced embeddings to range [-1, 1]
            normalizer = MinMaxScaler(feature_range=(-1, 1))
            reduced = normalizer.fit_transform(reduced)

        except Exception as e:
            print(f"[Error] Dimensionality reduction failed: {str(e)}. Returning zeros.")
            reduced = np.zeros((seq_len, n_components))  # Return safe default

        return reduced

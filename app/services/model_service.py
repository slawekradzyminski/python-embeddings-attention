import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from typing import List, Tuple, Any, Dict, Optional

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
            outputs = self.model(**inputs, output_attentions=True)
        
        # Get token embeddings (last hidden state)
        hidden_states = outputs.last_hidden_state[0].cpu().numpy()  # Remove batch dimension
        
        # Get attention weights
        # Format: list of tensors, one per layer, each with shape [batch_size, num_heads, seq_len, seq_len]
        attentions = [layer[0].cpu().numpy().tolist() for layer in outputs.attentions]  # Remove batch dimension
        
        # Get token strings
        token_ids = inputs.input_ids[0].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        
        return tokens, hidden_states, attentions

    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text using the model's tokenizer without computing embeddings or attention.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of token strings
        """
        inputs = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")
        token_ids = inputs.input_ids[0].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        return tokens

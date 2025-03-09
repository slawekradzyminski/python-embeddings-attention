import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from app.services.model_service import ModelService


class TestModelService:
    """Tests for the ModelService class using the mock implementation."""
    
    def test_get_embeddings_and_attention(self):
        # given
        # Mock the tokenizer and model
        with patch("app.services.model_service._TOKENIZER_CACHE", {}), \
             patch("app.services.model_service._MODEL_CACHE", {}), \
             patch("app.services.model_service.AutoTokenizer") as mock_tokenizer, \
             patch("app.services.model_service.AutoModel") as mock_model:
            
            # Set up mock tokenizer
            mock_tokenizer_instance = MagicMock()
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            # Mock tokenizer behavior
            mock_inputs = MagicMock()
            mock_inputs.input_ids = MagicMock()
            mock_inputs.input_ids.__getitem__.return_value = MagicMock()
            mock_inputs.input_ids.__getitem__().tolist.return_value = [0, 1]
            mock_tokenizer_instance.return_value = mock_inputs
            mock_tokenizer_instance.convert_ids_to_tokens.return_value = ["Hello", "world"]
            
            # Set up mock model
            mock_model_instance = MagicMock()
            mock_model.from_pretrained.return_value = mock_model_instance
            
            # Mock model output
            mock_output = MagicMock()
            
            # Mock last_hidden_state as a tensor that can be converted to numpy
            mock_tensor = MagicMock()
            mock_tensor.cpu.return_value = mock_tensor
            mock_tensor.numpy.return_value = np.random.rand(2, 768)
            mock_output.last_hidden_state = MagicMock()
            mock_output.last_hidden_state.__getitem__.return_value = mock_tensor
            
            # Mock attentions as a list of tensors
            mock_attention_tensor = MagicMock()
            mock_attention_tensor.__getitem__.return_value = mock_attention_tensor
            mock_attention_tensor.cpu.return_value = mock_attention_tensor
            
            # Create a numpy array and convert it to a list
            attention_array = np.random.rand(4, 2, 2)
            attention_list = attention_array.tolist()
            mock_attention_tensor.numpy.return_value = attention_array
            
            # Mock the model to return the attention tensor
            mock_output.attentions = [mock_attention_tensor]
            mock_model_instance.return_value = mock_output
            
            # Create the model service
            service = ModelService("gpt2")
            
            # when
            tokens, hidden_states, attentions = service.get_embeddings_and_attention("Hello world")
            
            # then
            assert len(tokens) == 2
            assert tokens == ["Hello", "world"]
            assert hidden_states.shape[0] == 2  # Two tokens
            assert hidden_states.shape[1] == 768  # Hidden dimension
            assert len(attentions) == 1  # One layer of attention 
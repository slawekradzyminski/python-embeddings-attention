import pytest
from fastapi.testclient import TestClient
import numpy as np
import json
import os
from unittest.mock import patch, MagicMock

# Import app directly from app.main
from app.main import app

# Create test client
client = TestClient(app)

def test_health_check():
    # given
    # when
    response = client.get("/health")
    
    # then
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_list_models():
    # given
    # when
    response = client.get("/models")
    
    # then
    assert response.status_code == 200
    assert "models" in response.json()
    assert isinstance(response.json()["models"], list)

@patch("app.routes.ModelService")
def test_process_text_without_reduction(mock_model_service):
    # given
    # Mock the model service
    mock_instance = MagicMock()
    mock_model_service.return_value = mock_instance
    
    # Mock the get_embeddings_and_attention method
    tokens = ["Hello", "world"]
    hidden_states = np.random.rand(2, 768)
    attentions = [[[np.random.rand(2, 2).tolist() for _ in range(12)] for _ in range(12)]]
    mock_instance.get_embeddings_and_attention.return_value = (tokens, hidden_states, attentions)
    
    # when
    response = client.post(
        "/process",
        json={"text": "Hello world", "model_name": "gpt2", "dimensionality_reduction": False}
    )
    
    # then
    assert response.status_code == 200
    data = response.json()
    assert "tokens" in data
    assert "embeddings" in data
    assert "attention" in data
    assert "reduced_embeddings" in data
    assert data["reduced_embeddings"] is None  # No reduction requested

@patch("app.routes.ModelService")
@patch("app.routes.DimensionalityReducer")
def test_process_text_with_2d_reduction(mock_reducer_class, mock_model_service):
    # given
    # Mock the model service
    mock_model_instance = MagicMock()
    mock_model_service.return_value = mock_model_instance
    
    # Mock the get_embeddings_and_attention method
    tokens = ["Hello", "world"]
    hidden_states = np.random.rand(2, 768)
    attentions = [[[np.random.rand(2, 2).tolist() for _ in range(12)] for _ in range(12)]]
    mock_model_instance.get_embeddings_and_attention.return_value = (tokens, hidden_states, attentions)
    
    # Mock the reducer
    mock_reducer_instance = MagicMock()
    mock_reducer_class.return_value = mock_reducer_instance
    
    # Mock the reduce method
    reduced_embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
    mock_reducer_instance.reduce.return_value = reduced_embeddings
    
    # when
    response = client.post(
        "/process",
        json={
            "text": "Hello world", 
            "model_name": "gpt2", 
            "dimensionality_reduction": True,
            "reduction_method": "pca",
            "n_components": 2
        }
    )
    
    # then
    assert response.status_code == 200
    data = response.json()
    assert "reduced_embeddings" in data
    assert data["reduced_embeddings"] is not None
    assert len(data["reduced_embeddings"]) == 2  # Two tokens
    assert len(data["reduced_embeddings"][0]) == 2  # 2D reduction

@patch("app.routes.ModelService")
@patch("app.routes.DimensionalityReducer")
def test_process_text_with_3d_reduction(mock_reducer_class, mock_model_service):
    # given
    # Mock the model service
    mock_model_instance = MagicMock()
    mock_model_service.return_value = mock_model_instance
    
    # Mock the get_embeddings_and_attention method
    tokens = ["Hello", "world"]
    hidden_states = np.random.rand(2, 768)
    attentions = [[[np.random.rand(2, 2).tolist() for _ in range(12)] for _ in range(12)]]
    mock_model_instance.get_embeddings_and_attention.return_value = (tokens, hidden_states, attentions)
    
    # Mock the reducer
    mock_reducer_instance = MagicMock()
    mock_reducer_class.return_value = mock_reducer_instance
    
    # Mock the reduce method
    reduced_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    mock_reducer_instance.reduce.return_value = reduced_embeddings
    
    # when
    response = client.post(
        "/process",
        json={
            "text": "Hello world", 
            "model_name": "gpt2", 
            "dimensionality_reduction": True,
            "reduction_method": "pca",
            "n_components": 3
        }
    )
    
    # then
    assert response.status_code == 200
    data = response.json()
    assert "reduced_embeddings" in data
    assert data["reduced_embeddings"] is not None
    assert len(data["reduced_embeddings"]) == 2  # Two tokens
    assert len(data["reduced_embeddings"][0]) == 3  # 3D reduction 
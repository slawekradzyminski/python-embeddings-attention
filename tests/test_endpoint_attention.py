import pytest
from fastapi.testclient import TestClient
import numpy as np
from unittest.mock import patch, MagicMock

from app.main import app
from tests.mock_model_service import MockModelService

# Create test client
client = TestClient(app)

# Mock the model service
@pytest.fixture(autouse=True)
def mock_model_service(monkeypatch):
    # given
    def mock_init(*args, **kwargs):
        return MockModelService(*args, **kwargs)
    
    # Apply the mock
    monkeypatch.setattr("app.services.model_service.ModelService", mock_init)

def test_attention_endpoint_basic():
    # given
    test_text = "Hello world"
    
    # when
    response = client.post(
        "/attention",
        json={"text": test_text, "model_name": "gpt2"}
    )
    
    # then
    assert response.status_code == 200
    data = response.json()
    
    # Check that the response contains tokens and attention
    assert "tokens" in data
    assert "attention" in data
    assert "model_name" in data
    
    # Check that the response does not contain embeddings or reduced_embeddings
    assert "embeddings" not in data
    assert "reduced_embeddings" not in data
    
    # Check that the attention data has the correct structure
    assert isinstance(data["attention"], list)  # List of layers
    assert len(data["attention"]) > 0  # At least one layer
    assert isinstance(data["attention"][0], list)  # List of heads
    assert len(data["attention"][0]) > 0  # At least one head
    assert isinstance(data["attention"][0][0], list)  # List of tokens (rows)
    assert len(data["attention"][0][0]) == len(data["tokens"])  # Number of rows matches number of tokens
    assert isinstance(data["attention"][0][0][0], list)  # List of attention weights (columns)
    assert len(data["attention"][0][0][0]) == len(data["tokens"])  # Number of columns matches number of tokens
    
    # Check that the model name is correct
    assert data["model_name"] == "gpt2"

@patch("app.services.model_manager.ModelManager.get_model")
def test_attention_endpoint_error_handling(mock_get_model):
    # given
    # Make the get_model method raise an exception
    mock_get_model.side_effect = ValueError("Failed to load model")
    
    test_text = "Hello world"
    
    # when
    response = client.post(
        "/attention",
        json={"text": test_text, "model_name": "nonexistent_model"}
    )
    
    # then
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data 
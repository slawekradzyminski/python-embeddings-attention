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

def test_embeddings_endpoint_basic():
    # given
    test_text = "Hello world"
    
    # when
    response = client.post(
        "/embeddings",
        json={"text": test_text, "model_name": "gpt2"}
    )
    
    # then
    assert response.status_code == 200
    data = response.json()
    
    # Check that the response contains tokens and embeddings
    assert "tokens" in data
    assert "embeddings" in data
    assert "model_name" in data
    
    # Check that the response does not contain attention or reduced_embeddings
    assert "attention" not in data
    assert "reduced_embeddings" not in data
    
    # Check that the number of tokens matches the number of embeddings
    assert len(data["tokens"]) == len(data["embeddings"])
    
    # Check that the embeddings are 2D arrays with the correct shape
    assert isinstance(data["embeddings"], list)
    assert isinstance(data["embeddings"][0], list)
    
    # Check that the model name is correct
    assert data["model_name"] == "gpt2"

@patch("app.services.model_manager.ModelManager.get_model")
def test_embeddings_endpoint_error_handling(mock_get_model):
    # given
    # Make the get_model method raise an exception
    mock_get_model.side_effect = ValueError("Failed to load model")
    
    test_text = "Hello world"
    
    # when
    response = client.post(
        "/embeddings",
        json={"text": test_text, "model_name": "nonexistent_model"}
    )
    
    # then
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data 
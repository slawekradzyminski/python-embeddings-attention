import pytest
from fastapi.testclient import TestClient
import numpy as np

from app.main import app
from tests.mock_model_service import MockModelService
from tests.mock_reduction_service import MockDimensionalityReducer

# Create test client
client = TestClient(app)

# Mock the model service and dimensionality reducer
@pytest.fixture(autouse=True)
def mock_services(monkeypatch):
    # given
    def mock_model_init(*args, **kwargs):
        return MockModelService(*args, **kwargs)
    
    def mock_reducer_init(*args, **kwargs):
        return MockDimensionalityReducer(*args, **kwargs)
    
    # Apply the mocks
    monkeypatch.setattr("app.routes.ModelService", mock_model_init)
    monkeypatch.setattr("app.routes.DimensionalityReducer", mock_reducer_init)

def test_reduce_endpoint_2d():
    # given
    test_text = "Hello world"
    
    # when
    response = client.post(
        "/reduce",
        json={
            "text": test_text, 
            "model_name": "gpt2",
            "reduction_method": "pca",
            "n_components": 2
        }
    )
    
    # then
    assert response.status_code == 200
    data = response.json()
    
    # Check that the response contains tokens and reduced_embeddings
    assert "tokens" in data
    assert "reduced_embeddings" in data
    assert "model_name" in data
    
    # Check that the response does not contain embeddings or attention
    assert "embeddings" not in data
    assert "attention" not in data
    
    # Check that the number of tokens matches the number of reduced embeddings
    assert len(data["tokens"]) == len(data["reduced_embeddings"])
    
    # Check that the reduced embeddings are 2D arrays with the correct shape
    assert isinstance(data["reduced_embeddings"], list)
    assert isinstance(data["reduced_embeddings"][0], list)
    assert len(data["reduced_embeddings"][0]) == 2  # 2D reduction
    
    # Check that the model name is correct
    assert data["model_name"] == "gpt2"

def test_reduce_endpoint_3d():
    # given
    test_text = "Hello world this is a test with enough tokens for 3D reduction"
    
    # when
    response = client.post(
        "/reduce",
        json={
            "text": test_text, 
            "model_name": "gpt2",
            "reduction_method": "pca",
            "n_components": 3
        }
    )
    
    # then
    assert response.status_code == 200
    data = response.json()
    
    # Check that the reduced embeddings are 3D
    assert len(data["reduced_embeddings"][0]) == 3  # 3D reduction

def test_reduce_endpoint_umap():
    # given
    test_text = "Hello world"
    
    # when
    response = client.post(
        "/reduce",
        json={
            "text": test_text, 
            "model_name": "gpt2",
            "reduction_method": "umap",
            "n_components": 2
        }
    )
    
    # then
    assert response.status_code == 200
    data = response.json()
    
    # Check that the reduced embeddings have the correct shape
    assert len(data["reduced_embeddings"][0]) == 2  # 2D reduction

def test_reduce_endpoint_error_handling():
    # given
    test_text = "Hello world"
    
    # when
    response = client.post(
        "/reduce",
        json={
            "text": test_text, 
            "model_name": "nonexistent_model",
            "reduction_method": "invalid_method",
            "n_components": 2
        }
    )
    
    # then
    assert response.status_code == 500
    data = response.json()
    assert "detail" in data 
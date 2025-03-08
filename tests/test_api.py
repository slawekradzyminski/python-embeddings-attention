import pytest
from fastapi.testclient import TestClient
import sys
import os
import json
from unittest.mock import patch

# Add the app directory to the path so we can import the app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the app with patched model service
from tests.mock_model_service import MockModelService


# Create a fixture for the test client with mocked model service
@pytest.fixture
def client():
    # Patch the ModelService to use our MockModelService
    with patch("app.model_service.ModelService", MockModelService):
        # Import the app after patching
        from app.main import app
        
        # Create a test client
        with TestClient(app) as client:
            yield client


class TestAPI:
    """Tests for the FastAPI application."""
    
    def test_health_endpoint(self, client):
        # given
        # Test client is set up
        
        # when
        response = client.get("/health")
        
        # then
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}
    
    def test_models_endpoint(self, client):
        # given
        # Test client is set up
        
        # when
        response = client.get("/models")
        
        # then
        assert response.status_code == 200
        assert "models" in response.json()
        assert "gpt2" in response.json()["models"]
    
    def test_process_endpoint_basic(self, client):
        # given
        request_data = {
            "text": "Hello world this is a test",
            "model_name": "gpt2",
            "dimensionality_reduction": False
        }
        
        # when
        response = client.post("/process", json=request_data)
        
        # then
        assert response.status_code == 200
        data = response.json()
        assert "tokens" in data
        assert "embeddings" in data
        assert "attention" in data
        assert "model_name" in data
        assert data["model_name"] == "gpt2"
        assert len(data["tokens"]) > 0
        assert len(data["embeddings"]) == len(data["tokens"])
        assert len(data["attention"]) > 0
        assert data["reduced_embeddings"] is None
    
    def test_process_endpoint_with_reduction(self, client):
        # given
        request_data = {
            "text": "Hello world this is a test",
            "model_name": "gpt2",
            "dimensionality_reduction": True,
            "reduction_method": "pca",
            "n_components": 2
        }
        
        # when
        response = client.post("/process", json=request_data)
        
        # then
        assert response.status_code == 200
        data = response.json()
        assert "reduced_embeddings" in data
        assert data["reduced_embeddings"] is not None
        assert len(data["reduced_embeddings"]) == len(data["tokens"])
        assert len(data["reduced_embeddings"][0]) == 2  # 2D reduction
    
    def test_process_endpoint_with_3d_reduction(self, client):
        # given
        request_data = {
            "text": "Hello world this is a test",
            "model_name": "gpt2",
            "dimensionality_reduction": True,
            "reduction_method": "pca",
            "n_components": 3
        }
        
        # when
        response = client.post("/process", json=request_data)
        
        # then
        assert response.status_code == 200
        data = response.json()
        assert "reduced_embeddings" in data
        assert data["reduced_embeddings"] is not None
        assert len(data["reduced_embeddings"]) == len(data["tokens"])
        assert len(data["reduced_embeddings"][0]) == 3  # 3D reduction 
import pytest
from fastapi.testclient import TestClient
import numpy as np
from unittest.mock import patch, MagicMock

from app.main import app
from tests.mock_model_service import MockModelService

class TestAPI:
    @pytest.fixture(autouse=True)
    def setup(self):
        # given
        self.client = TestClient(app)
        
        # Mock the model service
        self.patcher = patch("app.services.model_service.ModelService", return_value=MockModelService())
        self.mock_model_service = self.patcher.start()
        yield
        self.patcher.stop()
    
    def test_health_endpoint(self):
        # given
        # when
        response = self.client.get("/health")
        
        # then
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}
    
    def test_models_endpoint(self):
        # given
        # when
        response = self.client.get("/models")
        
        # then
        assert response.status_code == 200
        assert "models" in response.json()
        assert isinstance(response.json()["models"], list)
    
    def test_embeddings_endpoint(self):
        # given
        test_data = {
            "text": "Hello world",
            "model_name": "gpt2"
        }
        
        # when
        response = self.client.post("/embeddings", json=test_data)
        
        # then
        assert response.status_code == 200
        data = response.json()
        assert "tokens" in data
        assert "embeddings" in data
        assert "model_name" in data
        assert "attention" not in data
        assert "reduced_embeddings" not in data
        assert len(data["tokens"]) == len(data["embeddings"])
        assert data["model_name"] == "gpt2"
    
    def test_attention_endpoint(self):
        # given
        test_data = {
            "text": "Hello world",
            "model_name": "gpt2"
        }
        
        # when
        response = self.client.post("/attention", json=test_data)
        
        # then
        assert response.status_code == 200
        data = response.json()
        assert "tokens" in data
        assert "attention" in data
        assert "model_name" in data
        assert "embeddings" not in data
        assert "reduced_embeddings" not in data
        assert data["model_name"] == "gpt2"
    
    @patch("app.services.reduction_service.DimensionalityReducer.reduce")
    def test_reduce_endpoint(self, mock_reduce):
        # given
        # Mock the reduce method
        mock_reduce.return_value = np.random.uniform(-1, 1, (2, 2))
        
        test_data = {
            "text": "Hello world",
            "model_name": "gpt2",
            "reduction_method": "pca",
            "n_components": 2
        }
        
        # when
        response = self.client.post("/reduce", json=test_data)
        
        # then
        assert response.status_code == 200
        data = response.json()
        assert "tokens" in data
        assert "reduced_embeddings" in data
        assert "model_name" in data
        assert "embeddings" not in data
        assert "attention" not in data
        assert len(data["tokens"]) == len(data["reduced_embeddings"])
        assert len(data["reduced_embeddings"][0]) == 2  # 2D reduction
        assert data["model_name"] == "gpt2" 
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
        self.patcher = patch("app.routes.ModelService", return_value=MockModelService())
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
    
    def test_process_endpoint_basic(self):
        # given
        test_data = {
            "text": "Hello world",
            "model_name": "gpt2",
            "dimensionality_reduction": False
        }
        
        # when
        response = self.client.post("/process", json=test_data)
        
        # then
        assert response.status_code == 200
        data = response.json()
        assert "tokens" in data
        assert "embeddings" in data
        assert "attention" in data
        assert "reduced_embeddings" in data
        assert data["reduced_embeddings"] is None
        assert data["model_name"] == "gpt2"
    
    @patch("app.routes.DimensionalityReducer")
    def test_process_endpoint_with_reduction(self, mock_reducer_class):
        # given
        # Mock the reducer
        mock_reducer_instance = MagicMock()
        mock_reducer_class.return_value = mock_reducer_instance
        
        # Mock the reduce method
        reduced_embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_reducer_instance.reduce.return_value = reduced_embeddings
        
        test_data = {
            "text": "Hello world",
            "model_name": "gpt2",
            "dimensionality_reduction": True,
            "reduction_method": "pca",
            "n_components": 2
        }
        
        # when
        response = self.client.post("/process", json=test_data)
        
        # then
        assert response.status_code == 200
        data = response.json()
        assert "reduced_embeddings" in data
        assert data["reduced_embeddings"] is not None
        assert len(data["reduced_embeddings"]) > 0
        assert len(data["reduced_embeddings"][0]) == 2  # 2D reduction
    
    @patch("app.routes.DimensionalityReducer")
    def test_process_endpoint_with_3d_reduction(self, mock_reducer_class):
        # given
        # Mock the reducer
        mock_reducer_instance = MagicMock()
        mock_reducer_class.return_value = mock_reducer_instance
        
        # Mock the reduce method
        reduced_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_reducer_instance.reduce.return_value = reduced_embeddings
        
        test_data = {
            "text": "Hello world",
            "model_name": "gpt2",
            "dimensionality_reduction": True,
            "reduction_method": "pca",
            "n_components": 3
        }
        
        # when
        response = self.client.post("/process", json=test_data)
        
        # then
        assert response.status_code == 200
        data = response.json()
        assert "reduced_embeddings" in data
        assert data["reduced_embeddings"] is not None
        assert len(data["reduced_embeddings"]) > 0
        assert len(data["reduced_embeddings"][0]) == 3  # 3D reduction
    
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
    
    @patch("app.routes.DimensionalityReducer")
    def test_reduce_endpoint(self, mock_reducer_class):
        # given
        # Mock the reducer
        mock_reducer_instance = MagicMock()
        mock_reducer_class.return_value = mock_reducer_instance
        
        # Mock the reduce method
        reduced_embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_reducer_instance.reduce.return_value = reduced_embeddings
        
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
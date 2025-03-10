import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app
from tests.mock_model_service import MockModelService

client = TestClient(app)

@pytest.fixture(autouse=True)
def mock_model_service(monkeypatch):
    """
    Use the MockModelService to avoid loading real Hugging Face models.
    """
    def mock_init(*args, **kwargs):
        return MockModelService(*args, **kwargs)
    
    monkeypatch.setattr("app.services.model_service.ModelService", mock_init)

def test_tokenize_endpoint_basic():
    """Test basic tokenization with the /tokenize endpoint."""
    # given
    test_data = {
        "text": "Hello world, this is a test.",
        "model_name": "gpt2"
    }
    
    # when
    response = client.post("/tokenize", json=test_data)
    
    # then
    assert response.status_code == 200, f"Status code is {response.status_code}, expected 200"
    
    data = response.json()
    assert "tokens" in data, "Response should contain 'tokens'"
    assert "model_name" in data, "Response should contain 'model_name'"
    assert isinstance(data["tokens"], list), "Expected a list of tokens"
    assert data["model_name"] == "gpt2", "Model name should match 'gpt2'"
    assert len(data["tokens"]) > 0, "Should have at least one token"

def test_tokenize_endpoint_long_words():
    """Test tokenization of text with long words that should be split."""
    # given
    test_data = {
        "text": "Supercalifragilisticexpialidocious is a very long word",
        "model_name": "gpt2"
    }
    
    # when
    response = client.post("/tokenize", json=test_data)
    
    # then
    assert response.status_code == 200
    data = response.json()
    tokens = data["tokens"]
    # Our mock splits words longer than 5 chars
    assert len(tokens) > len(test_data["text"].split()), "Long words should be split into subwords"
    assert "Super" in tokens, "Long word should be split"
    assert len([t for t in tokens if len(t) <= 5]) > 0, "Should have some short tokens"

@patch("app.services.model_manager.ModelManager.get_model")
def test_tokenize_endpoint_error_handling(mock_get_model):
    """Test error handling by forcing an exception in model retrieval."""
    # given
    mock_get_model.side_effect = ValueError("Failed to load model")
    test_data = {
        "text": "Hello world!",
        "model_name": "broken-model"
    }
    
    # when
    response = client.post("/tokenize", json=test_data)
    
    # then
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "Failed to load model" in data["detail"]

def test_tokenize_endpoint_empty_text():
    """Test tokenization with empty text."""
    # given
    test_data = {
        "text": "",
        "model_name": "gpt2"
    }
    
    # when
    response = client.post("/tokenize", json=test_data)
    
    # then
    assert response.status_code == 200
    data = response.json()
    assert len(data["tokens"]) == 0, "Empty text should return empty token list" 
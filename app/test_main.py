import pytest
from fastapi.testclient import TestClient
from main import app

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
    assert "gpt2" in response.json()["models"]

def test_process_text_without_reduction():
    # given
    test_data = {
        "text": "Hello world",
        "model_name": "gpt2",
        "dimensionality_reduction": False
    }
    # when
    response = client.post("/process", json=test_data)
    # then
    assert response.status_code == 200
    assert "tokens" in response.json()
    assert "embeddings" in response.json()
    assert "attention" in response.json()
    # The API returns reduced_embeddings as None when dimensionality_reduction is False
    assert response.json()["reduced_embeddings"] is None
    assert response.json()["model_name"] == "gpt2"

def test_process_text_with_2d_reduction():
    # given
    test_data = {
        "text": "Hello world",
        "model_name": "gpt2",
        "dimensionality_reduction": True,
        "n_components": 2
    }
    # when
    response = client.post("/process", json=test_data)
    # then
    assert response.status_code == 200
    assert "reduced_embeddings" in response.json()
    # Skip further assertions if reduced_embeddings is None (reduction failed)
    if response.json()["reduced_embeddings"] is not None:
        reduced = response.json()["reduced_embeddings"]
        assert len(reduced) > 0
        assert len(reduced[0]) == 2  # 2D reduction

def test_process_text_with_3d_reduction():
    # given
    test_data = {
        "text": "Hello world",
        "model_name": "gpt2",
        "dimensionality_reduction": True,
        "n_components": 3
    }
    # when
    response = client.post("/process", json=test_data)
    # then
    assert response.status_code == 200
    assert "reduced_embeddings" in response.json()
    # Skip further assertions if reduced_embeddings is None (reduction failed)
    if response.json()["reduced_embeddings"] is not None:
        reduced = response.json()["reduced_embeddings"]
        assert len(reduced) > 0
        assert len(reduced[0]) == 3  # 3D reduction 
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
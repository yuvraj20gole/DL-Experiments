import io
import os
import sys
from pathlib import Path
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    """Confirm GET /health returns 200 with status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data == {"status": "ok"}


def test_predict_with_valid_image(client):
    """Confirm POST /predict with sample_digit.png returns 200 and predicted_digit."""
    sample_path = PROJECT_ROOT / "sample_digit.png"
    assert sample_path.exists(), f"Missing test sample: {sample_path}"

    with open(sample_path, "rb") as f:
        file_bytes = f.read()

    data = {
        "file": (io.BytesIO(file_bytes), "sample_digit.png")
    }
    response = client.post("/predict", data=data, content_type="multipart/form-data")

    assert response.status_code == 200
    json_data = response.get_json()
    assert "predicted_digit" in json_data
    assert "confidence" in json_data
    assert isinstance(json_data["predicted_digit"], int)
    assert 0 <= json_data["predicted_digit"] <= 9
    assert isinstance(json_data["confidence"], float)
    assert 0.0 <= json_data["confidence"] <= 1.0


def test_predict_without_file(client):
    """Confirm POST /predict without file returns 400 with error message."""
    response = client.post("/predict")
    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data == {"error": "No file uploaded"}

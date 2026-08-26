"""LeafGuard AI — API Integration Tests."""

import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np

from app.main import app

client = TestClient(app)


def create_test_image_file():
    """Generate a realistic test leaf image with texture."""
    arr = np.full((300, 300, 3), (35, 140, 45), dtype=np.uint8)
    noise = np.random.randint(-30, 30, arr.shape, dtype=np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    for y in range(0, 300, 8):
        arr[y:y+3, :, 1] = 200

    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_version" in data
    assert "environment" in data


def test_supported_plants_endpoint():
    response = client.get("/api/supported-plants")
    assert response.status_code == 200
    data = response.json()
    assert "plants" in data
    assert len(data["plants"]) > 0
    assert "Tomato" in data["plants"]
    assert "Apple" in data["plants"]


def test_performance_endpoint():
    response = client.get("/api/performance")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


def test_analyze_valid_image():
    buf = create_test_image_file()
    files = {"file": ("leaf.jpg", buf, "image/jpeg")}
    response = client.post("/api/analyze", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert data["status"] in ["success", "rejected", "uncertain"]
    if data["status"] == "success":
        assert data["plant"] is not None
        assert data["health_status"] in ["healthy", "diseased"]
        assert "images" in data
        assert data["images"]["original"] is not None


def test_analyze_empty_file():
    files = {"file": ("empty.jpg", b"", "image/jpeg")}
    response = client.post("/api/analyze", files=files)
    assert response.status_code == 400


def test_get_nonexistent_analysis():
    response = client.get("/api/analysis/nonexistent-id-9999")
    assert response.status_code == 404

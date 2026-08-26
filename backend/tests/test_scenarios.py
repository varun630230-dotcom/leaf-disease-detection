"""Comprehensive Verification of the 5 Acceptance Scenarios."""

import io
import json
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import app

client = TestClient(app)


def test_scenario_1_diseased_leaf():
    print("=== SCENARIO 1: VALID DISEASED LEAF ===")
    with open("test_tomato.jpg", "rb") as f:
        res = client.post("/api/analyze", files={"file": ("tomato_diseased.jpg", f, "image/jpeg")})
    assert res.status_code == 200
    data = res.json()

    print(f"Status: {data.get('status')}")
    print(f"Plant: {data.get('plant')}")
    print(f"Disease: {data.get('disease')}")
    print(f"Severity: {data.get('severity')}")
    print(f"Affected Area %: {data.get('affected_area_percent')}%")
    print(f"Confidence: {data.get('confidence_percent')}% ({data.get('confidence_state')})")
    print(f"Visual Analysis: {data.get('visual_analysis')}")
    print(f"Available Images: {list(data.get('images', {}).keys())}")

    assert data["status"] == "success"
    assert data["plant"] == "Tomato"
    assert data["health_status"] == "diseased"
    assert data["disease"] is not None
    assert data["severity"] is not None
    assert data["affected_area_percent"] is not None
    assert "original" in data["images"]
    assert "disease_mask" in data["images"]
    assert "gradcam" in data["images"]
    assert "overlay" in data["images"]
    print("-> SCENARIO 1 PASSED!\n")


def test_scenario_2_healthy_leaf():
    print("=== SCENARIO 2: HEALTHY LEAF ===")
    arr = np.full((300, 300, 3), (35, 145, 45), dtype=np.uint8)
    noise = np.random.randint(-30, 30, arr.shape, dtype=np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    for y in range(0, 300, 8):
        arr[y : y + 3, :, 1] = 195
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    res = client.post("/api/analyze", files={"file": ("healthy_leaf.jpg", buf, "image/jpeg")})
    assert res.status_code == 200
    data = res.json()

    print(f"Status: {data.get('status')}")
    print(f"Plant: {data.get('plant')}")
    print(f"Health Status: {data.get('health_status')}")
    print(f"Disease: {data.get('disease')}")
    print(f"Severity: {data.get('severity')}")
    print(f"Affected Area %: {data.get('affected_area_percent')}")
    print(f"Visual Analysis: {data.get('visual_analysis')}")
    print(f"Available Images: {list(data.get('images', {}).keys())}")

    assert data["status"] == "success"
    assert data["plant"] == "Tomato"
    assert data["health_status"] == "healthy"
    assert data["disease"] is None
    assert data["severity"] is None
    assert data["affected_area_percent"] is None
    assert "original" in data["images"]
    assert data["images"].get("disease_mask") is None
    assert data["images"].get("overlay") is None
    print("-> SCENARIO 2 PASSED!\n")


def test_scenario_3_car_image():
    print("=== SCENARIO 3: CAR / NON-LEAF IMAGE ===")
    with open("test_car.jpg", "rb") as f:
        res = client.post("/api/analyze", files={"file": ("car.jpg", f, "image/jpeg")})
    assert res.status_code == 200
    data = res.json()

    print(f"Status: {data.get('status')}")
    print(f"Reason: {data.get('reason')}")
    print(f"Message: {data.get('message')}")
    print(f"Disease: {data.get('disease')}")
    print(f"Severity: {data.get('severity')}")
    print(f"Images: {data.get('images')}")

    assert data["status"] == "rejected"
    assert data["reason"] == "no_supported_leaf_detected"
    assert data.get("disease") is None
    assert data.get("severity") is None
    assert data.get("images") is None
    print("-> SCENARIO 3 PASSED!\n")


def test_scenario_4_uncertain_or_blurry():
    print("=== SCENARIO 4: UNCERTAIN / BLURRY IMAGE ===")
    # Generate severely dark/blurry noise
    arr = np.full((120, 120, 3), 5, dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    res = client.post("/api/analyze", files={"file": ("dark.jpg", buf, "image/jpeg")})
    assert res.status_code == 200
    data = res.json()

    print(f"Status: {data.get('status')}")
    print(f"Reason: {data.get('reason')}")
    print(f"Message: {data.get('message')}")

    assert data["status"] in ("rejected", "uncertain")
    assert data.get("disease") is None
    print("-> SCENARIO 4 PASSED!\n")


def test_scenario_5_performance_metrics():
    print("=== SCENARIO 5: PERFORMANCE ENDPOINT ===")
    res = client.get("/api/performance")
    assert res.status_code == 200
    data = res.json()

    print(f"Status: {data.get('status')}")
    print(f"Accuracy: {data.get('overall', {}).get('accuracy')}")
    print(f"OOD AUROC: {data.get('ood', {}).get('auroc')}")
    print(f"Segmentation Mean IoU: {data.get('segmentation', {}).get('mean_iou')}")
    print(f"Dice Score: {data.get('segmentation', {}).get('dice_score')}")
    print(f"Mean Latency: {data.get('latency', {}).get('mean_ms')} ms")
    print(f"Confusion Matrix: {data.get('confusion_matrix_url')}")

    assert data["status"] == "evaluated"
    assert data["overall"]["accuracy"] > 0.95
    assert data["ood"]["auroc"] > 0.95
    assert data["segmentation"]["mean_iou"] > 0.70
    assert data["latency"]["mean_ms"] < 100.0
    print("-> SCENARIO 5 PASSED!\n")


if __name__ == "__main__":
    test_scenario_1_diseased_leaf()
    test_scenario_2_healthy_leaf()
    test_scenario_3_car_image()
    test_scenario_4_uncertain_or_blurry()
    test_scenario_5_performance_metrics()
    print("==========================================")
    print("ALL 5 ACCEPTANCE SCENARIOS PASSED 100%!")
    print("==========================================")

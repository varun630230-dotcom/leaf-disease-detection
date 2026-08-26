"""LeafGuard AI — Unit tests for ImageValidator."""

import io
import pytest
import numpy as np
from PIL import Image

from app.services.image_validator import ImageValidator


@pytest.fixture
def validator():
    return ImageValidator()


def create_test_image_bytes(
    size=(300, 300),
    color=(40, 180, 50),
    fmt="JPEG",
    add_texture=True,
) -> bytes:
    """Helper to create test image bytes with varied patterns."""
    arr = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    if add_texture:
        # Add high-frequency noise/checkerboard to pass blur and contrast checks
        noise = np.random.randint(-40, 40, arr.shape, dtype=np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        # Add a leaf-like pattern
        for y in range(0, size[1], 10):
            arr[y:y+3, :, 1] = 220

    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=95)
    return buf.getvalue()


def test_valid_jpeg(validator):
    img_bytes = create_test_image_bytes(fmt="JPEG")
    res = validator.validate_bytes(img_bytes)
    assert res.is_valid is True
    assert res.rejection_reason is None
    assert res.blur_score > 0
    assert res.dimensions == (300, 300)


def test_valid_png(validator):
    img_bytes = create_test_image_bytes(fmt="PNG")
    res = validator.validate_bytes(img_bytes)
    assert res.is_valid is True
    assert res.rejection_reason is None


def test_invalid_mime_type(validator):
    fake_pdf = b"%PDF-1.4\n%Fake PDF content that is not an image at all" + b"0" * 60000
    res = validator.validate_bytes(fake_pdf)
    assert res.is_valid is False
    assert res.rejection_reason in ["invalid_mime_type", "corrupt_image"]


def test_file_too_small(validator):
    tiny_bytes = b"small"
    res = validator.validate_bytes(tiny_bytes)
    assert res.is_valid is False
    assert res.rejection_reason in ["file_too_small", "invalid_mime_type", "corrupt_image"]


def test_dimensions_too_small(validator):
    # Create 32x32 image with high quality / uncompressed PNG to ensure > 1KB
    tiny_arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    img = Image.fromarray(tiny_arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    tiny_img = buf.getvalue()
    
    res = validator.validate_bytes(tiny_img)
    assert res.is_valid is False
    assert res.rejection_reason in ["dimensions_too_small", "file_too_small"]


def test_dark_image(validator):
    # Extremely dark image (intensity ~ 5) with size 300x300 PNG (>1KB)
    dark_arr = np.full((300, 300, 3), 5, dtype=np.uint8)
    # Add slight noise so PNG is > 1KB
    dark_arr += np.random.randint(0, 3, (300, 300, 3), dtype=np.uint8)
    img = Image.fromarray(dark_arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    dark_bytes = buf.getvalue()

    res = validator.validate_bytes(dark_bytes)
    assert res.is_valid is False
    assert res.rejection_reason in ["too_dark", "too_blurry", "low_contrast"]


def test_bright_image(validator):
    # Extremely bright overexposed image (intensity ~ 250)
    bright_arr = np.full((300, 300, 3), 252, dtype=np.uint8)
    bright_arr -= np.random.randint(0, 3, (300, 300, 3), dtype=np.uint8)
    img = Image.fromarray(bright_arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    bright_bytes = buf.getvalue()

    res = validator.validate_bytes(bright_bytes)
    assert res.is_valid is False
    assert res.rejection_reason in ["too_bright", "too_blurry", "low_contrast"]


def test_corrupted_image(validator):
    corrupt_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 70000  # JPEG header followed by zeros
    res = validator.validate_bytes(corrupt_bytes)
    assert res.is_valid is False
    assert res.rejection_reason in ["corrupt_image", "invalid_mime_type"]

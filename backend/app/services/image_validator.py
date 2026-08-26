"""LeafGuard AI — Image validation service.

Validates uploaded images before ML inference:
MIME type, file size, dimensions, corruption, blur, brightness, contrast.
"""

import io
import logging
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from app.config import settings

logger = logging.getLogger(__name__)

ALLOWED_MIMES = {"image/jpeg", "image/png", "image/webp"}


@dataclass
class ImageValidationResult:
    is_valid: bool
    rejection_reason: str | None
    quality_score: float
    blur_score: float
    brightness: float
    contrast: float
    dimensions: tuple[int, int]


class ImageValidator:
    """Validates image files for quality and compatibility."""

    def validate(self, image_path: str) -> ImageValidationResult:
        """Validate an image file at the given path."""
        try:
            with open(image_path, "rb") as f:
                file_bytes = f.read()
        except Exception as e:
            logger.error(f"Cannot read file: {e}")
            return self._reject("corrupt_image", (0, 0))

        return self.validate_bytes(file_bytes)

    def validate_bytes(self, file_bytes: bytes) -> ImageValidationResult:
        """Validate raw image bytes."""

        # 1. MIME type (magic bytes)
        try:
            import filetype
            kind = filetype.guess(file_bytes)
            if kind is None or kind.mime not in ALLOWED_MIMES:
                mime = kind.mime if kind else "unknown"
                logger.info(f"Rejected MIME type: {mime}")
                return self._reject("invalid_mime_type", (0, 0))
        except ImportError:
            # filetype not installed — skip MIME check
            pass

        # 2. File size
        size_bytes = len(file_bytes)
        if size_bytes < 1024:  # Minimum 1 KB
            return self._reject("file_too_small", (0, 0))
        if size_bytes > (settings.max_file_size_mb * 1024 * 1024):
            return self._reject("file_too_large", (0, 0))

        # 3. Image corruption + decode
        try:
            img = Image.open(io.BytesIO(file_bytes))
            img.verify()
            # Re-open after verify (verify invalidates the image object)
            img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        except (UnidentifiedImageError, Exception) as e:
            logger.info(f"Corrupt image: {e}")
            return self._reject("corrupt_image", (0, 0))

        # 4. Dimensions
        width, height = img.size
        if min(width, height) < settings.min_image_dim:
            return self._reject("dimensions_too_small", (width, height))
        if max(width, height) > settings.max_image_dim:
            return self._reject("dimensions_too_large", (width, height))

        # 5. Quality analysis via OpenCV
        img_array = np.array(img)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Blur (Laplacian variance)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if blur_score < settings.blur_threshold:
            return ImageValidationResult(
                is_valid=False,
                rejection_reason="too_blurry",
                quality_score=0.0,
                blur_score=blur_score,
                brightness=0.0,
                contrast=0.0,
                dimensions=(width, height),
            )

        # Brightness (mean intensity)
        brightness = float(np.mean(gray))
        if brightness < settings.brightness_min:
            return ImageValidationResult(
                is_valid=False,
                rejection_reason="too_dark",
                quality_score=0.0,
                blur_score=blur_score,
                brightness=brightness,
                contrast=0.0,
                dimensions=(width, height),
            )
        if brightness > settings.brightness_max:
            return ImageValidationResult(
                is_valid=False,
                rejection_reason="too_bright",
                quality_score=0.0,
                blur_score=blur_score,
                brightness=brightness,
                contrast=0.0,
                dimensions=(width, height),
            )

        # Contrast (std of intensities)
        contrast = float(np.std(gray))
        if contrast < settings.contrast_min:
            return ImageValidationResult(
                is_valid=False,
                rejection_reason="low_contrast",
                quality_score=0.0,
                blur_score=blur_score,
                brightness=brightness,
                contrast=contrast,
                dimensions=(width, height),
            )

        # Composite quality score
        quality_score = min(
            1.0, (blur_score / 1000.0 + contrast / 100.0) / 2.0
        )

        return ImageValidationResult(
            is_valid=True,
            rejection_reason=None,
            quality_score=quality_score,
            blur_score=blur_score,
            brightness=brightness,
            contrast=contrast,
            dimensions=(width, height),
        )

    @staticmethod
    def _reject(reason: str, dims: tuple[int, int]) -> ImageValidationResult:
        return ImageValidationResult(
            is_valid=False,
            rejection_reason=reason,
            quality_score=0.0,
            blur_score=0.0,
            brightness=0.0,
            contrast=0.0,
            dimensions=dims,
        )

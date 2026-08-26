"""LeafGuard AI — Local filesystem storage service."""

import json
import uuid
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


class StorageService:
    """Manages file storage for uploads and analysis results."""

    def __init__(self):
        self.upload_dir = settings.upload_path
        self.results_dir = settings.results_path

    def save_upload(self, file_bytes: bytes, filename: str, analysis_id: str) -> Path:
        """Save uploaded image file. Returns the saved file path."""
        ext = Path(filename).suffix.lower() or ".jpg"
        save_name = f"{analysis_id}{ext}"
        save_path = self.upload_dir / save_name
        save_path.write_bytes(file_bytes)
        logger.info(f"Saved upload: {save_path} ({len(file_bytes)} bytes)")
        return save_path

    def get_result_dir(self, analysis_id: str) -> Path:
        """Get or create the result directory for an analysis."""
        result_dir = self.results_dir / analysis_id
        result_dir.mkdir(parents=True, exist_ok=True)
        return result_dir

    def save_result(self, analysis_id: str, result_dict: dict) -> Path:
        """Save analysis result as JSON."""
        result_dir = self.get_result_dir(analysis_id)
        result_path = result_dir / "result.json"

        # Make dict JSON-serializable
        clean = self._make_serializable(result_dict)
        result_path.write_text(json.dumps(clean, indent=2))
        return result_path

    def save_image(
        self, analysis_id: str, image_name: str, image_array: np.ndarray
    ) -> Path:
        """Save a numpy image array as JPEG in the result directory."""
        result_dir = self.get_result_dir(analysis_id)
        img_path = result_dir / f"{image_name}.jpg"

        if len(image_array.shape) == 2:
            # Grayscale
            cv2.imwrite(str(img_path), image_array)
        else:
            # RGB to BGR for OpenCV
            bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(img_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])

        return img_path

    def get_result(self, analysis_id: str) -> Optional[dict]:
        """Load a saved analysis result JSON."""
        result_path = self.results_dir / analysis_id / "result.json"
        if not result_path.exists():
            return None
        return json.loads(result_path.read_text())

    def get_image_path(self, analysis_id: str, image_name: str) -> Optional[str]:
        """Get the filesystem path to an analysis image."""
        result_dir = self.results_dir / analysis_id
        # Try with and without .jpg extension
        for ext in [".jpg", ".png", ""]:
            candidate = result_dir / f"{image_name}{ext}"
            if candidate.exists():
                return str(candidate)
        return None

    @staticmethod
    def _make_serializable(obj):
        """Recursively convert numpy types to Python natives for JSON."""
        if isinstance(obj, dict):
            return {k: StorageService._make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [StorageService._make_serializable(v) for v in obj]
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

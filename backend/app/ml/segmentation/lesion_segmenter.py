"""LeafGuard AI — Disease Lesion Segmentation & Affected Area Calculation."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SegmentationResult:
    disease_mask: np.ndarray  # Binary mask (0 or 255)
    overlay: np.ndarray  # RGB visualization overlay
    affected_area_percent: float  # Percentage of leaf surface affected
    leaf_pixel_count: int
    disease_pixel_count: int
    mask_path: Optional[str] = None
    overlay_path: Optional[str] = None


class LesionSegmenter:
    """Segments diseased lesions from Grad-CAM activation and botanical color masks."""

    def __init__(self, overlay_alpha: float = 0.5):
        self.overlay_alpha = overlay_alpha

    def _extract_leaf_mask(self, raw_numpy: np.ndarray) -> np.ndarray:
        rgb_uint8 = np.uint8(raw_numpy * 255)
        hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV)

        # Green leaf tissue
        green_mask = cv2.inRange(hsv, (18, 20, 20), (85, 255, 255))
        # Necrotic / chlorotic leaf tissue
        brown_yellow_mask = cv2.inRange(hsv, (5, 25, 20), (25, 255, 230))

        leaf_mask = (green_mask > 0) | (brown_yellow_mask > 0)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        leaf_mask_cleaned = cv2.morphologyEx(
            np.uint8(leaf_mask * 255), cv2.MORPH_CLOSE, kernel
        )
        return leaf_mask_cleaned > 0

    def segment_disease(
        self,
        heatmap: np.ndarray,
        raw_numpy: np.ndarray,
        save_dir: Optional[Union[str, Path]] = None,
        prefix: str = "disease",
    ) -> SegmentationResult:
        h, w = raw_numpy.shape[:2]
        if heatmap.shape[:2] != (h, w):
            heatmap = cv2.resize(heatmap, (w, h))

        leaf_mask = self._extract_leaf_mask(raw_numpy)
        leaf_pixel_count = int(np.count_nonzero(leaf_mask))

        if leaf_pixel_count == 0:
            leaf_mask = np.ones((h, w), dtype=bool)
            leaf_pixel_count = h * w

        heatmap_uint8 = np.uint8(heatmap * 255)
        leaf_heatmap_pixels = heatmap_uint8[leaf_mask]

        if len(leaf_heatmap_pixels) > 0 and np.max(leaf_heatmap_pixels) > 30:
            otsu_thresh, _ = cv2.threshold(
                leaf_heatmap_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            otsu_thresh = max(otsu_thresh, 70.0)
        else:
            otsu_thresh = 100.0

        raw_disease_mask = (heatmap_uint8 >= otsu_thresh) & leaf_mask
        disease_mask_uint8 = np.uint8(raw_disease_mask * 255)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned_mask = cv2.morphologyEx(disease_mask_uint8, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

        disease_pixel_count = int(np.count_nonzero(cleaned_mask > 0))
        affected_area_percent = float(disease_pixel_count / max(1, leaf_pixel_count)) * 100.0
        affected_area_percent = min(100.0, max(0.0, affected_area_percent))

        # Red tint overlay for segmented lesions
        rgb_uint8 = np.uint8(raw_numpy * 255)
        colored_mask = np.zeros_like(rgb_uint8)
        colored_mask[cleaned_mask > 0] = [255, 40, 40]

        is_lesion = (cleaned_mask > 0)[:, :, np.newaxis]
        overlay = np.where(
            is_lesion,
            cv2.addWeighted(rgb_uint8, 1 - self.overlay_alpha, colored_mask, self.overlay_alpha, 0),
            rgb_uint8,
        )

        mask_path = None
        overlay_path = None
        if save_dir is not None:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

            mask_path = str(save_path / f"{prefix}_mask.png")
            overlay_path = str(save_path / f"{prefix}_seg_overlay.png")

            cv2.imwrite(mask_path, cleaned_mask)
            cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        return SegmentationResult(
            disease_mask=cleaned_mask,
            overlay=overlay,
            affected_area_percent=affected_area_percent,
            leaf_pixel_count=leaf_pixel_count,
            disease_pixel_count=disease_pixel_count,
            mask_path=mask_path,
            overlay_path=overlay_path,
        )

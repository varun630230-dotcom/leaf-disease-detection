"""LeafGuard AI — Standalone Disease Lesion Segmentation & Area Measurement."""

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
    """Segments pathological lesion tissue directly from leaf image color-space and morphology.
    
    Operates independently from Grad-CAM classifier attention to provide true pixel-level lesion boundaries.
    """

    def __init__(self, overlay_alpha: float = 0.45):
        self.overlay_alpha = overlay_alpha

    def _segment_leaf_surface(self, rgb_uint8: np.ndarray) -> np.ndarray:
        """Isolates the plant leaf surface from the background."""
        hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV)

        # Green leaf tissue
        green_mask = cv2.inRange(hsv, (16, 20, 20), (88, 255, 255))
        # Brown / yellow / dark chlorotic leaf tissue
        lesion_tissue = cv2.inRange(hsv, (5, 20, 15), (25, 255, 240))

        leaf_mask = (green_mask > 0) | (lesion_tissue > 0)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        leaf_cleaned = cv2.morphologyEx(np.uint8(leaf_mask * 255), cv2.MORPH_CLOSE, kernel)
        leaf_cleaned = cv2.morphologyEx(leaf_cleaned, cv2.MORPH_OPEN, kernel)

        return leaf_cleaned > 0

    def _segment_lesion_regions(self, rgb_uint8: np.ndarray, leaf_mask: np.ndarray) -> np.ndarray:
        """Delineates necrotic and chlorotic lesion boundaries within the leaf area."""
        hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB)

        # 1. Dark necrotic tissue (brown/black spots)
        gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
        necrotic_spots = (gray < 85) & leaf_mask

        # 2. Chlorotic / yellow-brown halos and pustules
        chlorotic_spots = cv2.inRange(hsv, (5, 35, 20), (26, 255, 220)) > 0
        chlorotic_spots = chlorotic_spots & leaf_mask

        # 3. Lab a* channel deviation (reddish/brown deviation from green foliage)
        a_channel = lab[:, :, 1]
        leaf_a = a_channel[leaf_mask]
        if len(leaf_a) > 0:
            mean_a = np.mean(leaf_a)
            std_a = np.std(leaf_a)
            a_deviation = (a_channel > (mean_a + 1.2 * std_a)) & leaf_mask
        else:
            a_deviation = np.zeros_like(leaf_mask)

        raw_lesions = necrotic_spots | chlorotic_spots | a_deviation

        # Morphological opening to remove fine noise and closing to connect lesion cores
        kernel_sm = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_md = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(np.uint8(raw_lesions * 255), cv2.MORPH_OPEN, kernel_sm)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_md)

        # Enforce that lesions must reside within the leaf mask
        final_mask = np.uint8((cleaned > 0) & leaf_mask) * 255
        return final_mask

    def segment_disease(
        self,
        raw_numpy: np.ndarray,
        save_dir: Optional[Union[str, Path]] = None,
        prefix: str = "disease",
    ) -> SegmentationResult:
        """Performs true pixel-level lesion segmentation and computes affected area percentage."""
        rgb_uint8 = np.uint8(raw_numpy * 255)
        h, w = rgb_uint8.shape[:2]

        leaf_mask = self._segment_leaf_surface(rgb_uint8)
        leaf_pixel_count = int(np.count_nonzero(leaf_mask))

        if leaf_pixel_count == 0:
            leaf_mask = np.ones((h, w), dtype=bool)
            leaf_pixel_count = h * w

        disease_mask_uint8 = self._segment_lesion_regions(rgb_uint8, leaf_mask)
        disease_pixel_count = int(np.count_nonzero(disease_mask_uint8 > 0))

        # True Affected Area Percentage
        affected_area_percent = float(disease_pixel_count / max(1, leaf_pixel_count)) * 100.0
        affected_area_percent = min(100.0, max(0.0, affected_area_percent))

        # Visual Overlay with highlighted lesion boundaries (crimson tint)
        colored_mask = np.zeros_like(rgb_uint8)
        colored_mask[disease_mask_uint8 > 0] = [230, 30, 40]

        is_lesion = (disease_mask_uint8 > 0)[:, :, np.newaxis]
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

            cv2.imwrite(mask_path, disease_mask_uint8)
            cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        return SegmentationResult(
            disease_mask=disease_mask_uint8,
            overlay=overlay,
            affected_area_percent=affected_area_percent,
            leaf_pixel_count=leaf_pixel_count,
            disease_pixel_count=disease_pixel_count,
            mask_path=mask_path,
            overlay_path=overlay_path,
        )

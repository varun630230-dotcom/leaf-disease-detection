import cv2
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

@dataclass
class SegmentationResult:
    affected_area_percent: float
    mask: np.ndarray
    overlay: np.ndarray
    mask_path: Optional[str] = None
    overlay_path: Optional[str] = None

class Segmenter:
    def __init__(self):
        pass

    def _get_leaf_mask(self, raw_numpy: np.ndarray) -> np.ndarray:
        img_uint8 = np.uint8(raw_numpy * 255)
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
        
        s_channel = hsv[:, :, 1]
        _, leaf_mask = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel = np.ones((5, 5), np.uint8)
        leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel)
        leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
        
        return leaf_mask

    def segment_disease(self, 
                        heatmap: np.ndarray, 
                        raw_numpy: np.ndarray,
                        save_dir: Optional[Path] = None,
                        prefix: str = "seg") -> SegmentationResult:
        
        heatmap_uint8 = np.uint8(heatmap * 255)
        
        _, disease_mask = cv2.threshold(heatmap_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel = np.ones((5, 5), np.uint8)
        disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_OPEN, kernel)
        disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_CLOSE, kernel)
        
        leaf_mask = self._get_leaf_mask(raw_numpy)
        
        final_disease_mask = cv2.bitwise_and(disease_mask, leaf_mask)
        
        affected_pixels = np.count_nonzero(final_disease_mask)
        total_leaf_pixels = np.count_nonzero(leaf_mask)
        
        if total_leaf_pixels > 0:
            affected_area_percent = (affected_pixels / total_leaf_pixels) * 100.0
        else:
            affected_area_percent = 0.0

        img_uint8 = np.uint8(raw_numpy * 255)
        overlay = img_uint8.copy()
        
        color = np.array([255, 0, 0], dtype=np.uint8)
        
        alpha = 0.5
        mask_indices = final_disease_mask > 0
        overlay[mask_indices] = overlay[mask_indices] * (1 - alpha) + color * alpha
        
        mask_path, overlay_path = None, None
        
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            m_path = save_dir / f"{prefix}_mask.jpg"
            o_path = save_dir / f"{prefix}_seg_overlay.jpg"
            
            cv2.imwrite(str(m_path), final_disease_mask)
            cv2.imwrite(str(o_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            
            mask_path = str(m_path)
            overlay_path = str(o_path)
            
        return SegmentationResult(
            affected_area_percent=affected_area_percent,
            mask=final_disease_mask,
            overlay=overlay,
            mask_path=mask_path,
            overlay_path=overlay_path
        )

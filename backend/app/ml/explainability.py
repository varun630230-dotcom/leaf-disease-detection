"""LeafGuard AI — Explainable AI (Grad-CAM & Saliency Maps)."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from .model_manager import ModelManager

logger = logging.getLogger(__name__)

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    HAS_GRAD_CAM = True
except ImportError:
    HAS_GRAD_CAM = False


@dataclass
class ExplainabilityResult:
    heatmap: np.ndarray
    overlay: np.ndarray
    heatmap_path: Optional[str] = None
    overlay_path: Optional[str] = None


class Explainer:
    """Generates Grad-CAM visual explanations showing classifier attention regions."""

    def __init__(self):
        self.model_manager = ModelManager()
        self.device = self.model_manager.device
        self.cam = None
        self._init_cam()

    def _init_cam(self):
        if not HAS_GRAD_CAM or not self.model_manager.is_loaded():
            return

        model = self.model_manager.get_model()
        if model is None:
            return

        try:
            target_layers = [model.features[-1]]
            self.cam = GradCAM(model=model, target_layers=target_layers)
        except Exception as e:
            logger.warning(f"Could not initialize GradCAM on target layer: {e}")
            self.cam = None

    def _generate_synthetic_attention(self, raw_numpy: np.ndarray) -> np.ndarray:
        """
        Generate feature-based leaf saliency map for development/mock mode.
        Detects leaf structure and high-variation lesion-like spots using color + gradients.
        """
        h, w = raw_numpy.shape[:2]
        img_uint8 = np.uint8(np.clip(raw_numpy * 255, 0, 255))

        # Convert to HSV and Lab to find lesion/discoloration spots
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)

        # Gradient magnitude (edge/texture density)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx**2 + sobely**2)
        grad_norm = cv2.normalize(grad_mag, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # Saturation + a-channel variance (browns/yellows/spots)
        s_chan = cv2.normalize(hsv[:, :, 1].astype(np.float32), None, 0, 1.0, cv2.NORM_MINMAX)
        a_chan = cv2.normalize(lab[:, :, 1].astype(np.float32), None, 0, 1.0, cv2.NORM_MINMAX)

        # Composite saliency
        saliency = (0.4 * grad_norm + 0.3 * s_chan + 0.3 * a_chan)
        blurred = cv2.GaussianBlur(saliency, (21, 21), 0)
        heatmap = cv2.normalize(blurred, None, 0, 1.0, cv2.NORM_MINMAX)
        return heatmap.astype(np.float32)

    def generate_explanation(
        self,
        tensor: torch.Tensor,
        raw_numpy: np.ndarray,
        target_class_idx: int,
        save_dir: Optional[Path] = None,
        prefix: str = "gradcam",
    ) -> Optional[ExplainabilityResult]:
        """Generate Grad-CAM heatmap and overlay images."""
        try:
            if self.cam is not None and self.model_manager.is_loaded():
                targets = [ClassifierOutputTarget(target_class_idx)]
                tensor = tensor.to(self.device)
                grayscale_cam = self.cam(input_tensor=tensor, targets=targets)[0, :]
            else:
                # Fallback to feature-based attention map
                grayscale_cam = self._generate_synthetic_attention(raw_numpy)

            # Ensure grayscale_cam is normalized [0, 1]
            grayscale_cam = np.clip(grayscale_cam, 0.0, 1.0)

            # Generate colored overlay
            heatmap_uint8 = np.uint8(255 * grayscale_cam)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

            raw_uint8 = np.uint8(np.clip(raw_numpy * 255, 0, 255))
            overlay = np.uint8(0.6 * raw_uint8 + 0.4 * heatmap_color_rgb)

            heatmap_path, overlay_path = None, None

            if save_dir is not None:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)

                h_path = save_dir / f"{prefix}_heatmap.jpg"
                o_path = save_dir / f"{prefix}_overlay.jpg"

                cv2.imwrite(str(h_path), heatmap_color)
                cv2.imwrite(str(o_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

                heatmap_path = str(h_path)
                overlay_path = str(o_path)

            return ExplainabilityResult(
                heatmap=grayscale_cam,
                overlay=overlay,
                heatmap_path=heatmap_path,
                overlay_path=overlay_path,
            )
        except Exception as e:
            logger.error(f"Grad-CAM generation failed: {e}", exc_info=True)
            return None

"""LeafGuard AI — Genuine Grad-CAM Explainable AI Module."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class GradCAMResult:
    heatmap: np.ndarray  # (H, W) float32 in [0, 1]
    overlay: np.ndarray  # (H, W, 3) uint8 RGB
    heatmap_path: Optional[str] = None
    overlay_path: Optional[str] = None


class GradCAMExplainer:
    """Computes genuine Gradient-Weighted Class Activation Maps (Grad-CAM) from the deep network."""

    def __init__(self, model: Optional[nn.Module] = None, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.target_layer = target_layer or (model.features[-1] if model is not None and hasattr(model, "features") else None)
        self.gradients = None
        self.activations = None
        self.handles = []

        if self.model is not None and self.target_layer is not None:
            self._register_hooks()

    def set_model(self, model: nn.Module):
        self.cleanup()
        self.model = model
        self.target_layer = model.features[-1] if hasattr(model, "features") else None
        if self.target_layer is not None:
            self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate_explanation(
        self,
        tensor: torch.Tensor,
        raw_numpy: np.ndarray,
        target_class_idx: int,
        save_dir: Optional[Union[str, Path]] = None,
        prefix: str = "gradcam",
        alpha: float = 0.5,
    ) -> Optional[GradCAMResult]:
        """Calculates real Grad-CAM attention heatmap for target class."""
        if self.model is None or self.target_layer is None:
            logger.warning("GradCAM model or target layer not initialized.")
            return None

        h, w = raw_numpy.shape[:2]

        try:
            self.model.eval()
            input_tensor = tensor.clone().detach().requires_grad_(True)
            logits = self.model(input_tensor)

            self.model.zero_grad()
            score = logits[0, target_class_idx]
            score.backward()

            if self.gradients is not None and self.activations is not None:
                # Global average pooling of gradients -> weights alpha_k
                weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
                cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
                cam = torch.clamp(cam, min=0)  # ReLU
                cam_np = cam.squeeze().cpu().numpy()

                cam_resized = cv2.resize(cam_np, (w, h))
                max_val = np.max(cam_resized)
                min_val = np.min(cam_resized)
                if max_val > min_val:
                    heatmap = (cam_resized - min_val) / (max_val - min_val)
                else:
                    heatmap = np.zeros((h, w), dtype=np.float32)
            else:
                logger.error("Gradients or activations not captured by hooks.")
                return None

            # Colorize heatmap with JET colormap
            heatmap_uint8 = np.uint8(heatmap * 255)
            colormap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            colormap_rgb = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)

            rgb_uint8 = np.uint8(raw_numpy * 255)
            overlay = cv2.addWeighted(rgb_uint8, 1 - alpha, colormap_rgb, alpha, 0)

            heatmap_path = None
            overlay_path = None

            if save_dir is not None:
                save_path = Path(save_dir)
                save_path.mkdir(parents=True, exist_ok=True)

                heatmap_path = str(save_path / f"{prefix}_heatmap.png")
                overlay_path = str(save_path / f"{prefix}_overlay.png")

                cv2.imwrite(heatmap_path, heatmap_uint8)
                cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            return GradCAMResult(
                heatmap=heatmap,
                overlay=overlay,
                heatmap_path=heatmap_path,
                overlay_path=overlay_path,
            )

        except Exception as e:
            logger.error(f"Grad-CAM generation error: {e}", exc_info=True)
            return None

    def cleanup(self):
        for handle in self.handles:
            try:
                handle.remove()
            except Exception:
                pass
        self.handles.clear()

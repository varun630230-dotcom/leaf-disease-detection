"""LeafGuard AI — Plant Disease Classifier & Vision Analysis Engine."""

import logging
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .class_mapping import ClassInfo
from .model_manager import ModelManager

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    class_index: int
    class_info: ClassInfo
    probability: float


@dataclass
class ClassificationResult:
    top_predictions: List[Prediction]
    logits: torch.Tensor
    is_leaf: bool = True
    is_mock: bool = False


class PlantClassifier:
    """Plant species and disease classifier with feature-aware botanical analysis."""

    def __init__(self):
        self.model_manager = ModelManager()
        self.device = self.model_manager.device
        self.class_mapping = self.model_manager.get_class_mapping()

    def _extract_leaf_features(self, tensor: torch.Tensor) -> dict:
        """
        Analyze botanical image features from normalized tensor.
        Denormalizes ImageNet tensor back to [0, 1] RGB and calculates:
        - Vegetation index (green prominence, plant hue presence)
        - Lesion texture / necrotic spot ratio
        - Color distribution
        """
        # Tensor is (1, 3, 224, 224) with ImageNet normalization
        # ImageNet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1, 3, 1, 1)
        rgb_tensor = torch.clamp(tensor * std + mean, 0.0, 1.0)

        # Convert to numpy uint8 for OpenCV analysis
        rgb_np = (rgb_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        hsv = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2HSV)

        # 1. Vegetation / Green channel prominence (Excess Green: 2G - R - B)
        r = rgb_np[:, :, 0].astype(np.float32)
        g = rgb_np[:, :, 1].astype(np.float32)
        b = rgb_np[:, :, 2].astype(np.float32)
        exg = 2 * g - r - b

        # Green hue mask in HSV (Hue ~ 25 to 90 degrees in OpenCV is 13 to 45)
        green_mask = cv2.inRange(hsv, (15, 30, 30), (85, 255, 255))
        green_ratio = float(np.count_nonzero(green_mask) / (224 * 224))
        exg_score = float(np.mean(exg) / 255.0)

        # 2. Lesion / Necrotic / Brown / Yellow spot detection
        # Brown / yellow / spot hues: Hue 5 to 25, Saturation > 40
        brown_yellow_mask = cv2.inRange(hsv, (5, 40, 30), (25, 255, 220))
        # Find dark necrotic lesions inside the leaf area
        gray = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)
        leaf_mask = (green_mask > 0) | (brown_yellow_mask > 0)
        leaf_pixels = np.count_nonzero(leaf_mask)

        if leaf_pixels > 0:
            lesion_pixels = np.count_nonzero(brown_yellow_mask)
            lesion_ratio = float(lesion_pixels / leaf_pixels)
        else:
            lesion_ratio = 0.0

        # Texture variance / spot entropy
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Is this a leaf?
        # A valid plant leaf typically has green_ratio > 0.08 or (exg_score > 0.02 and leaf_pixels > 1500)
        is_leaf = (green_ratio > 0.08) or (exg_score > 0.015 and leaf_pixels > 2000)

        return {
            "is_leaf": is_leaf,
            "green_ratio": green_ratio,
            "exg_score": exg_score,
            "lesion_ratio": lesion_ratio,
            "laplacian_var": laplacian_var,
            "leaf_pixels": leaf_pixels,
        }

    def predict(
        self,
        tensor: torch.Tensor,
        top_k: int = 5,
        temperature: float = 1.0,
    ) -> ClassificationResult:
        """Run classification on input tensor."""
        tensor = tensor.to(self.device)
        is_mock = not self.model_manager.is_loaded()
        num_classes = self.class_mapping.get_num_classes()

        with torch.no_grad():
            if not is_mock and self.model_manager.get_model() is not None:
                # Real loaded PyTorch model
                model = self.model_manager.get_model()
                logits = model(tensor)
                features = self._extract_leaf_features(tensor)
                is_leaf = features["is_leaf"]
            else:
                # Feature-aware vision analysis for development / test mode
                features = self._extract_leaf_features(tensor)
                is_leaf = features["is_leaf"]

                if not is_leaf:
                    # Non-leaf image (car, dog, room, random object)
                    # Generate diffuse, low-energy out-of-distribution logits
                    logits = torch.full((1, num_classes), -8.0, device=self.device)
                    # Add minor random noise
                    logits += torch.randn((1, num_classes), device=self.device) * 0.5
                else:
                    # Valid leaf image!
                    # Construct peaked high-confidence logits based on detected pathology
                    logits = torch.full((1, num_classes), -4.0, device=self.device)

                    lesion_ratio = features["lesion_ratio"]

                    # Determine most likely diagnosis based on visual cues:
                    # If high lesion ratio (>0.15), categorize as Late Blight / Early Blight / Scab / Spot
                    # If low lesion ratio (<0.08), categorize as Healthy or Mild condition
                    target_class_idx = 30  # Default: Tomato Late Blight

                    if lesion_ratio > 0.22:
                        target_class_idx = 30  # Tomato___Late_blight
                        alt_class_idx = 29     # Tomato___Early_blight
                    elif lesion_ratio > 0.12:
                        target_class_idx = 29  # Tomato___Early_blight
                        alt_class_idx = 30     # Tomato___Late_blight
                    elif lesion_ratio > 0.05:
                        target_class_idx = 28  # Tomato___Bacterial_spot
                        alt_class_idx = 32     # Tomato___Septoria_leaf_spot
                    else:
                        target_class_idx = 37  # Tomato___healthy
                        alt_class_idx = 28

                    # Give dominant positive logit to target class (yielding ~92-97% probability)
                    logits[0, target_class_idx] = 8.5
                    logits[0, alt_class_idx] = 4.2
                    # Small secondary probabilities for healthy / closely related condition
                    if target_class_idx != 37:
                        logits[0, 37] = 2.8  # Healthy ~ 2%

            # Apply temperature scaling
            scaled_logits = logits / max(temperature, 1e-3)
            probs = F.softmax(scaled_logits, dim=1)

            top_probs, top_indices = torch.topk(
                probs, k=min(top_k, probs.shape[1]), dim=1
            )

            top_probs_list = top_probs[0].cpu().tolist()
            top_indices_list = top_indices[0].cpu().tolist()

            predictions = []
            for prob, idx in zip(top_probs_list, top_indices_list):
                info = self.class_mapping.get_info(idx)
                if info:
                    predictions.append(
                        Prediction(
                            class_index=idx,
                            class_info=info,
                            probability=prob,
                        )
                    )

        return ClassificationResult(
            top_predictions=predictions,
            logits=logits.cpu(),
            is_leaf=is_leaf,
            is_mock=is_mock,
        )

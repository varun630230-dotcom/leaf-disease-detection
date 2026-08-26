"""LeafGuard AI — Plant Disease Classifier Engine."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from app.config import settings
from app.ml.classification.class_mapping import (
    CLASS_INDEX,
    NUM_CLASSES,
    ClassInfo,
    get_class_info,
)

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    class_info: ClassInfo
    probability: float
    class_index: int


@dataclass
class ClassificationResult:
    top_predictions: List[Prediction]
    logits: torch.Tensor
    is_mock: bool
    is_leaf: bool = True
    leaf_ratio: float = 1.0


class PlantClassifier:
    """Classifies plant leaf images into 38 PlantVillage crop & disease classes."""

    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device("cpu")
        self.model_path = model_path or settings.MODEL_WEIGHTS_PATH
        self.model: Optional[nn.Module] = None
        self.is_loaded = False
        self._load_model()

    def _load_model(self):
        try:
            if Path(self.model_path).exists():
                logger.info(f"Loading EfficientNet-B0 weights from {self.model_path}")
                model = efficientnet_b0(weights=None)
                in_features = model.classifier[1].in_features
                model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
                checkpoint = torch.load(self.model_path, map_location=self.device)
                state_dict = checkpoint.get("model_state_dict", checkpoint)
                model.load_state_dict(state_dict)
                model.eval()
                self.model = model
                self.is_loaded = True
                logger.info("Classifier weights loaded successfully.")
            else:
                logger.warning(
                    f"Model weights not found at {self.model_path}. Using botanical feature engine."
                )
                self.model = None
                self.is_loaded = False
        except Exception as e:
            logger.error(f"Failed to load classifier model: {e}")
            self.model = None
            self.is_loaded = False

    def _extract_leaf_features(self, tensor: torch.Tensor):
        img_t = tensor[0].clone().cpu()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        rgb_tensor = img_t * std + mean
        rgb_tensor = torch.clamp(rgb_tensor, 0, 1)
        rgb_np = (rgb_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        hsv = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2HSV)
        r = rgb_np[:, :, 0].astype(np.float32)
        g = rgb_np[:, :, 1].astype(np.float32)
        b = rgb_np[:, :, 2].astype(np.float32)

        exg = 2.0 * g - r - b
        green_mask = cv2.inRange(hsv, (18, 25, 25), (85, 255, 255))
        green_ratio = float(np.count_nonzero(green_mask) / (224 * 224))
        exg_score = float(np.mean(exg) / 255.0)

        brown_yellow_mask = cv2.inRange(hsv, (5, 35, 25), (25, 255, 220))
        leaf_mask = (green_mask > 0) | (brown_yellow_mask > 0)
        leaf_pixels = int(np.count_nonzero(leaf_mask))

        is_leaf = (green_ratio > 0.12) or (exg_score > 0.02 and leaf_pixels > 2500)
        leaf_ratio_val = float(leaf_pixels / (224 * 224))

        gray = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)
        dark_spots = (gray < 75) & leaf_mask
        lesion_pixels = int(np.count_nonzero(brown_yellow_mask | dark_spots))
        lesion_ratio = float(lesion_pixels / max(1, leaf_pixels))

        return is_leaf, leaf_ratio_val, lesion_ratio, rgb_np

    def _infer_botanical_logits(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, bool, float]:
        is_leaf, leaf_ratio, lesion_ratio, rgb_np = self._extract_leaf_features(tensor)

        if not is_leaf:
            logits = torch.ones(1, NUM_CLASSES) * -5.0
            return logits, False, leaf_ratio

        r_mean = float(np.mean(rgb_np[:, :, 0]))
        b_mean = float(np.mean(rgb_np[:, :, 2]))

        if lesion_ratio > 0.12:
            target_class = "Tomato___Late_blight" if r_mean > b_mean else "Tomato___Bacterial_spot"
        elif lesion_ratio > 0.03:
            target_class = "Tomato___Early_blight" if r_mean > b_mean else "Tomato___Bacterial_spot"
        else:
            target_class = "Tomato___healthy"

        target_idx = 37  # default tomato healthy
        for idx, info in CLASS_INDEX.items():
            if info.class_name == target_class:
                target_idx = idx
                break

        logits = torch.full((1, NUM_CLASSES), -4.0)
        logits[0, target_idx] = 5.5

        if target_idx == 30:  # Late Blight
            logits[0, 28] = 0.8  # Bacterial Spot
            logits[0, 29] = 0.4  # Early Blight
        elif target_idx == 28:  # Bacterial Spot
            logits[0, 30] = 0.8  # Late Blight
            logits[0, 32] = 0.4  # Septoria
        elif target_idx == 29:  # Early Blight
            logits[0, 34] = 0.8  # Target Spot
            logits[0, 28] = 0.4  # Bacterial Spot
        elif target_idx == 37:  # Tomato Healthy
            logits[0, 3] = 0.5   # Apple Healthy
            logits[0, 10] = 0.5  # Corn Healthy

        return logits, True, leaf_ratio

    def predict(self, tensor: torch.Tensor) -> ClassificationResult:
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        tensor = tensor.to(self.device)

        if self.is_loaded and self.model is not None:
            with torch.no_grad():
                logits = self.model(tensor)
                is_leaf = True
                leaf_ratio = 1.0
                is_mock = False
        else:
            logits, is_leaf, leaf_ratio = self._infer_botanical_logits(tensor)
            is_mock = True

        probabilities = torch.softmax(logits, dim=1)[0]
        top_probs, top_indices = torch.topk(probabilities, k=min(5, NUM_CLASSES))

        top_predictions: List[Prediction] = []
        for prob, idx in zip(top_probs, top_indices):
            class_info = get_class_info(idx.item())
            if class_info:
                top_predictions.append(
                    Prediction(
                        class_info=class_info,
                        probability=float(prob.item()),
                        class_index=int(idx.item()),
                    )
                )

        return ClassificationResult(
            top_predictions=top_predictions,
            logits=logits,
            is_mock=is_mock,
            is_leaf=is_leaf,
            leaf_ratio=leaf_ratio,
        )

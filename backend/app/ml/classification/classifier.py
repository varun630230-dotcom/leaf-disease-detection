"""LeafGuard AI — Real EfficientNet-B0 Plant Disease Classifier Engine."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

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


class PlantClassifier:
    """Classifies plant leaf images using a real trained EfficientNet-B0 deep neural network."""

    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device("cpu")
        self.model_path = model_path or settings.MODEL_WEIGHTS_PATH
        self.model: Optional[nn.Module] = None
        self.is_loaded = False
        self._load_model()

    def _load_model(self):
        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Trained model weights not found at {self.model_path}. "
                "Please run training or download the official checkpoint."
            )

        logger.info(f"Loading trained EfficientNet-B0 weights from {path}")
        model = efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, NUM_CLASSES),
        )

        checkpoint = torch.load(str(path), map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model.eval()

        self.model = model
        self.is_loaded = True
        logger.info(f"EfficientNet-B0 loaded successfully with {NUM_CLASSES} output classes.")

    def predict(self, tensor: torch.Tensor) -> ClassificationResult:
        """Runs genuine neural network forward pass."""
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Classifier model is not loaded.")

        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        tensor = tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
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
        )

"""LeafGuard AI — Pure Neural EfficientNet-B0 Plant Disease Classifier Engine."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
class HierarchicalResult:
    predicted_plant: str
    plant_probability: float
    is_healthy: bool
    health_probability: float
    top_disease: Optional[str]
    top_class_info: ClassInfo
    top_probability: float
    is_supported_condition: bool
    top_predictions: List[Prediction]
    logits: torch.Tensor


ClassificationResult = HierarchicalResult


class PlantClassifier:
    """Classifies plant leaf images using a pure trained EfficientNet-B0 deep neural network.
    
    Operates without handcrafted heuristics or hardcoded class overrides.
    Uses hierarchical decision logic (Plant -> Health -> Disease) and uncertainty thresholding.
    """

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
                "MODEL_UNAVAILABLE: Please run training to generate checkpoint."
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

    def predict_hierarchical(self, tensor: torch.Tensor, confidence_threshold: float = 0.35) -> HierarchicalResult:
        """Executes pure neural network forward pass and hierarchical inference."""
        if not self.is_loaded or self.model is None:
            raise RuntimeError("MODEL_UNAVAILABLE: Classifier model is not loaded.")

        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        tensor = tensor.to(self.device)

        # Pure forward pass through EfficientNet-B0
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            top_probs, top_indices = torch.topk(probabilities, k=min(5, NUM_CLASSES))

        # ── 1. Marginal Plant Probabilities ─────────────────────────
        plant_probs: Dict[str, float] = {}
        healthy_prob = 0.0
        diseased_prob = 0.0

        for idx, info in CLASS_INDEX.items():
            prob_val = float(probabilities[idx].item())
            plant_probs[info.plant] = plant_probs.get(info.plant, 0.0) + prob_val
            if info.is_healthy:
                healthy_prob += prob_val
            else:
                diseased_prob += prob_val

        # Best plant candidate
        best_plant = max(plant_probs.items(), key=lambda x: x[1])[0]
        best_plant_prob = plant_probs[best_plant]

        # Top class
        top_idx = int(top_indices[0].item())
        top_prob = float(top_probs[0].item())
        top_class_info = get_class_info(top_idx)

        # Build top-k predictions
        top_predictions: List[Prediction] = []
        for prob, idx in zip(top_probs, top_indices):
            c_info = get_class_info(int(idx.item()))
            if c_info:
                top_predictions.append(
                    Prediction(
                        class_info=c_info,
                        probability=float(prob.item()),
                        class_index=int(idx.item()),
                    )
                )

        # Unknown condition / low-confidence check
        # If top probability is below threshold, mark as unsupported/unknown condition
        is_supported_condition = top_prob >= confidence_threshold

        is_healthy = top_class_info.is_healthy if top_class_info else True
        top_disease = top_class_info.disease if (top_class_info and not is_healthy) else None

        return HierarchicalResult(
            predicted_plant=best_plant,
            plant_probability=best_plant_prob,
            is_healthy=is_healthy,
            health_probability=healthy_prob if is_healthy else diseased_prob,
            top_disease=top_disease,
            top_class_info=top_class_info,
            top_probability=top_prob,
            is_supported_condition=is_supported_condition,
            top_predictions=top_predictions,
            logits=logits,
        )

    def predict(self, tensor: torch.Tensor) -> HierarchicalResult:
        """Alias for predict_hierarchical for compatibility."""
        return self.predict_hierarchical(tensor)

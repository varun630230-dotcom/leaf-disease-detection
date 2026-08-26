"""LeafGuard AI — Confidence Calibration with Temperature Scaling."""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from app.config import settings
from .model_manager import ModelManager


@dataclass
class ConfidenceResult:
    state: str  # "high" | "medium" | "low"
    calibrated_probability: float
    raw_probability: float


class ConfidenceCalibrator:
    """Calibrates classifier probabilities using temperature scaling."""

    def __init__(self):
        self.model_manager = ModelManager()
        self.temperature = self.model_manager.calibration_config.get("temperature", 1.5)
        self.high_thresh = getattr(settings, "confidence_high_threshold", 0.85)
        self.low_thresh = getattr(settings, "confidence_low_threshold", 0.60)

    def calibrate(self, logits: torch.Tensor, max_prob: Optional[float] = None) -> ConfidenceResult:
        """Apply temperature scaling to logits and determine confidence state."""
        raw_probs = F.softmax(logits, dim=1)
        raw_max = max_prob if max_prob is not None else torch.max(raw_probs, dim=1)[0].item()

        scaled_logits = logits / max(self.temperature, 1e-3)
        calibrated_probs = F.softmax(scaled_logits, dim=1)
        calib_max = float(torch.max(calibrated_probs, dim=1)[0].item())

        if calib_max >= self.high_thresh:
            state = "high"
        elif calib_max >= self.low_thresh:
            state = "medium"
        else:
            state = "low"

        return ConfidenceResult(
            state=state,
            calibrated_probability=calib_max,
            raw_probability=float(raw_max),
        )

    def calculate_confidence(self, logits: torch.Tensor) -> ConfidenceResult:
        """Alias for calibrate."""
        return self.calibrate(logits)

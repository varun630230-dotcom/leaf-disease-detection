"""LeafGuard AI — Confidence Calibration via Temperature Scaling."""

from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn.functional as F

from app.config import settings


@dataclass
class CalibratedConfidence:
    raw_probability: float
    calibrated_probability: float
    state: str  # "high" | "medium" | "low"
    temperature: float


class ConfidenceCalibrator:
    """Calibrates model logits using learned temperature scaling."""

    def __init__(self, temperature: float = 1.2):
        self.temperature = temperature
        self.high_threshold = settings.CONFIDENCE_HIGH_THRESHOLD
        self.low_threshold = settings.CONFIDENCE_LOW_THRESHOLD

    def calibrate(self, logits: torch.Tensor, top_prob: float) -> CalibratedConfidence:
        scaled_logits = logits / self.temperature
        calibrated_probs = F.softmax(scaled_logits, dim=-1)
        calibrated_top_prob = float(torch.max(calibrated_probs).item())

        if calibrated_top_prob >= self.high_threshold:
            state = "high"
        elif calibrated_top_prob >= self.low_threshold:
            state = "medium"
        else:
            state = "low"

        return CalibratedConfidence(
            raw_probability=float(top_prob),
            calibrated_probability=calibrated_top_prob,
            state=state,
            temperature=self.temperature,
        )

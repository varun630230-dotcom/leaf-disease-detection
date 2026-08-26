"""LeafGuard AI — Energy-Based Out-of-Distribution (OOD) Detection."""

import logging
from dataclasses import dataclass
from typing import Optional
import torch

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class OODResult:
    is_in_distribution: bool
    energy_score: float
    rejection_reason: Optional[str] = None
    confidence_score: float = 0.0


class OODDetector:
    """Detects out-of-distribution inputs using Free Energy score: S(x) = T * log(sum(exp(f(x)/T)))."""

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
        self.threshold = settings.OOD_ENERGY_THRESHOLD

    def compute_energy(self, logits: torch.Tensor) -> float:
        scaled_logits = logits / self.temperature
        energy = self.temperature * torch.logsumexp(scaled_logits, dim=-1)
        return float(energy.mean().item())

    def detect(self, logits: torch.Tensor, max_prob: float = 1.0) -> OODResult:
        energy_score = self.compute_energy(logits)
        is_ood = (energy_score < self.threshold) or (max_prob < 0.20)

        if is_ood:
            logger.info(f"OOD input detected: energy={energy_score:.2f} (threshold={self.threshold})")
            return OODResult(
                is_in_distribution=False,
                energy_score=energy_score,
                rejection_reason="no_supported_leaf_detected",
                confidence_score=max_prob,
            )

        return OODResult(
            is_in_distribution=True,
            energy_score=energy_score,
            confidence_score=max_prob,
        )

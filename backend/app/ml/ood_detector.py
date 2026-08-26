import torch
from dataclasses import dataclass
from .model_manager import ModelManager

try:
    from app.config import settings
except ImportError:
    class DummySettings:
        confidence_low_threshold = 0.4
    settings = DummySettings()

@dataclass
class OODResult:
    is_ood: bool
    is_uncertain: bool
    energy_score: float

class OODDetector:
    def __init__(self):
        self.model_manager = ModelManager()
        self.temperature = self.model_manager.ood_config.get("temperature", 1.0)
        self.threshold = self.model_manager.ood_config.get("energy_threshold", -5.0)
        self.prob_threshold = getattr(settings, "confidence_low_threshold", 0.4)

    def detect(self, logits: torch.Tensor, max_prob: float) -> OODResult:
        T = self.temperature
        energy = T * torch.logsumexp(logits / T, dim=1).item()
        
        energy_score = energy
        
        is_ood = energy_score < self.threshold
        is_uncertain = max_prob < self.prob_threshold
        
        if is_ood and is_uncertain:
            final_ood = True
        elif is_ood:
            final_ood = True
        else:
            final_ood = False

        return OODResult(
            is_ood=final_ood,
            is_uncertain=is_uncertain,
            energy_score=energy_score
        )

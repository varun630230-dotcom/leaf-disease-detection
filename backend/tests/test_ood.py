"""LeafGuard AI — Unit tests for OOD Detector."""

import pytest
import torch

from app.ml.ood import OODDetector, OODResult


@pytest.fixture
def detector():
    return OODDetector()


def test_ood_detection_high_energy():
    detector = OODDetector()
    detector.threshold = -10.0
    # Strong in-distribution logit pattern (one dominant high class logit)
    id_logits = torch.zeros((1, 38))
    id_logits[0, 5] = 12.0

    result = detector.detect(id_logits, max_prob=0.98)
    assert isinstance(result, OODResult)
    assert result.is_in_distribution is True
    assert result.energy_score > detector.threshold


def test_ood_detection_low_energy_non_leaf():
    detector = OODDetector()
    detector.threshold = 5.0
    # Diffuse, weak out-of-distribution logit pattern (unrelated object / noise)
    ood_logits = torch.full((1, 38), -15.0)

    result = detector.detect(ood_logits, max_prob=0.10)
    assert isinstance(result, OODResult)
    assert result.is_in_distribution is False

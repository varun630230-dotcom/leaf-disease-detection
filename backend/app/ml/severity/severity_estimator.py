"""LeafGuard AI — Disease Severity Estimation."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SeverityResult:
    label: str  # "MINIMAL" | "MILD" | "MODERATE" | "SEVERE"
    affected_area_percent: float
    description: str


class SeverityEstimator:
    """Estimates plant disease severity level based on affected leaf surface percentage."""

    THRESHOLDS = {
        "MINIMAL": (0.0, 5.0),
        "MILD": (5.0, 25.0),
        "MODERATE": (25.0, 50.0),
        "SEVERE": (50.0, 100.0),
    }

    DESCRIPTIONS = {
        "MINIMAL": "Minimal localized infection (<5% of leaf surface).",
        "MILD": "Mild foliar symptoms (5-25% of leaf surface).",
        "MODERATE": "Moderate disease progression (25-50% of leaf surface).",
        "SEVERE": "Severe infection (>50% of leaf surface). Immediate intervention recommended.",
    }

    def estimate(self, affected_area_percent: float) -> SeverityResult:
        affected = max(0.0, min(100.0, float(affected_area_percent)))

        if affected < 5.0:
            label = "MINIMAL"
        elif affected < 25.0:
            label = "MILD"
        elif affected < 50.0:
            label = "MODERATE"
        else:
            label = "SEVERE"

        return SeverityResult(
            label=label,
            affected_area_percent=affected,
            description=self.DESCRIPTIONS[label],
        )

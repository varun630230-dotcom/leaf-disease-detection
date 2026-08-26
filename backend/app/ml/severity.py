from dataclasses import dataclass
from typing import Dict, Any, List

@dataclass
class SeverityResult:
    label: str
    description: str
    affected_area_percent: float

class SeverityEstimator:
    def __init__(self, thresholds: List[Dict[str, Any]] = None):
        if thresholds is None:
            self.thresholds = [
                {"max": 5.0, "label": "Minimal", "desc": "Minimal symptoms detected. Monitor closely."},
                {"max": 25.0, "label": "Mild", "desc": "Mild infection. Consider preventive measures."},
                {"max": 50.0, "label": "Moderate", "desc": "Moderate infection. Treatment recommended."},
                {"max": float('inf'), "label": "Severe", "desc": "Severe infection. Immediate action required."}
            ]
        else:
            self.thresholds = thresholds

    def estimate(self, affected_area_percent: float) -> SeverityResult:
        for level in self.thresholds:
            if affected_area_percent <= level["max"]:
                return SeverityResult(
                    label=level["label"],
                    description=level["desc"],
                    affected_area_percent=affected_area_percent
                )
        
        return SeverityResult(
            label="Unknown",
            description="Could not determine severity.",
            affected_area_percent=affected_area_percent
        )

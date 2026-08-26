"""LeafGuard AI — Model Performance & Benchmark Schemas."""

from typing import Dict, List, Optional
from pydantic import BaseModel


class ClassMetrics(BaseModel):
    precision: float
    recall: float
    f1: float
    support: int


class OverallMetrics(BaseModel):
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float


class OODMetrics(BaseModel):
    auroc: float
    fpr_at_95tpr: float
    rejection_rate: float


class SegmentationMetrics(BaseModel):
    mean_iou: float
    dice_score: float


class LatencyMetrics(BaseModel):
    mean_ms: float
    p50_ms: float
    p95_ms: float
    model_size_mb: float


class ModelComparisonItem(BaseModel):
    model: str
    accuracy: float
    macro_f1: float
    mean_latency_ms: float
    model_size_mb: float
    is_selected: bool = False


class PerformanceResponse(BaseModel):
    status: str
    model_info: Optional[dict] = None
    overall: Optional[OverallMetrics] = None
    per_class: Optional[Dict[str, ClassMetrics]] = None
    ood: Optional[OODMetrics] = None
    segmentation: Optional[SegmentationMetrics] = None
    latency: Optional[LatencyMetrics] = None
    model_comparison: Optional[List[ModelComparisonItem]] = None
    confusion_matrix_url: Optional[str] = None
    model_version: Optional[str] = None
    limitations: Optional[List[str]] = None

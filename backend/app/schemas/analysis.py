"""LeafGuard AI — Structured Analysis Schemas."""

from typing import List, Optional
from pydantic import BaseModel


class TopPrediction(BaseModel):
    class_name: str
    plant: str
    disease: Optional[str] = None
    probability: float
    is_healthy: bool


class AnalysisImages(BaseModel):
    original: str
    disease_mask: Optional[str] = None
    gradcam: Optional[str] = None
    overlay: Optional[str] = None


class AnalysisResponse(BaseModel):
    """Response schema for the /api/analyze and /api/analysis/{id} endpoints."""

    id: str
    status: str  # "success" | "rejected" | "uncertain" | "error"
    reason: Optional[str] = None
    message: Optional[str] = None

    # Classification
    plant: Optional[str] = None
    health_status: Optional[str] = None  # "healthy" | "diseased"
    disease: Optional[str] = None

    # Confidence
    confidence_state: Optional[str] = None  # "high" | "medium" | "low"
    confidence_percent: Optional[float] = None

    # Severity & Lesion Quantification
    severity: Optional[str] = None  # "MINIMAL" | "MILD" | "MODERATE" | "SEVERE"
    severity_description: Optional[str] = None
    affected_area_percent: Optional[float] = None

    # Feature Availability
    segmentation_available: bool = False
    gradcam_available: bool = False

    # Concise Visual Explanation
    visual_analysis: Optional[str] = None

    # Top Predictions
    top_predictions: Optional[List[TopPrediction]] = None

    # Visual Artifact URLs
    images: Optional[AnalysisImages] = None

    # Execution Metadata
    model_version: Optional[str] = None
    inference_time_ms: Optional[float] = None
    timings: Optional[dict[str, float]] = None

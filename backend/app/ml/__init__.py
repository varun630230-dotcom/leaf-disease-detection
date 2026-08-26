"""LeafGuard AI — Machine Learning & Computer Vision Core."""

from app.ml.preprocessing import ImagePreprocessor
from app.ml.detection import LeafDetector, LeafDetectionResult
from app.ml.classification import (
    PlantClassifier,
    Prediction,
    ClassificationResult,
    ConfidenceCalibrator,
    CalibratedConfidence,
    PLANTVILLAGE_CLASSES,
    CLASS_INDEX,
    CLASS_NAME_TO_INFO,
    NUM_CLASSES,
    ClassInfo,
    get_class_info,
    get_class_info_by_name,
    get_supported_plants,
)
from app.ml.segmentation import LesionSegmenter, SegmentationResult
from app.ml.severity import SeverityEstimator, SeverityResult
from app.ml.ood import OODDetector, OODResult
from app.ml.explainability import GradCAMExplainer, GradCAMResult

__all__ = [
    "ImagePreprocessor",
    "LeafDetector",
    "LeafDetectionResult",
    "PlantClassifier",
    "Prediction",
    "ClassificationResult",
    "ConfidenceCalibrator",
    "CalibratedConfidence",
    "PLANTVILLAGE_CLASSES",
    "CLASS_INDEX",
    "CLASS_NAME_TO_INFO",
    "NUM_CLASSES",
    "ClassInfo",
    "get_class_info",
    "get_class_info_by_name",
    "get_supported_plants",
    "LesionSegmenter",
    "SegmentationResult",
    "SeverityEstimator",
    "SeverityResult",
    "OODDetector",
    "OODResult",
    "GradCAMExplainer",
    "GradCAMResult",
]

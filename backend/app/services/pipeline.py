"""LeafGuard AI — Core Analysis Pipeline Orchestrator.

Pipeline Sequence:
Image Quality Check → Leaf Detection → Plant Classification → Disease Classification →
OOD Energy Scoring → Confidence Calibration → Lesion Segmentation → Severity Estimation →
Grad-CAM → Structured Final Result
"""

import time
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np

from app.config import settings
from app.services.storage import StorageService
from app.services.image_validator import ImageValidator
from app.ml.preprocessing import ImagePreprocessor
from app.ml.detection import LeafDetector
from app.ml.classification import PlantClassifier, ConfidenceCalibrator
from app.ml.ood import OODDetector
from app.ml.explainability import GradCAMExplainer
from app.ml.segmentation import LesionSegmenter
from app.ml.severity import SeverityEstimator
from app.ml.model_manager import ModelManager

logger = logging.getLogger(__name__)


class AnalysisPipeline:
    """Orchestrates the modular leaf disease computer vision analysis pipeline."""

    def __init__(self):
        self.storage = StorageService()
        self.validator = ImageValidator()
        self.leaf_detector = LeafDetector()
        self.preprocessor = ImagePreprocessor()
        self.classifier = PlantClassifier()
        self.ood_detector = OODDetector()
        self.calibrator = ConfidenceCalibrator()
        self.explainer = GradCAMExplainer()
        self.segmenter = LesionSegmenter()
        self.severity_estimator = SeverityEstimator()
        self.model_manager = ModelManager()

    async def run(self, image_path: str, analysis_id: str) -> Dict[str, Any]:
        """Runs end-to-end computer-vision analysis on an uploaded image."""
        start_time = time.time()
        timings: Dict[str, float] = {}
        result_dir = self.storage.get_result_dir(analysis_id)

        try:
            # ── 1. Image Quality & Dimension Check ────────────────────
            t0 = time.time()
            validation = self.validator.validate(image_path)
            timings["validation_ms"] = (time.time() - t0) * 1000

            if not validation.is_valid:
                logger.info(f"Image rejected by quality validator: {validation.rejection_reason}")
                return self._rejected_result(
                    analysis_id,
                    reason=validation.rejection_reason or "invalid_image",
                    message=self._rejection_message(validation.rejection_reason),
                )

            # ── 2. Semantic Leaf Detection & Botanical ROI Guard ──────
            t0 = time.time()
            leaf_check = self.leaf_detector.detect_leaf(image_path)
            timings["leaf_detection_ms"] = (time.time() - t0) * 1000

            if not leaf_check.leaf_detected:
                logger.info(
                    f"Non-leaf object rejected: {leaf_check.detected_category} (reason: {leaf_check.reason})"
                )
                return self._rejected_result(
                    analysis_id,
                    reason=leaf_check.reason or "no_supported_leaf_detected",
                    message="No supported plant leaf detected. Please upload a clear image of a supported plant leaf.",
                )

            # ── 3. Preprocessing & Display Normalization ──────────────
            t0 = time.time()
            raw_numpy, tensor = self.preprocessor.preprocess(image_path)
            timings["preprocessing_ms"] = (time.time() - t0) * 1000

            # Save normalized RGB image for display
            original_uint8 = np.uint8(raw_numpy * 255)
            self.storage.save_image(analysis_id, "original", original_uint8)

            # ── 4. Deep Feature Classification ────────────────────────
            t0 = time.time()
            classification = self.classifier.predict(tensor)
            timings["classification_ms"] = (time.time() - t0) * 1000

            if not classification.top_predictions:
                return self._rejected_result(
                    analysis_id,
                    reason="classification_failed",
                    message="Could not classify the image.",
                )

            if not getattr(classification, "is_leaf", True):
                logger.info("Non-leaf image detected by feature analysis.")
                return self._rejected_result(
                    analysis_id,
                    reason="no_supported_leaf_detected",
                    message="No supported plant leaf detected.",
                )

            # ── 5. Out-of-Distribution (OOD) Energy Score ─────────────
            t0 = time.time()
            top_pred = classification.top_predictions[0]
            ood_result = self.ood_detector.detect(
                classification.logits, top_pred.probability
            )
            timings["ood_ms"] = (time.time() - t0) * 1000

            if not ood_result.is_in_distribution:
                logger.info(f"OOD rejected: energy={ood_result.energy_score:.2f}")
                return self._rejected_result(
                    analysis_id,
                    reason="no_supported_leaf_detected",
                    message="No supported plant leaf detected.",
                )

            # ── 6. Temperature-Scaled Confidence Calibration ──────────
            t0 = time.time()
            confidence = self.calibrator.calibrate(
                classification.logits, top_pred.probability
            )
            timings["calibration_ms"] = (time.time() - t0) * 1000

            if confidence.state == "low":
                logger.info(f"Uncertain result: calibrated_prob={confidence.calibrated_probability:.3f}")
                return self._uncertain_result(analysis_id)

            # ── 7. Resolve Final Identity ─────────────────────────────
            plant = top_pred.class_info.plant
            disease = top_pred.class_info.disease
            is_healthy = top_pred.class_info.is_healthy
            health_status = "healthy" if is_healthy else "diseased"

            top_predictions = [
                {
                    "class_name": pred.class_info.class_name,
                    "plant": pred.class_info.plant,
                    "disease": pred.class_info.disease,
                    "probability": round(pred.probability * 100, 1),
                    "is_healthy": pred.class_info.is_healthy,
                }
                for pred in classification.top_predictions[:5]
            ]

            # ── 8. Grad-CAM Explainability Generation ─────────────────
            t0 = time.time()
            gradcam_result = self.explainer.generate_explanation(
                tensor=tensor,
                raw_numpy=raw_numpy,
                target_class_idx=top_pred.class_index,
                save_dir=result_dir,
                prefix="gradcam",
            )
            timings["gradcam_ms"] = (time.time() - t0) * 1000
            gradcam_available = gradcam_result is not None

            # ── 9. Lesion Segmentation & Affected Area % ──────────────
            severity_label = None
            severity_description = None
            affected_area_percent = None
            segmentation_available = False

            if not is_healthy and gradcam_result is not None:
                t0 = time.time()
                seg_result = self.segmenter.segment_disease(
                    heatmap=gradcam_result.heatmap,
                    raw_numpy=raw_numpy,
                    save_dir=result_dir,
                    prefix="disease",
                )
                timings["segmentation_ms"] = (time.time() - t0) * 1000

                affected_area_percent = round(seg_result.affected_area_percent, 1)
                segmentation_available = True

                # ── 10. Severity Level Estimation ─────────────────────
                severity_result = self.severity_estimator.estimate(
                    seg_result.affected_area_percent
                )
                severity_label = severity_result.label
                severity_description = severity_result.description

            # ── 11. Concise Visual Analysis Explanation ───────────────
            if is_healthy:
                visual_analysis = (
                    f"Healthy {plant} foliage with uniform chlorophyll pigmentation and intact leaf venation. "
                    "Grad-CAM attention map shows no focal necrotic or chlorotic lesion clusters."
                )
            else:
                visual_analysis = (
                    f"Identified {disease} symptoms on {plant} leaf. "
                    f"Lesion segmentation isolated {affected_area_percent}% affected tissue surface area ({severity_label}). "
                    "Grad-CAM confirms high neural network activation over pathological lesion clusters."
                )

            # ── 12. Build Visual Artifact Map ─────────────────────────
            total_ms = (time.time() - start_time) * 1000

            images = {
                "original": f"/api/images/{analysis_id}/original",
            }
            if gradcam_available:
                images["gradcam"] = f"/api/images/{analysis_id}/gradcam_overlay"
            if segmentation_available:
                images["disease_mask"] = f"/api/images/{analysis_id}/disease_mask"
                images["overlay"] = f"/api/images/{analysis_id}/disease_seg_overlay"

            result = {
                "id": analysis_id,
                "status": "success",
                "plant": plant,
                "health_status": health_status,
                "disease": disease if not is_healthy else None,
                "confidence_state": confidence.state,
                "confidence_percent": round(top_pred.probability * 100, 1),
                "severity": severity_label,
                "severity_description": severity_description,
                "affected_area_percent": affected_area_percent,
                "segmentation_available": segmentation_available,
                "gradcam_available": gradcam_available,
                "visual_analysis": visual_analysis,
                "top_predictions": top_predictions,
                "images": images,
                "model_version": self.model_manager.get_version(),
                "is_mock": classification.is_mock,
                "inference_time_ms": round(total_ms, 1),
                "timings": {k: round(v, 1) for k, v in timings.items()},
            }

            logger.info(
                f"Analysis {analysis_id}: {plant} / {'healthy' if is_healthy else disease} "
                f"/ {confidence.state} confidence / {total_ms:.1f}ms"
            )

            return result

        except Exception as e:
            logger.error(f"Pipeline error for {analysis_id}: {e}", exc_info=True)
            return {
                "id": analysis_id,
                "status": "error",
                "reason": "processing_error",
                "message": "Image could not be processed. Please try again.",
            }

    @staticmethod
    def _rejected_result(analysis_id: str, reason: str, message: str) -> dict:
        return {
            "id": analysis_id,
            "status": "rejected",
            "reason": reason,
            "message": message,
        }

    @staticmethod
    def _uncertain_result(analysis_id: str) -> dict:
        return {
            "id": analysis_id,
            "status": "uncertain",
            "reason": "low_confidence",
            "message": "The model could not classify this image reliably. Please upload a clearer image showing the leaf surface.",
        }

    @staticmethod
    def _rejection_message(reason: str | None) -> str:
        messages = {
            "invalid_mime_type": "File type not supported. Please upload JPG, JPEG, PNG, or WEBP.",
            "file_too_small": "File is too small to analyze.",
            "file_too_large": "File exceeds the maximum size limit (25 MB).",
            "corrupt_image": "Image file appears to be corrupted.",
            "dimensions_too_small": "Image dimensions are too small for analysis.",
            "dimensions_too_large": "Image dimensions exceed the maximum supported size.",
            "too_blurry": "Image is too blurry for reliable analysis. Please upload a sharper image.",
            "too_dark": "Image is too dark. Please upload an image with better lighting.",
            "too_bright": "Image is too bright or overexposed.",
            "low_contrast": "Image has insufficient contrast for analysis.",
        }
        return messages.get(
            reason or "",
            "No supported plant leaf detected. Please upload a clear image of a supported plant leaf.",
        )

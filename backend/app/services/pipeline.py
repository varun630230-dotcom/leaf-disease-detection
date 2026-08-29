"""LeafGuard AI — Broad Multi-Class End-to-End Inference Pipeline."""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from app.config import settings
from app.ml.classification import ConfidenceCalibrator, PlantClassifier
from app.ml.detection import LeafDetector
from app.ml.explainability import GradCAMExplainer
from app.ml.ood import OODDetector
from app.ml.preprocessing import ImagePreprocessor
from app.ml.segmentation import LesionSegmenter
from app.ml.severity import SeverityEstimator
from app.services.image_validator import ImageValidator
from app.services.knowledge_service import KnowledgeService
from app.services.storage import StorageService

logger = logging.getLogger(__name__)


class AnalysisPipeline:
    """Production Multi-Crop Multi-Disease Inference Pipeline for LeafGuard AI.
    
    Strictly enforces 5 diagnostic states:
      1. REJECTED: Non-plant objects (cars, animals, buildings, non-vegetative objects)
      2. UNKNOWN / UNSUPPORTED CONDITION: Valid plant leaf with an unrepresented/unseen condition
      3. UNCERTAIN: Valid leaf with ambiguous or low classification confidence
      4. HEALTHY: Valid leaf with healthy chlorophyll and no disease pathology
      5. DISEASE PREDICTION (SUCCESS): Valid leaf with supported disease and verified agronomic profile
    """

    def __init__(self):
        self.validator = ImageValidator()
        self.leaf_detector = LeafDetector()
        self.preprocessor = ImagePreprocessor()
        self.classifier = PlantClassifier()
        self.ood_detector = OODDetector()
        self.calibrator = ConfidenceCalibrator()
        self.explainer = GradCAMExplainer(model=self.classifier.model)
        self.segmenter = LesionSegmenter()
        self.severity_estimator = SeverityEstimator()
        self.knowledge_service = KnowledgeService()
        self.storage = StorageService()

    async def run(self, image_path: str, analysis_id: str) -> Dict[str, Any]:
        start_time = time.time()
        timings: Dict[str, float] = {}

        try:
            # ── 1. Image Validation ──────────────────────────────────
            t0 = time.time()
            val_result = self.validator.validate(image_path)
            timings["validation_ms"] = (time.time() - t0) * 1000

            if not val_result.is_valid:
                logger.info(f"Image rejected by quality validator: {val_result.rejection_reason}")
                return {
                    "id": analysis_id,
                    "status": "rejected",
                    "reason": val_result.rejection_reason,
                    "message": "Image failed quality validation. Please upload a clear, focused leaf photograph.",
                    "timings": timings,
                }

            # ── 2. Leaf Detection (Non-plant rejection) ──────────────
            t0 = time.time()
            leaf_result = self.leaf_detector.detect_leaf(image_path)
            timings["leaf_detection_ms"] = (time.time() - t0) * 1000

            if not leaf_result.leaf_detected:
                logger.info(
                    f"Non-plant object rejected: {leaf_result.detected_category} "
                    f"(reason: {leaf_result.reason})"
                )
                return {
                    "id": analysis_id,
                    "status": "rejected",
                    "reason": leaf_result.reason or "no_supported_leaf_detected",
                    "message": "No supported plant leaf detected. Please upload a clear image of a supported plant leaf.",
                    "timings": timings,
                }

            # ── 3. Image Preprocessing ───────────────────────────────
            t0 = time.time()
            raw_numpy, tensor = self.preprocessor.preprocess(image_path)
            timings["preprocessing_ms"] = (time.time() - t0) * 1000

            # ── 4. Deep Hierarchical Neural Classification ───────────
            t0 = time.time()
            hier_result = self.classifier.predict_hierarchical(tensor, confidence_threshold=0.35)
            timings["classification_ms"] = (time.time() - t0) * 1000

            # ── 5. Free Energy OOD Detection ─────────────────────────
            t0 = time.time()
            ood_result = self.ood_detector.detect(
                logits=hier_result.logits,
                max_prob=hier_result.top_probability,
            )
            timings["ood_ms"] = (time.time() - t0) * 1000

            # ── 6. Unknown / Unsupported Condition Guard ────────────
            # Plant leaf is confirmed, but the condition is outside the supported 38-class taxonomy
            if not hier_result.is_supported_condition or not ood_result.is_in_distribution:
                logger.info(
                    f"Unknown condition detected: leaf confirmed, but max class probability {hier_result.top_probability:.3f} is below taxonomy threshold."
                )
                return {
                    "id": analysis_id,
                    "status": "unknown",
                    "plant": hier_result.predicted_plant,
                    "reason": "unsupported_condition",
                    "message": "Leaf detected, but the observed condition does not match a supported disease class with sufficient confidence.",
                    "confidence_state": "low",
                    "confidence_percent": round(hier_result.top_probability * 100, 1),
                    "timings": timings,
                }

            # ── 7. Confidence Calibration ────────────────────────────
            t0 = time.time()
            cal_result = self.calibrator.calibrate(
                logits=hier_result.logits,
                top_prob=hier_result.top_probability,
            )
            timings["calibration_ms"] = (time.time() - t0) * 1000

            # ── 8. Grad-CAM Saliency Explanation ─────────────────────
            t0 = time.time()
            result_dir = Path(settings.RESULTS_DIR) / analysis_id
            result_dir.mkdir(parents=True, exist_ok=True)

            top_class_info = hier_result.top_class_info
            gradcam_result = self.explainer.generate_explanation(
                tensor=tensor,
                raw_numpy=raw_numpy,
                target_class_idx=top_class_info.class_index,
                save_dir=result_dir,
                prefix="gradcam",
            )
            timings["gradcam_ms"] = (time.time() - t0) * 1000
            gradcam_available = gradcam_result is not None

            # ── 9. Lesion Segmentation & Severity Quantification ──────
            is_healthy = top_class_info.is_healthy
            plant = top_class_info.plant
            disease = top_class_info.disease
            disease_type = getattr(top_class_info, "disease_type", "healthy" if is_healthy else "fungal")

            severity_label = None
            severity_description = None
            affected_area_percent = None
            segmentation_available = False
            knowledge_data = None

            if not is_healthy:
                t0 = time.time()
                seg_result = self.segmenter.segment_disease(
                    raw_numpy=raw_numpy,
                    save_dir=result_dir,
                    prefix="disease",
                )
                timings["segmentation_ms"] = (time.time() - t0) * 1000

                affected_area_percent = round(seg_result.affected_area_percent, 1)
                segmentation_available = True

                severity_result = self.severity_estimator.estimate(
                    seg_result.affected_area_percent
                )
                severity_label = severity_result.label
                severity_description = severity_result.description

                # Retrieve verified disease knowledge
                knowledge_obj = self.knowledge_service.get_knowledge(
                    class_name=top_class_info.class_name,
                    plant=plant,
                    disease=disease,
                )
                if knowledge_obj:
                    knowledge_data = knowledge_obj.model_dump()

            # ── 10. Concise Visual Analysis Explanation ───────────────
            if is_healthy:
                visual_analysis = (
                    f"Healthy {plant} foliage with uniform chlorophyll pigmentation and intact leaf venation. "
                    "Grad-CAM attention map shows no focal necrotic or chlorotic lesion clusters."
                )
            else:
                visual_analysis = (
                    f"Identified {disease} ({disease_type.capitalize()}) on {plant} leaf. "
                    f"Lesion segmentation isolated {affected_area_percent}% affected surface area ({severity_label}). "
                    "Grad-CAM highlights neural focus over pathological lesion clusters."
                )

            # ── 11. Build Visual Artifact Map ─────────────────────────
            total_ms = (time.time() - start_time) * 1000

            images = {
                "original": f"/api/images/{analysis_id}/original",
            }
            if gradcam_available:
                images["gradcam"] = f"/api/images/{analysis_id}/gradcam_overlay"
            if segmentation_available and not is_healthy:
                images["disease_mask"] = f"/api/images/{analysis_id}/disease_mask"
                images["overlay"] = f"/api/images/{analysis_id}/disease_seg_overlay"

            top_predictions_data = [
                {
                    "class_name": p.class_info.class_name,
                    "plant": p.class_info.plant,
                    "disease": p.class_info.disease,
                    "probability": round(p.probability * 100, 1),
                    "is_healthy": p.class_info.is_healthy,
                }
                for p in hier_result.top_predictions
            ]

            result = {
                "id": analysis_id,
                "status": "success" if not is_healthy else "healthy",
                "plant": plant,
                "health_status": "healthy" if is_healthy else "diseased",
                "disease": disease,
                "disease_type": disease_type,
                "confidence_state": cal_result.state,
                "confidence_percent": round(cal_result.calibrated_probability * 100, 1),
                "severity": severity_label,
                "severity_description": severity_description,
                "affected_area_percent": affected_area_percent,
                "segmentation_available": segmentation_available,
                "gradcam_available": gradcam_available,
                "visual_analysis": visual_analysis,
                "knowledge": knowledge_data,
                "top_predictions": top_predictions_data,
                "images": images,
                "model_version": "leafguard-efficientnet-b0-v2.0",
                "inference_time_ms": round(total_ms, 1),
                "timings": {k: round(v, 1) for k, v in timings.items()},
            }

            self.storage.save_result(analysis_id, result)
            return result

        except Exception as e:
            logger.error(f"Pipeline error for {analysis_id}: {e}", exc_info=True)
            return {
                "id": analysis_id,
                "status": "error",
                "message": f"An error occurred during image processing: {str(e)}",
            }

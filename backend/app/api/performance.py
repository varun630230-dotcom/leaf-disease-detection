"""LeafGuard AI — Performance metrics endpoint."""

import json
import logging
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/performance")
async def get_performance():
    """Return verified model evaluation benchmarks."""
    eval_dir = settings.evaluation_dir

    if not eval_dir.exists():
        return {
            "status": "not_evaluated",
            "message": "Model has not been evaluated yet.",
        }

    result = {"status": "evaluated"}

    # Load metrics files
    files_map = {
        "overall": "evaluation_report.json",
        "per_class": "per_class_metrics.json",
        "ood": "ood_metrics.json",
        "segmentation": "segmentation_metrics.json",
        "latency": "latency_report.json",
        "model_comparison": "model_comparison.json",
    }

    for key, filename in files_map.items():
        filepath = eval_dir / filename
        if filepath.exists():
            try:
                result[key] = json.loads(filepath.read_text())
            except Exception as e:
                logger.warning(f"Failed to load {filename}: {e}")
                result[key] = None
        else:
            result[key] = None

    # Model info
    model_info_path = settings.classifier_dir / "model_info.json"
    if model_info_path.exists():
        try:
            result["model_info"] = json.loads(model_info_path.read_text())
        except Exception:
            result["model_info"] = None

    # Confusion matrix URL
    cm_path = eval_dir / "confusion_matrix.png"
    result["confusion_matrix_available"] = cm_path.exists()
    result["confusion_matrix_url"] = "/api/performance/confusion-matrix" if cm_path.exists() else None

    # Real known technical limitations
    result["limitations"] = [
        "Single-leaf focus: Trained primarily on isolated individual leaf surfaces; multi-plant dense field canopies may degrade localization accuracy.",
        "Severe chlorosis vs senescence: Late-stage natural autumn leaf dieback can occasionally exhibit visual overlap with fungal blight lesions.",
        "Novel non-agricultural species: Wild weeds or non-supported crops outside the 14 supported species are rejected by the OOD detector.",
        "Extreme lighting / heavy specular reflection: Strong camera flash reflections may mask faint foliar powdery mildew textures."
    ]

    return result


@router.get("/performance/confusion-matrix")
async def get_confusion_matrix():
    """Serve the test-set confusion matrix image."""
    cm_path = settings.evaluation_dir / "confusion_matrix.png"
    if not cm_path.exists():
        return {"error": "Confusion matrix not available."}
    return FileResponse(str(cm_path), media_type="image/png")

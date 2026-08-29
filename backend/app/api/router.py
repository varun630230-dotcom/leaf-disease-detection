"""LeafGuard AI — API Router aggregation."""

from fastapi import APIRouter

from app.api.analyze import router as analyze_router
from app.api.results import router as results_router
from app.api.performance import router as performance_router
from app.ml.classification.class_mapping import get_supported_plants as get_plants, NUM_CLASSES

api_router = APIRouter()

api_router.include_router(analyze_router, tags=["Analysis"])
api_router.include_router(results_router, tags=["Results"])
api_router.include_router(performance_router, tags=["Performance"])


@api_router.get("/supported-plants")
async def get_supported_plants():
    """Return list of plants the model supports."""
    return {
        "plants": get_plants(),
        "total_classes": NUM_CLASSES,
    }

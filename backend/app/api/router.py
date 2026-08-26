"""LeafGuard AI — API Router aggregation."""

from fastapi import APIRouter

from app.api.analyze import router as analyze_router
from app.api.results import router as results_router
from app.api.performance import router as performance_router

api_router = APIRouter()

api_router.include_router(analyze_router, tags=["Analysis"])
api_router.include_router(results_router, tags=["Results"])
api_router.include_router(performance_router, tags=["Performance"])


@api_router.get("/supported-plants")
async def get_supported_plants():
    """Return list of plants the model supports."""
    from app.ml.class_mapping import ClassMapping
    mapping = ClassMapping()
    return {
        "plants": mapping.get_supported_plants(),
        "total_classes": mapping.get_num_classes(),
    }

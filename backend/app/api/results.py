"""LeafGuard AI — Results and image serving endpoints."""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.services.storage import StorageService
from app.schemas.analysis import AnalysisResponse

logger = logging.getLogger(__name__)
router = APIRouter()

storage = StorageService()


@router.get("/analysis/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis(analysis_id: str):
    """Retrieve a saved analysis result by ID."""
    result = storage.get_result(analysis_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    return AnalysisResponse(**result)


@router.get("/images/{analysis_id}/{image_name}")
async def get_analysis_image(analysis_id: str, image_name: str):
    """Serve an analysis image (original, gradcam, mask, overlay)."""
    image_path = storage.get_image_path(analysis_id, image_name)
    if image_path is None:
        raise HTTPException(status_code=404, detail="Image not found.")

    path = Path(image_path)
    suffix = path.suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    media_type = media_types.get(suffix, "image/jpeg")

    return FileResponse(path=str(path), media_type=media_type)

"""LeafGuard AI — Analyze endpoint."""

import uuid
import logging

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.services.storage import StorageService
from app.services.pipeline import AnalysisPipeline
from app.schemas.analysis import AnalysisResponse

logger = logging.getLogger(__name__)
router = APIRouter()

storage = StorageService()
pipeline = AnalysisPipeline()


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze a plant leaf image.

    Accepts JPG, PNG, or WEBP images up to 25 MB.
    Returns disease classification, localization, severity, and visual analysis.
    """
    analysis_id = uuid.uuid4().hex[:12]

    try:
        content = await file.read()

        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")

        # Save uploaded file
        upload_path = storage.save_upload(
            content, file.filename or "upload.jpg", analysis_id
        )

        # Run the full inference pipeline
        result = await pipeline.run(
            image_path=str(upload_path),
            analysis_id=analysis_id,
        )

        # Save result JSON
        storage.save_result(analysis_id, result)

        return AnalysisResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed for {analysis_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Image could not be processed. Please try again.",
        )

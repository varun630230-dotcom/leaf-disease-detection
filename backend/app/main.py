"""LeafGuard AI — FastAPI Application."""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("leafguard")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models on startup, cleanup on shutdown."""
    logger.info("Starting LeafGuard AI...")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Device: {settings.device}")

    # Pre-load models so first request isn't slow
    try:
        from app.ml.model_manager import ModelManager
        manager = ModelManager()
        if manager.is_loaded():
            logger.info(f"Model loaded: {manager.get_version()}")
        else:
            logger.warning("Model weights not found — running in mock mode")
    except Exception as e:
        logger.error(f"Failed to initialize ML pipeline: {e}")

    yield
    logger.info("Shutting down LeafGuard AI.")


app = FastAPI(
    title="LeafGuard AI",
    description="Plant Disease Detection & Visual Analysis API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=600,
)


# Request timing middleware
@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = (time.time() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed:.1f}"
    return response


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "An internal error occurred. Please try again.",
        },
    )


# Include API routes
from app.api.router import api_router  # noqa: E402
app.include_router(api_router, prefix="/api")


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    from app.ml.model_manager import ModelManager
    manager = ModelManager()
    return {
        "status": "ok",
        "model_loaded": manager.is_loaded(),
        "model_version": manager.get_version(),
        "environment": settings.environment,
    }

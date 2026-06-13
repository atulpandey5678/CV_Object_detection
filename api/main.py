"""
FastAPI application for the Industrial Surface Defect Detection Platform.

Provides REST API endpoints for:
    - POST /predict: Run defect detection on uploaded images
    - GET /history: Retrieve inspection history
    - GET /analytics: Get aggregated analytics
    - DELETE /inspection/{id}: Remove an inspection record

Usage:
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.database.crud import (
    create_inspection,
    delete_inspection,
    get_analytics,
    get_inspections,
    get_total_count,
)
from api.database.session import get_db, init_db
from api.schemas.prediction import (
    AnalyticsResponse,
    InspectionListResponse,
    InspectionResponse,
    PredictionResponse,
)
from config.settings import API_HOST, API_PORT, MODELS_DIR

# Global inference pipeline instance
_pipeline = None


def get_pipeline():
    """Get or create the inference pipeline singleton."""
    global _pipeline
    if _pipeline is None:
        from src.inference.pipeline import InferencePipeline

        # Try to load best classifier
        classifier_path = MODELS_DIR / "best_classifier.pth"
        detector_path = MODELS_DIR / "best_detector.pt"

        cls_path = str(classifier_path) if classifier_path.exists() else None
        det_path = str(detector_path) if detector_path.exists() else None

        _pipeline = InferencePipeline(
            classifier_path=cls_path,
            detector_path=det_path,
        )
    return _pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    # Startup: Initialize database
    init_db()
    # Pre-load models
    get_pipeline()
    yield
    # Shutdown: cleanup if needed


# Create FastAPI app
app = FastAPI(
    title="Industrial Surface Defect Detection API",
    description=(
        "REST API for steel surface defect detection and quality control. "
        "Supports image upload, defect classification, bounding box detection, "
        "and inspection history management."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Upload directory
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Allowed image extensions
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Run defect detection on an uploaded image.

    Accepts an image file, runs inference, stores the result in the database,
    and returns the prediction with confidence scores and bounding boxes.
    """
    # Validate file type
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}",
        )

    # Save uploaded file
    file_id = str(uuid.uuid4())[:8]
    filename = f"{file_id}_{file.filename}"
    file_path = UPLOAD_DIR / filename

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    # Run inference
    try:
        import cv2
        import numpy as np

        # Read image from bytes
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")

        pipeline = get_pipeline()
        result = pipeline.predict_single(image)

        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    # Store in database
    try:
        inspection = create_inspection(
            db=db,
            image_path=str(file_path),
            image_filename=file.filename,
            defect_type=result["defect_type"],
            confidence=result["confidence"],
            bounding_boxes=result.get("bounding_boxes"),
            all_scores=result.get("all_scores"),
            inference_time_ms=result.get("inference_time_ms"),
        )
        inspection_id = inspection.id
    except Exception as e:
        inspection_id = None

    return PredictionResponse(
        defect_type=result["defect_type"],
        confidence=result["confidence"],
        bounding_boxes=result.get("bounding_boxes", []),
        all_scores=result.get("all_scores", {}),
        inference_time_ms=result.get("inference_time_ms", 0),
        image_filename=file.filename,
        inspection_id=inspection_id,
    )


@app.get("/history", response_model=InspectionListResponse)
async def get_history(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    defect_type: Optional[str] = Query(None, description="Filter by defect type"),
    min_confidence: Optional[float] = Query(None, ge=0, le=1),
    db: Session = Depends(get_db),
):
    """
    Get paginated inspection history with optional filters.
    """
    skip = (page - 1) * page_size
    inspections = get_inspections(
        db=db,
        skip=skip,
        limit=page_size,
        defect_type=defect_type,
        min_confidence=min_confidence,
    )
    total = get_total_count(db)

    return InspectionListResponse(
        inspections=[
            InspectionResponse(
                id=i.id,
                inspection_id=i.inspection_id,
                image_filename=i.image_filename,
                defect_type=i.defect_type,
                confidence=i.confidence,
                bounding_boxes=i.bounding_boxes,
                all_scores=i.all_scores,
                inference_time_ms=i.inference_time_ms,
                model_type=i.model_type,
                created_at=i.created_at,
            )
            for i in inspections
        ],
        total=total,
        page=page,
        page_size=page_size,
    )


@app.get("/analytics", response_model=AnalyticsResponse)
async def analytics(db: Session = Depends(get_db)):
    """Get aggregated analytics for the quality-control dashboard."""
    data = get_analytics(db)
    return AnalyticsResponse(**data)


@app.delete("/inspection/{inspection_id}")
async def remove_inspection(
    inspection_id: int,
    db: Session = Depends(get_db),
):
    """Delete an inspection record by ID."""
    success = delete_inspection(db, inspection_id)
    if not success:
        raise HTTPException(status_code=404, detail="Inspection not found")
    return {"message": "Inspection deleted successfully", "id": inspection_id}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "defect-detection-api"}

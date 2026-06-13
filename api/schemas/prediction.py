"""
Pydantic schemas for API request/response validation.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """Response model for POST /predict endpoint."""

    defect_type: str = Field(..., description="Predicted defect class name")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    bounding_boxes: list[list[float]] = Field(
        default=[], description="Bounding boxes [[x1,y1,x2,y2], ...]"
    )
    all_scores: dict[str, float] = Field(
        default={}, description="Scores for all classes"
    )
    inference_time_ms: float = Field(..., description="Inference time in ms")
    image_filename: str = Field(..., description="Uploaded filename")
    inspection_id: Optional[int] = Field(None, description="Database record ID")


class InspectionResponse(BaseModel):
    """Response model for inspection history records."""

    id: int
    inspection_id: str
    image_filename: str
    defect_type: str
    confidence: float
    bounding_boxes: Optional[list] = None
    all_scores: Optional[dict] = None
    inference_time_ms: Optional[float] = None
    model_type: str
    created_at: datetime

    class Config:
        from_attributes = True


class InspectionListResponse(BaseModel):
    """Paginated response for GET /history."""

    inspections: list[InspectionResponse]
    total: int
    page: int
    page_size: int


class AnalyticsResponse(BaseModel):
    """Response model for GET /analytics endpoint."""

    total_inspections: int
    defect_counts: dict[str, int]
    avg_confidence: float
    avg_inference_time_ms: float
    most_common_defect: str
    daily_counts: list[dict]


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str
    status_code: int = 400

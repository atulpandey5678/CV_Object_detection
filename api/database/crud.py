"""
CRUD operations for the inspection database.

Provides create, read, update, delete operations along with
analytics queries for the dashboard.
"""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from api.database.models import Inspection


def create_inspection(
    db: Session,
    image_path: str,
    image_filename: str,
    defect_type: str,
    confidence: float,
    bounding_boxes: Optional[list] = None,
    all_scores: Optional[dict] = None,
    inference_time_ms: Optional[float] = None,
    model_type: str = "classification",
    notes: Optional[str] = None,
) -> Inspection:
    """
    Create a new inspection record.

    Args:
        db: Database session.
        image_path: Path to the inspected image.
        image_filename: Original filename of the image.
        defect_type: Predicted defect class name.
        confidence: Prediction confidence score.
        bounding_boxes: Optional list of bounding boxes.
        all_scores: Optional dict of all class scores.
        inference_time_ms: Inference time in milliseconds.
        model_type: Type of model used ('classification' or 'detection').
        notes: Optional notes about the inspection.

    Returns:
        The created Inspection record.
    """
    inspection = Inspection(
        image_path=image_path,
        image_filename=image_filename,
        defect_type=defect_type,
        confidence=confidence,
        bounding_boxes=bounding_boxes,
        all_scores=all_scores,
        inference_time_ms=inference_time_ms,
        model_type=model_type,
        notes=notes,
    )
    db.add(inspection)
    db.commit()
    db.refresh(inspection)
    return inspection


def get_inspections(
    db: Session,
    skip: int = 0,
    limit: int = 50,
    defect_type: Optional[str] = None,
    min_confidence: Optional[float] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> list[Inspection]:
    """
    Get inspection records with filtering and pagination.

    Args:
        db: Database session.
        skip: Number of records to skip (offset).
        limit: Maximum number of records to return.
        defect_type: Filter by defect type.
        min_confidence: Filter by minimum confidence.
        start_date: Filter by start date.
        end_date: Filter by end date.

    Returns:
        List of Inspection records.
    """
    query = db.query(Inspection)

    if defect_type:
        query = query.filter(Inspection.defect_type == defect_type)
    if min_confidence is not None:
        query = query.filter(Inspection.confidence >= min_confidence)
    if start_date:
        query = query.filter(Inspection.created_at >= start_date)
    if end_date:
        query = query.filter(Inspection.created_at <= end_date)

    query = query.order_by(desc(Inspection.created_at))
    return query.offset(skip).limit(limit).all()


def get_inspection_by_id(db: Session, inspection_id: int) -> Optional[Inspection]:
    """Get a single inspection by its primary key."""
    return db.query(Inspection).filter(Inspection.id == inspection_id).first()


def delete_inspection(db: Session, inspection_id: int) -> bool:
    """
    Delete an inspection record.

    Args:
        db: Database session.
        inspection_id: Primary key of the inspection to delete.

    Returns:
        True if deleted successfully, False if not found.
    """
    inspection = db.query(Inspection).filter(Inspection.id == inspection_id).first()
    if inspection is None:
        return False
    db.delete(inspection)
    db.commit()
    return True


def get_analytics(db: Session) -> dict:
    """
    Get aggregated analytics for the dashboard.

    Returns:
        Dictionary with analytics data:
            - total_inspections
            - defect_counts: {defect_type: count}
            - avg_confidence
            - defect_rate
            - daily_counts: [{date, count}]
            - most_common_defect
    """
    total = db.query(func.count(Inspection.id)).scalar() or 0

    # Defect type distribution
    defect_counts_query = (
        db.query(Inspection.defect_type, func.count(Inspection.id))
        .group_by(Inspection.defect_type)
        .all()
    )
    defect_counts = {row[0]: row[1] for row in defect_counts_query}

    # Average confidence
    avg_confidence = db.query(func.avg(Inspection.confidence)).scalar() or 0.0

    # Daily inspection counts (last 30 days)
    daily_query = (
        db.query(
            func.date(Inspection.created_at).label("date"),
            func.count(Inspection.id).label("count"),
        )
        .group_by(func.date(Inspection.created_at))
        .order_by(func.date(Inspection.created_at))
        .limit(30)
        .all()
    )
    daily_counts = [{"date": str(row.date), "count": row.count} for row in daily_query]

    # Most common defect
    most_common = max(defect_counts, key=defect_counts.get) if defect_counts else "None"

    # Average inference time
    avg_inference_time = db.query(func.avg(Inspection.inference_time_ms)).scalar() or 0.0

    return {
        "total_inspections": total,
        "defect_counts": defect_counts,
        "avg_confidence": round(float(avg_confidence), 4),
        "avg_inference_time_ms": round(float(avg_inference_time), 2),
        "most_common_defect": most_common,
        "daily_counts": daily_counts,
    }


def get_total_count(db: Session) -> int:
    """Get total number of inspections."""
    return db.query(func.count(Inspection.id)).scalar() or 0

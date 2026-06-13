"""
SQLAlchemy ORM models for the inspection database.

Defines the schema for storing inspection records including
predictions, confidence scores, bounding boxes, and timestamps.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, JSON
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """SQLAlchemy declarative base class."""
    pass


class Inspection(Base):
    """
    Inspection record model.

    Stores the results of each defect detection inference run,
    including the image path, predicted defect type, confidence,
    bounding boxes, and timestamps.
    """

    __tablename__ = "inspections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    inspection_id = Column(
        String(36),
        unique=True,
        nullable=False,
        default=lambda: str(uuid.uuid4()),
    )
    image_path = Column(String(500), nullable=False)
    image_filename = Column(String(255), nullable=False)
    defect_type = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    bounding_boxes = Column(JSON, nullable=True)  # [[x1, y1, x2, y2], ...]
    all_scores = Column(JSON, nullable=True)  # {"class_name": score, ...}
    inference_time_ms = Column(Float, nullable=True)
    model_type = Column(String(50), default="classification")
    notes = Column(Text, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    def __repr__(self) -> str:
        return (
            f"<Inspection(id={self.id}, defect={self.defect_type}, "
            f"conf={self.confidence:.3f}, time={self.created_at})>"
        )

    def to_dict(self) -> dict:
        """Convert inspection record to dictionary."""
        return {
            "id": self.id,
            "inspection_id": self.inspection_id,
            "image_path": self.image_path,
            "image_filename": self.image_filename,
            "defect_type": self.defect_type,
            "confidence": self.confidence,
            "bounding_boxes": self.bounding_boxes,
            "all_scores": self.all_scores,
            "inference_time_ms": self.inference_time_ms,
            "model_type": self.model_type,
            "notes": self.notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

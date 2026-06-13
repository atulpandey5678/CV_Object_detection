"""
Database session management with connection pooling.

Provides async-compatible session factory and engine configuration
for PostgreSQL with fallback to SQLite for development.
"""

import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from api.database.models import Base
from config.settings import DATABASE_URL

# Use SQLite fallback for development if PostgreSQL is unavailable
_database_url = os.getenv("DATABASE_URL", DATABASE_URL)

# If PostgreSQL connection fails, fallback to SQLite
if "postgresql" in _database_url:
    try:
        engine = create_engine(
            _database_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
    except Exception:
        # Fallback to SQLite for local development
        sqlite_path = Path(__file__).resolve().parent.parent.parent / "data" / "inspections.db"
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        _database_url = f"sqlite:///{sqlite_path}"
        engine = create_engine(_database_url, connect_args={"check_same_thread": False})
else:
    engine = create_engine(_database_url, connect_args={"check_same_thread": False})

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Create all tables in the database."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Session:
    """
    Get a database session (dependency injection for FastAPI).

    Yields a session and ensures it is closed after use.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

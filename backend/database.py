"""
backend/database.py

SQLAlchemy engine, session factory, and base class.
Every backend route and pipeline script that needs DB access imports from here.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise EnvironmentError(
        "DATABASE_URL is not set in .env. "
        "Expected format: postgresql://user:password@host:port/dbname"
    )

# ─── Engine ──────────────────────────────────────────────────────────────────
# pool_pre_ping=True — test connections before use (handles Docker restarts)
# echo=False — set True temporarily if you want to see raw SQL for debugging
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    echo=False,
)

# ─── Session factory ─────────────────────────────────────────────────────────
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

# ─── Base class for ORM models ────────────────────────────────────────────────
Base = declarative_base()


def get_db():
    """
    FastAPI dependency — yields a database session and closes it after the request.

    Usage in a route:
        from backend.database import get_db
        from sqlalchemy.orm import Session
        from fastapi import Depends

        @router.get("/detections")
        def list_detections(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_session():
    """
    Direct session for pipeline scripts (not FastAPI).
    Must be closed manually:

        session = get_db_session()
        try:
            ...
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    """
    return SessionLocal()


def check_connection():
    """Quick health check — call at startup to confirm DB is reachable."""
    try:
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        logger.info("Database connection OK.")
        return True
    except Exception as e:
        logger.error(f"Database connection FAILED: {e}")
        return False
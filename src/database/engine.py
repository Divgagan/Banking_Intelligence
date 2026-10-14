"""
src/database/engine.py
SQLAlchemy engine and session factory.
All modules must import `get_engine` or `get_session` from here.
"""
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from loguru import logger

from src.config import settings


def get_engine(echo: bool = False):
    """Return a SQLAlchemy engine connected to PostgreSQL."""
    engine = create_engine(
        settings.database_url,
        pool_pre_ping=True,      # test connection before use
        pool_size=5,
        max_overflow=10,
        echo=echo,
    )
    return engine


def get_session(engine=None):
    """Return a SQLAlchemy session factory."""
    if engine is None:
        engine = get_engine()
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    return SessionLocal


def test_connection() -> bool:
    """Verify database is reachable. Returns True on success."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Database connection OK")
        return True
    except Exception as exc:
        logger.error(f"Database connection FAILED: {exc}")
        return False


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass

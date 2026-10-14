"""
scripts/create_db.py
────────────────────────────────────────────────────────────
Creates the PostgreSQL database and all tables from the ORM schema.
Run this ONCE before loading data.

Usage:
    python scripts/create_db.py

Prerequisites:
    1. PostgreSQL installed and running
    2. .env configured with DB_* variables
    3. The database `banking_intelligence` created manually (or use --create-db flag)
"""
import argparse
import sys
from pathlib import Path

# Make src importable from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from loguru import logger
from sqlalchemy import inspect, text

from src.config import settings
from src.database.engine import Base, get_engine

# Import all models so SQLAlchemy knows about them
import src.database.schema  # noqa: F401


def create_database_if_missing() -> None:
    """Connect to 'postgres' default DB and create our DB if it doesn't exist."""
    try:
        conn = psycopg2.connect(
            host=settings.db_host,
            port=settings.db_port,
            user=settings.db_user,
            password=settings.db_password,
            dbname="postgres",  # connect to default first
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{settings.db_name}'")
        exists = cur.fetchone()
        if not exists:
            cur.execute(f'CREATE DATABASE "{settings.db_name}"')
            logger.info(f"  Created database: {settings.db_name}")
        else:
            logger.info(f"  Database already exists: {settings.db_name}")
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"Could not create database: {e}")
        logger.info("  → Make sure PostgreSQL is running and credentials in .env are correct")
        raise


def create_all_tables(engine) -> None:
    """Create all tables defined in schema.py (skips existing tables)."""
    Base.metadata.create_all(engine, checkfirst=True)
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    logger.info(f"  Tables in database ({len(tables)}):")
    for t in sorted(tables):
        logger.info(f"    • {t}")


def verify_connection(engine) -> None:
    with engine.connect() as conn:
        result = conn.execute(text("SELECT version()"))
        ver = result.scalar()
        logger.info(f"  PostgreSQL: {ver}")


def main(create_db: bool = True) -> None:
    logger.info("=" * 60)
    logger.info("DATABASE SETUP")
    logger.info(f"  Host     : {settings.db_host}:{settings.db_port}")
    logger.info(f"  Database : {settings.db_name}")
    logger.info(f"  User     : {settings.db_user}")
    logger.info("=" * 60)

    if create_db:
        logger.info("\n── Step 1: Ensure database exists ──")
        create_database_if_missing()

    logger.info("\n── Step 2: Connect & verify ──")
    engine = get_engine()
    verify_connection(engine)

    logger.info("\n── Step 3: Create tables ──")
    create_all_tables(engine)

    logger.info("\n✓ Database setup complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set up the banking intelligence PostgreSQL database")
    parser.add_argument("--no-create-db", action="store_true",
                        help="Skip creating the database (use if DB already exists)")
    args = parser.parse_args()
    main(create_db=not args.no_create_db)

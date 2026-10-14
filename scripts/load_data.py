"""
scripts/load_data.py
────────────────────────────────────────────────────────────
Loads cleaned parquet files into PostgreSQL.
Run AFTER:
  1. scripts/create_db.py
  2. python -m src.data.cleaning (to generate parquet files)

Loading strategy:
  - Raw tables: append-only mirror of CSVs (loaded once)
  - Cleaned tables: truncate + reload (idempotent)
  - Uses chunked inserts (10,000 rows at a time) for the 3M+ tx table

Usage:
    python scripts/load_data.py [--raw-only] [--clean-only] [--skip-tx]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from loguru import logger
from sqlalchemy import text
from tqdm import tqdm

from src.config import settings
from src.database.engine import get_engine

CHUNK_SIZE = 10_000


def _load_csv_to_raw(engine, csv_path: Path, table_name: str,
                     parse_dates: list[str] | None = None) -> None:
    """Load a CSV directly into a raw mirror table (append, no transform)."""
    logger.info(f"  Loading {csv_path.name} → {table_name} ...")

    # Check if already loaded
    with engine.connect() as conn:
        count = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()
    if count > 0:
        logger.warning(f"  {table_name} already has {count:,} rows. Skipping raw load.")
        logger.warning("  To reload, truncate the table first: TRUNCATE TABLE {table_name};")
        return

    df = pd.read_csv(csv_path, low_memory=False, parse_dates=parse_dates)
    total = len(df)
    logger.info(f"  {total:,} rows to insert into {table_name}")

    for start in tqdm(range(0, total, CHUNK_SIZE), desc=f"Loading {table_name}", unit="chunk"):
        chunk = df.iloc[start: start + CHUNK_SIZE]
        chunk.to_sql(
            table_name, engine,
            if_exists="append",
            index=False,
            method="multi",
        )
    logger.info(f"  ✓ {table_name}: {total:,} rows loaded")


def _load_parquet_to_clean(engine, parquet_path: Path, table_name: str) -> None:
    """Load a cleaned parquet file into PostgreSQL. Creates or replaces the table."""
    if not parquet_path.exists():
        logger.error(f"  Parquet not found: {parquet_path}")
        logger.error("  Run: python -m src.data.cleaning first")
        return

    logger.info(f"  Loading {parquet_path.name} → {table_name} ...")

    # Check if table exists; if so truncate, otherwise we'll create it
    table_exists = False
    try:
        with engine.connect() as conn:
            conn.execute(text(f"SELECT 1 FROM {table_name} LIMIT 1"))
            table_exists = True
    except Exception:
        pass

    if table_exists:
        with engine.begin() as conn:
            conn.execute(text(f"TRUNCATE TABLE {table_name} CASCADE"))
        logger.info(f"  Truncated {table_name}")

    df = pd.read_parquet(parquet_path)
    # Convert datetime columns to Python date objects for SQLAlchemy compat
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.date

    total = len(df)
    mode = "append" if table_exists else "replace"

    for i, start in enumerate(tqdm(range(0, total, CHUNK_SIZE), desc=f"Loading {table_name}", unit="chunk")):
        chunk = df.iloc[start: start + CHUNK_SIZE]
        chunk.to_sql(
            table_name, engine,
            if_exists=mode if i == 0 and not table_exists else "append",
            index=False,
            method="multi",
        )
        if i == 0 and not table_exists:
            mode = "append"  # subsequent chunks always append

    logger.info(f"  ✓ {table_name}: {total:,} rows loaded")



def load_raw_tables(engine) -> None:
    logger.info("\n── Loading RAW tables (CSV mirrors) ──")
    _load_csv_to_raw(
        engine,
        settings.customer_csv,
        "customers_raw",
        parse_dates=["first_tx", "last_tx", "last_survey_date",
                     "last_transaction_date", "first_transaction_date"],
    )
    _load_csv_to_raw(
        engine,
        settings.transactions_csv,
        "transactions_raw",
        parse_dates=["date"],
    )


def load_clean_tables(engine, skip_tx: bool = False) -> None:
    logger.info("\n── Loading CLEANED tables (from parquet) ──")
    _load_parquet_to_clean(engine, settings.customer_clean_parquet, "customers")
    if not skip_tx:
        _load_parquet_to_clean(engine, settings.transactions_clean_parquet, "transactions")
    else:
        logger.info("  Skipping transactions (--skip-tx flag set)")


def verify_counts(engine) -> None:
    logger.info("\n── Row count verification ──")
    tables = [
        "customers_raw", "transactions_raw",
        "customers", "transactions",
    ]
    with engine.connect() as conn:
        for t in tables:
            try:
                n = conn.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar()
                logger.info(f"  {t:<30} {n:>12,} rows")
            except Exception as e:
                logger.warning(f"  {t}: {e}")


def main(raw_only: bool, clean_only: bool, skip_tx: bool) -> None:
    logger.info("=" * 60)
    logger.info("DATA LOADER")
    logger.info("=" * 60)
    engine = get_engine()

    if not clean_only:
        load_raw_tables(engine)

    if not raw_only:
        load_clean_tables(engine, skip_tx=skip_tx)

    verify_counts(engine)
    logger.info("\n✓ Data loading complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load data into PostgreSQL")
    parser.add_argument("--raw-only",   action="store_true", help="Load only raw CSV mirror tables")
    parser.add_argument("--clean-only", action="store_true", help="Load only cleaned parquet tables")
    parser.add_argument("--skip-tx",    action="store_true", help="Skip loading transactions (faster)")
    args = parser.parse_args()
    main(args.raw_only, args.clean_only, args.skip_tx)

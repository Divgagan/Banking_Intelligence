"""Run the reproducible local data and analytics pipeline."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.data.cleaning import (
    clean_customers,
    clean_transactions,
    generate_cleaning_report,
    save_cleaned,
)
from src.features.engineering import build_customer_features
from src.models.churn import run_exploratory_risk_indicator
from src.models.propensity import build_product_ownership_summary
from src.models.segmentation import run_segmentation
from src.recommendations.nba_engine import run_nba_engine


def run_pipeline() -> None:
    customers = clean_customers()
    transactions = clean_transactions(
        valid_customer_ids=set(customers["customer_id"].dropna())
    )
    save_cleaned(customers, settings.customer_clean_parquet)
    save_cleaned(transactions, settings.transactions_clean_parquet)

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    cleaning_report = generate_cleaning_report(customers, transactions)
    (reports_dir / "cleaning_report.json").write_text(
        json.dumps(cleaning_report, indent=2, default=str), encoding="utf-8"
    )

    features = build_customer_features(customers, transactions)
    segments = run_segmentation(features)
    run_exploratory_risk_indicator(features)
    build_product_ownership_summary(features)
    risk_scores = pd.read_parquet(settings.processed_data_dir / "engagement_indicators.parquet")
    run_nba_engine(features, segments, risk_scores)
    print(
        "Pipeline complete: "
        f"{len(customers):,} customers, {len(transactions):,} valid transactions, "
        f"{len(segments):,} segment assignments."
    )


if __name__ == "__main__":
    run_pipeline()

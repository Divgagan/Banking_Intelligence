"""Inspect the actual raw data and write privacy-conscious audit outputs."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings


def inspect_dataset(
    raw_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Create a structural data dictionary and dataset-quality report."""
    raw_dir = raw_dir or settings.raw_data_dir
    customer_path = raw_dir / "customer_data.csv"
    transaction_path = raw_dir / "transactions_data.csv"
    if not customer_path.is_file() or not transaction_path.is_file():
        missing = [str(path) for path in (customer_path, transaction_path) if not path.is_file()]
        raise FileNotFoundError(f"Required raw dataset file(s) not found: {', '.join(missing)}")

    customers = pd.read_csv(customer_path, low_memory=False)
    transactions = pd.read_csv(transaction_path, low_memory=False)
    if "customer_id" not in customers or "customer_id" not in transactions:
        raise ValueError("Both source files must contain customer_id for relationship checks")

    customer_ids = set(customers["customer_id"].dropna())
    transaction_ids = set(transactions["customer_id"].dropna())
    dictionary_rows = []
    for source_name, frame in (
        (customer_path.name, customers),
        (transaction_path.name, transactions),
    ):
        for column in frame.columns:
            meaning = {
                "customer_id": "Source identifier; used to join customer and transaction records.",
                "date": "Source transaction date; no timezone or event-window metadata is provided.",
                "amount": "Source transaction amount; currency is not specified in the available file.",
                "type": "Source transaction category; category definitions should be confirmed with the provider.",
            }.get(column, "Source-provided field; detailed definition and units are not documented in the workspace.")
            dictionary_rows.append({
                "source_file": source_name,
                "column": column,
                "dtype": str(frame[column].dtype),
                "rows": int(len(frame)),
                "missing_count": int(frame[column].isna().sum()),
                "distinct_non_null": int(frame[column].nunique(dropna=True)),
                "description": meaning,
            })

    tx_date = pd.to_datetime(transactions["date"], errors="coerce") if "date" in transactions else None
    amount = pd.to_numeric(transactions["amount"], errors="coerce") if "amount" in transactions else None
    amount_p999 = (
        float(amount.quantile(0.999))
        if amount is not None and amount.notna().any()
        else None
    )
    report = {
        "dataset_name": "COFINFAD (as described by project owner; provenance not independently verified)",
        "source_files": {
            "customers": {
                "file": customer_path.name,
                "rows": int(len(customers)),
                "columns": int(len(customers.columns)),
                "duplicate_customer_ids": int(customers["customer_id"].duplicated().sum()),
            },
            "transactions": {
                "file": transaction_path.name,
                "rows": int(len(transactions)),
                "columns": int(len(transactions.columns)),
                "exact_duplicate_rows": int(transactions.duplicated().sum()),
                "duplicate_customer_date_amount_type_rows": (
                    int(transactions.duplicated(subset=["customer_id", "date", "amount", "type"]).sum())
                    if {"customer_id", "date", "amount", "type"}.issubset(transactions.columns)
                    else None
                ),
                "invalid_dates": int(tx_date.isna().sum()) if tx_date is not None else None,
                "date_min": str(tx_date.min().date()) if tx_date is not None and tx_date.notna().any() else None,
                "date_max": str(tx_date.max().date()) if tx_date is not None and tx_date.notna().any() else None,
                "invalid_or_nonpositive_amounts": (
                    int((amount.isna() | (amount <= 0)).sum()) if amount is not None else None
                ),
                "amount_p99_9": amount_p999,
                "amount_outlier_count_p99_9": (
                    int((amount > amount_p999).sum())
                    if amount is not None and amount_p999 is not None else None
                ),
                "currency_field_present": any("currency" in name.lower() for name in transactions.columns),
            },
        },
        "relationships": {
            "unique_customer_ids": int(customers["customer_id"].nunique(dropna=True)),
            "transaction_customer_ids_matching_customers": int(len(transaction_ids & customer_ids)),
            "orphan_transaction_customer_ids": int(len(transaction_ids - customer_ids)),
            "customers_without_transactions": int(len(customer_ids - transaction_ids)),
        },
        "target_validity": {
            "verified_churn_outcome_available": False,
            "source_churn_probability_is_verified_target": False,
            "historical_product_uptake_or_campaign_response_available": False,
            "currency_confirmed": False,
        },
        "provenance_and_licence": {
            "source_url": "not documented in the workspace",
            "licence": "not documented in the workspace; redistribution rights must be verified",
            "handling": "Keep raw files private until provenance and licence terms are confirmed.",
        },
        "column_dictionary_rows": len(dictionary_rows),
    }

    reports_dir = output_dir or PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "data_quality_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    pd.DataFrame(dictionary_rows).to_csv(reports_dir / "data_dictionary.csv", index=False)
    return report


if __name__ == "__main__":
    print(json.dumps(inspect_dataset(), indent=2))

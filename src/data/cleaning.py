"""
src/data/cleaning.py
─────────────────────────────────────────────────────────────
Data cleaning pipeline for:
  - customer_data.csv  → customers_clean.parquet
  - transactions_data.csv → transactions_clean.parquet

Design rules:
  1. Raw files are NEVER modified.
  2. Every transformation is logged with before/after counts.
  3. Pre-computed columns are preserved but renamed with `source_`
     prefix so they are clearly distinguishable from our engineered features.
  4. churn_probability is kept as `source_churn_probability` — NOT treated
     as a verified churn outcome or model target.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from loguru import logger

from src.config import settings


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _log_shape(df: pd.DataFrame, label: str) -> None:
    logger.info(f"  [{label}] shape={df.shape}")


def _drop_duplicates(df: pd.DataFrame, subset: list[str], label: str) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=subset)
    after = len(df)
    if before != after:
        logger.warning(f"  [{label}] Dropped {before - after} duplicate rows")
    return df


# ─────────────────────────────────────────────
# CUSTOMER CLEANING
# ─────────────────────────────────────────────

def clean_customers(raw_path: Path | None = None) -> pd.DataFrame:
    """
    Load and clean customer_data.csv.

    Returns a cleaned DataFrame with:
      - Proper dtypes
      - Renamed pre-computed columns (source_ prefix)
      - Missing values handled
      - Derived helper columns: age_group, has_complaints, has_feature_requests
    """
    if raw_path is None:
        raw_path = settings.customer_csv

    logger.info(f"Loading customer data from: {raw_path}")
    df = pd.read_csv(raw_path, low_memory=False)
    _log_shape(df, "raw")

    # ── 1. Deduplicate ──────────────────────────────────────────────
    df = _drop_duplicates(df, subset=["customer_id"], label="customers")

    # ── 2. Fix types ────────────────────────────────────────────────
    date_cols = ["first_tx", "last_tx", "last_survey_date",
                 "last_transaction_date", "first_transaction_date"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Boolean columns (may be read as string 'True'/'False')
    bool_cols = [
        "savings_account", "credit_card", "personal_loan",
        "investment_account", "insurance_product",
        "bill_payment_user", "auto_savings_enabled",
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower().map(
                {"true": True, "false": False, "1": True, "0": False}
            ).fillna(False).astype(bool)

    # ── 3. Handle missing values ────────────────────────────────────
    # credit_utilization_ratio: missing because customer has no credit card
    # → impute with 0 for non-credit-card holders; median for cc holders
    if "credit_utilization_ratio" in df.columns:
        cc_median = df.loc[df["credit_card"] == True, "credit_utilization_ratio"].median()
        df["credit_utilization_ratio"] = df.apply(
            lambda r: 0.0 if (r["credit_card"] == False and pd.isna(r["credit_utilization_ratio"]))
            else (cc_median if pd.isna(r["credit_utilization_ratio"]) else r["credit_utilization_ratio"]),
            axis=1,
        )
        logger.info("  credit_utilization_ratio: imputed 0 for non-CC holders, median for CC holders")

    # complaint_topics: missing = no complaint
    df["has_complaints"] = df["complaint_topics"].notna()

    # feature_requests: missing = no requests
    df["has_feature_requests"] = df["feature_requests"].notna()

    # ── 4. Derived columns ──────────────────────────────────────────
    # age → age_group (age only has 6 values: 29,30,44,45,59,60 — treat as bands)
    age_map = {29: "Young (≤30)", 30: "Young (≤30)",
               44: "Middle (31–45)", 45: "Middle (31–45)",
               59: "Senior (46–60)", 60: "Senior (46–60)"}
    df["age_group"] = df["age"].map(age_map).fillna("Unknown")

    # ── 5. Rename pre-computed / source columns ─────────────────────
    rename_map = {
        "customer_segment":          "source_segment",
        "clv_segment":               "source_clv_segment",
        "churn_probability":         "source_churn_probability",
        "customer_lifetime_value":   "source_clv",
        # Remove exact duplicates (keep one canonical name)
        "average_transaction_value": None,   # duplicate of avg_tx_value
        "total_transaction_volume":  None,   # duplicate of total_tx_volume
        "avg_daily_transactions":    None,   # duplicate of transaction_frequency
        "first_transaction_date":    None,   # duplicate of first_tx
        "last_transaction_date":     None,   # duplicate of last_tx
    }
    cols_to_drop = [k for k, v in rename_map.items() if v is None and k in df.columns]
    df = df.drop(columns=cols_to_drop)

    actual_renames = {k: v for k, v in rename_map.items() if v is not None and k in df.columns}
    df = df.rename(columns=actual_renames)

    # ── 6. Rename tenure column for clarity ─────────────────────────
    if "customer_tenure" in df.columns:
        df = df.rename(columns={"customer_tenure": "customer_tenure_months"})

    _log_shape(df, "cleaned")
    logger.info(f"  Missing values per column (top 5):\n{df.isnull().sum().nlargest(5).to_string()}")

    return df


# ─────────────────────────────────────────────
# TRANSACTION CLEANING
# ─────────────────────────────────────────────

def clean_transactions(
    raw_path: Path | None = None,
    valid_customer_ids: set[int] | None = None,
) -> pd.DataFrame:
    """
    Load and clean transactions_data.csv.

    - Parses dates
    - Drops 102 exact duplicate rows
    - Validates customer_id references if valid_customer_ids is provided
    - Removes negative / zero amounts
    - Flags extreme outlier transactions (>99.9th percentile) for review
    """
    if raw_path is None:
        raw_path = settings.transactions_csv

    logger.info(f"Loading transactions from: {raw_path}")
    df = pd.read_csv(raw_path, low_memory=False, dtype={"customer_id": "int64", "amount": "int64"})
    _log_shape(df, "raw")

    # ── 1. Parse dates ───────────────────────────────────────────────
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    invalid_dates = df["date"].isna().sum()
    if invalid_dates:
        logger.warning(f"  Dropping {invalid_dates} rows with unparseable dates")
        df = df[df["date"].notna()]

    # ── 2. Deduplicate (exact row duplicates) ────────────────────────
    df = _drop_duplicates(df, subset=["customer_id", "date", "amount", "type"], label="transactions")

    # ── 3. Remove invalid amounts ────────────────────────────────────
    before = len(df)
    df = df[df["amount"] > 0]
    dropped = before - len(df)
    if dropped:
        logger.warning(f"  Dropped {dropped} rows with amount ≤ 0")

    # ── 4. Validate customer IDs ─────────────────────────────────────
    if valid_customer_ids is not None:
        before = len(df)
        df = df[df["customer_id"].isin(valid_customer_ids)]
        orphans = before - len(df)
        if orphans:
            logger.warning(f"  Dropped {orphans} transactions with no matching customer")
        else:
            logger.info("  All transaction customer_ids match customers table ✓")

    # ── 5. Flag extreme outliers (>99.9th percentile) ────────────────
    p999 = df["amount"].quantile(0.999)
    df["amount_outlier"] = df["amount"] > p999
    n_outliers = df["amount_outlier"].sum()
    logger.info(
        f"  Amount 99.9th percentile = {p999:,.0f} source units | "
        f"Flagged {n_outliers} outlier transactions (kept — not removed)"
    )

    # ── 6. Add useful derived columns ────────────────────────────────
    df["year"]        = df["date"].dt.year
    df["month"]       = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek   # 0=Monday, 6=Sunday
    df["is_weekend"]  = df["day_of_week"] >= 5

    _log_shape(df, "cleaned")
    logger.info(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")

    return df


# ─────────────────────────────────────────────
# SAVE & LOAD
# ─────────────────────────────────────────────

def save_cleaned(df: pd.DataFrame, path: Path) -> None:
    """Save cleaned DataFrame as parquet for fast reloading."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")
    logger.info(f"  Saved → {path}  ({path.stat().st_size / 1e6:.1f} MB)")


def load_clean_customers() -> pd.DataFrame:
    return pd.read_parquet(settings.customer_clean_parquet)


def load_clean_transactions() -> pd.DataFrame:
    return pd.read_parquet(settings.transactions_clean_parquet)


# ─────────────────────────────────────────────
# CLEANING REPORT
# ─────────────────────────────────────────────

def generate_cleaning_report(cust: pd.DataFrame, tx: pd.DataFrame) -> dict:
    """Return a JSON-serialisable cleaning summary."""
    report = {
        "customers": {
            "rows": int(len(cust)),
            "columns": int(len(cust.columns)),
            "missing_per_col": cust.isnull().sum()[cust.isnull().sum() > 0].to_dict(),
            "source_churn_probability_available": "source_churn_probability" in cust.columns,
            "age_group_dist": cust["age_group"].value_counts().to_dict(),
            "source_segment_dist": cust["source_segment"].value_counts().to_dict(),
        },
        "transactions": {
            "rows": int(len(tx)),
            "date_min": str(tx["date"].min().date()),
            "date_max": str(tx["date"].max().date()),
            "amount_min": int(tx["amount"].min()),
            "amount_max": int(tx["amount"].max()),
            "amount_mean": float(tx["amount"].mean()),
            "type_dist": tx["type"].value_counts().to_dict(),
            "outlier_count": int(tx["amount_outlier"].sum()),
        },
    }
    return report


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import json

    logger.info("=" * 60)
    logger.info("PHASE 2: DATA CLEANING PIPELINE")
    logger.info("=" * 60)

    # Clean customers
    logger.info("\n── Cleaning Customers ──")
    cust_df = clean_customers()
    save_cleaned(cust_df, settings.customer_clean_parquet)

    # Clean transactions (validate against cleaned customers)
    logger.info("\n── Cleaning Transactions ──")
    valid_ids = set(cust_df["customer_id"].unique())
    tx_df = clean_transactions(valid_customer_ids=valid_ids)
    save_cleaned(tx_df, settings.transactions_clean_parquet)

    # Report
    report = generate_cleaning_report(cust_df, tx_df)
    report_path = Path("reports/cleaning_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"\n── Cleaning Report saved → {report_path}")
    logger.info("\nCleaning Summary:")
    logger.info(f"  Customers  : {report['customers']['rows']:,} rows, {report['customers']['columns']} cols")
    logger.info(f"  Transactions: {report['transactions']['rows']:,} rows")
    logger.info(f"  Churn heuristic positive rate: {report['customers']['churn_heuristic_positive_rate']:.2%}")
    logger.info("\n✓ Phase 2 cleaning complete.")

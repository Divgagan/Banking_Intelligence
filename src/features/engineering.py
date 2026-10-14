"""
src/features/engineering.py
─────────────────────────────────────────────────────────────
Feature engineering pipeline: builds a customer-level analytical
feature table from cleaned transactions + customer attributes.

All features are computed from the RAW transaction history only
(not from pre-computed columns in customer_data.csv) to avoid
reusing stale/duplicate aggregates and to maintain data integrity.

Features created:
  ── RFM-style ──────────────────────────────────────
  recency_days          : days since last transaction (reference: 2023-12-31)
  frequency_total       : total number of transactions
  frequency_30d         : transactions in last 30 days
  frequency_90d         : transactions in last 90 days
  monetary_total        : sum of all transaction amounts (source units)
  monetary_avg          : mean transaction amount (source units)
  monetary_std          : std deviation of amounts (source units)
  monetary_median       : median amount (source units)

  ── Transaction type breakdown ──────────────────────
  pct_deposit           : fraction of deposits
  pct_withdrawal        : fraction of withdrawals
  pct_transfer          : fraction of transfers
  pct_payment           : fraction of payments
  deposit_total         : total deposited amount (source units)
  withdrawal_total      : total withdrawn amount (source units)

  ── Temporal patterns ───────────────────────────────
  active_months         : number of distinct calendar months with activity
  weekend_ratio         : fraction of transactions on Sat/Sun
  avg_tx_per_month      : frequency / active_months
  recency_bucket        : categorical: recent / moderate / dormant

  ── Volatility & consistency ─────────────────────────
  amount_cv             : coefficient of variation (std/mean)
  max_single_tx         : largest single transaction

  ── Engagement composite ────────────────────────────
  engagement_score      : normalised [0,1] composite of:
                          app_logins_frequency, feature_usage_diversity,
                          frequency_total, active_months
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import MinMaxScaler

from src.config import settings


REFERENCE_DATE = pd.Timestamp("2023-12-31")  # end of dataset period


# ─────────────────────────────────────────────
# TRANSACTION-LEVEL FEATURE COMPUTATION
# ─────────────────────────────────────────────

def compute_rfm_features(tx: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RFM-style and temporal features per customer from transactions.
    tx must have columns: customer_id, date (datetime), amount, type, is_weekend
    """
    logger.info("  Computing RFM and transaction features ...")

    # Ensure date is datetime
    tx = tx.copy()
    tx["date"] = pd.to_datetime(tx["date"])

    grp = tx.groupby("customer_id")

    # ── Recency ────────────────────────────────────────────────────
    last_tx_date = grp["date"].max()
    recency_days = (REFERENCE_DATE - last_tx_date).dt.days

    # ── Frequency ──────────────────────────────────────────────────
    frequency_total = grp.size().rename("frequency_total")

    cutoff_30d = REFERENCE_DATE - pd.Timedelta(days=30)
    cutoff_90d = REFERENCE_DATE - pd.Timedelta(days=90)

    freq_30d = tx[tx["date"] >= cutoff_30d].groupby("customer_id").size()
    freq_90d = tx[tx["date"] >= cutoff_90d].groupby("customer_id").size()

    # ── Monetary ───────────────────────────────────────────────────
    monetary_total  = grp["amount"].sum()
    monetary_avg    = grp["amount"].mean()
    monetary_std    = grp["amount"].std().fillna(0)
    monetary_median = grp["amount"].median()
    max_single_tx   = grp["amount"].max()

    # ── Type breakdown ─────────────────────────────────────────────
    type_counts = tx.groupby(["customer_id", "type"]).size().unstack(fill_value=0)
    for t in ["Deposit", "Withdrawal", "Transfer", "Payment"]:
        if t not in type_counts.columns:
            type_counts[t] = 0

    deposit_total    = tx[tx["type"] == "Deposit"].groupby("customer_id")["amount"].sum()
    withdrawal_total = tx[tx["type"] == "Withdrawal"].groupby("customer_id")["amount"].sum()

    pct_deposit    = type_counts["Deposit"]    / frequency_total
    pct_withdrawal = type_counts["Withdrawal"] / frequency_total
    pct_transfer   = type_counts["Transfer"]   / frequency_total
    pct_payment    = type_counts["Payment"]    / frequency_total

    # ── Temporal patterns ──────────────────────────────────────────
    active_months = tx.groupby("customer_id").apply(
        lambda g: g["date"].dt.to_period("M").nunique(), include_groups=False
    )

    weekend_ratio = grp["is_weekend"].mean()

    avg_tx_per_month = frequency_total / active_months.clip(lower=1)

    # ── Volatility ─────────────────────────────────────────────────
    amount_cv = (monetary_std / monetary_avg.clip(lower=1)).fillna(0)

    # ── Assemble ───────────────────────────────────────────────────
    feat = pd.DataFrame({
        "recency_days":     recency_days,
        "frequency_total":  frequency_total,
        "frequency_30d":    freq_30d,
        "frequency_90d":    freq_90d,
        "monetary_total":   monetary_total,
        "monetary_avg":     monetary_avg,
        "monetary_std":     monetary_std,
        "monetary_median":  monetary_median,
        "max_single_tx":    max_single_tx,
        "pct_deposit":      pct_deposit,
        "pct_withdrawal":   pct_withdrawal,
        "pct_transfer":     pct_transfer,
        "pct_payment":      pct_payment,
        "deposit_total":    deposit_total,
        "withdrawal_total": withdrawal_total,
        "active_months":    active_months,
        "weekend_ratio":    weekend_ratio,
        "avg_tx_per_month": avg_tx_per_month,
        "amount_cv":        amount_cv,
    }).fillna(0)

    feat.index.name = "customer_id"
    feat = feat.reset_index()

    # ── Recency bucket ──────────────────────────────────────────────
    feat["recency_bucket"] = pd.cut(
        feat["recency_days"],
        bins=[-1, 30, 90, 9999],
        labels=["recent", "moderate", "dormant"],
    )

    logger.info(f"  RFM features computed: {feat.shape}")
    return feat


# ─────────────────────────────────────────────
# ENGAGEMENT COMPOSITE
# ─────────────────────────────────────────────

def compute_engagement_score(
    cust: pd.DataFrame,
    rfm: pd.DataFrame,
) -> pd.Series:
    """
    Compute a normalised engagement score [0, 1] per customer.
    Components:
      - app_logins_frequency (from customer_data)
      - feature_usage_diversity (from customer_data)
      - frequency_total (from rfm)
      - active_months (from rfm)
    """
    logger.info("  Computing engagement composite score ...")
    merged = cust[["customer_id", "app_logins_frequency", "feature_usage_diversity"]].merge(
        rfm[["customer_id", "frequency_total", "active_months"]],
        on="customer_id",
        how="left",
    )
    components = merged[
        ["app_logins_frequency", "feature_usage_diversity", "frequency_total", "active_months"]
    ].copy().fillna(0)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(components)
    score = scaled.mean(axis=1)  # equal-weight mean of normalised components

    series = pd.Series(score, index=merged["customer_id"], name="engagement_score")
    return series


# ─────────────────────────────────────────────
# FULL FEATURE TABLE
# ─────────────────────────────────────────────

def build_customer_features(
    cust: pd.DataFrame | None = None,
    tx: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build the full customer_features table.
    Joins RFM features with selected customer attributes.
    Saves result to processed/customer_features.parquet.
    """
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING")
    logger.info("=" * 60)

    if cust is None:
        from src.data.cleaning import load_clean_customers
        cust = load_clean_customers()
        logger.info(f"  Loaded customers: {cust.shape}")

    if tx is None:
        from src.data.cleaning import load_clean_transactions
        tx = load_clean_transactions()
        logger.info(f"  Loaded transactions: {tx.shape}")

    # ── Step 1: RFM features from raw transactions ──────────────────
    rfm = compute_rfm_features(tx)

    # ── Step 2: Engagement score ────────────────────────────────────
    eng = compute_engagement_score(cust, rfm)
    rfm["engagement_score"] = rfm["customer_id"].map(eng)

    # ── Step 3: Merge with customer-level attributes ─────────────────
    # Only bring in ORIGINAL attributes, NOT the pre-computed aggregates
    cust_cols = [
        "customer_id",
        "age_group", "gender", "income_bracket", "occupation",
        "education_level", "marital_status", "household_size",
        "acquisition_channel",
        # Product ownership
        "savings_account", "credit_card", "personal_loan",
        "investment_account", "insurance_product",
        "active_products", "bill_payment_user", "auto_savings_enabled",
        # Engagement from customer data
        "app_logins_frequency", "feature_usage_diversity",
        "credit_utilization_ratio", "international_transactions",
        "failed_transactions",
        # Satisfaction
        "base_satisfaction", "tx_satisfaction", "product_satisfaction",
        "satisfaction_score", "nps_score",
        "support_tickets_count", "resolved_tickets_ratio",
        "app_store_rating", "feedback_sentiment",
        "has_complaints", "has_feature_requests",
        # Tenure
        "customer_tenure_months",
        "preferred_transaction_type",
        # Target / reference scores (source prefix = pre-computed, not targets)
        "source_churn_probability",
        "source_clv",
        "source_segment",
        "source_clv_segment",
    ]
    available_cols = [c for c in cust_cols if c in cust.columns]
    cust_subset = cust[available_cols]

    feat = rfm.merge(cust_subset, on="customer_id", how="left")

    # ── Step 4: Final derived columns ────────────────────────────────
    # net_cash_flow: deposits minus withdrawals
    feat["net_cash_flow"] = feat["deposit_total"].fillna(0) - feat["withdrawal_total"].fillna(0)

    # product_diversity_score: count of distinct product types held
    product_cols = ["savings_account", "credit_card", "personal_loan",
                    "investment_account", "insurance_product"]
    feat["product_diversity_score"] = feat[product_cols].sum(axis=1)

    logger.info(f"  Feature table shape: {feat.shape}")
    logger.info(f"  Missing values:\n{feat.isnull().sum()[feat.isnull().sum()>0].head(10).to_string()}")

    # ── Step 5: Save ─────────────────────────────────────────────────
    out_path = settings.customer_features_parquet
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(out_path, index=False, engine="pyarrow")
    logger.info(f"  Saved → {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")

    return feat


def load_customer_features() -> pd.DataFrame:
    return pd.read_parquet(settings.customer_features_parquet)


if __name__ == "__main__":
    features = build_customer_features()
    print("\nFeature Table Sample:")
    print(features[["customer_id", "recency_days", "frequency_total",
                     "monetary_total", "engagement_score",
                     "source_churn_probability", "source_segment"]].head(10).to_string(index=False))
    print(f"\nShape: {features.shape}")
    print("\nColumn list:")
    for c in features.columns:
        print(f"  {c}")

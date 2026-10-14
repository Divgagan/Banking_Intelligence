"""
src/analytics/kpis.py
─────────────────────────────────────────────────────────────
Business KPI computations for the dashboard.
All figures computed from cleaned parquet files.
No fabricated metrics — everything grounded in actual data.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger


def load_all_data() -> dict[str, pd.DataFrame]:
    """Load all processed parquet files into a dict."""
    files = {
        "customers":   "data/processed/customers_clean.parquet",
        "transactions":"data/processed/transactions_clean.parquet",
        "features":    "data/processed/customer_features.parquet",
        "segments":    "data/processed/customer_segments.parquet",
        "risk_indicator": "data/processed/engagement_indicators.parquet",
        "ownership":   "data/processed/product_ownership.parquet",
        "nba":         "data/processed/recommended_actions.parquet",
    }
    data = {}
    for key, path in files.items():
        p = Path(path)
        if p.exists():
            frame = pd.read_parquet(p)
            if key == "risk_indicator" and "risk_indicator_score" not in frame.columns:
                logger.warning("  Ignoring legacy churn artifact without exploratory-score metadata: {}", path)
                continue
            if key == "ownership" and (
                "score_type" not in frame.columns
                or not frame["score_type"].eq("historical_ownership_only").all()
            ):
                logger.warning("  Ignoring legacy product-propensity artifact: {}", path)
                continue
            if key == "segments" and (
                "model_version" not in frame.columns
                or not frame["model_version"].eq("kmeans_v2").all()
            ):
                logger.warning("  Ignoring legacy segment assignments; rerun the current segmentation pipeline")
                continue
            if key == "nba" and (
                "engine_version" not in frame.columns
                or not frame["engine_version"].eq("rule_v2").all()
            ):
                logger.warning("  Ignoring legacy next-best-action artifact: {}", path)
                continue
            data[key] = frame
            logger.info(f"  Loaded {key}: {frame.shape}")
        else:
            logger.warning(f"  Missing: {path}")
    return data


def executive_kpis(data: dict) -> dict:
    """Page A: Executive overview KPIs."""
    cust = data.get("customers")
    tx   = data.get("transactions")
    seg  = data.get("segments")
    risk = data.get("risk_indicator")

    kpis = {}

    if cust is not None:
        kpis["total_customers"] = int(len(cust))
        kpis["gender_dist"] = cust["gender"].value_counts().to_dict()
        kpis["income_dist"] = cust["income_bracket"].value_counts().to_dict()
        kpis["acquisition_dist"] = cust["acquisition_channel"].value_counts().to_dict()

        # Product ownership rates
        for p in ["savings_account","credit_card","personal_loan",
                   "investment_account","insurance_product"]:
            if p in cust.columns:
                kpis[f"{p}_pct"] = round(float(cust[p].mean() * 100), 1)

        kpis["avg_active_products"] = round(float(cust["active_products"].mean()), 2)
        kpis["avg_satisfaction"]    = round(float(cust["satisfaction_score"].mean()), 2)
        kpis["avg_nps"]             = round(float(cust["nps_score"].mean()), 1)

    if tx is not None:
        kpis["total_transactions"]   = int(len(tx))
        kpis["total_transaction_amount"] = int(tx["amount"].sum())
        kpis["average_transaction_amount"] = int(tx["amount"].mean())
        kpis["tx_type_dist"]         = tx["type"].value_counts().to_dict()
        kpis["tx_by_month"] = (
            tx.groupby("month")["amount"].agg(["count","sum"])
              .rename(columns={"count":"tx_count","sum":"amount"})
              .to_dict(orient="index")
        )

    if seg is not None:
        kpis["segment_dist"] = seg["segment_label"].value_counts().to_dict()

    if risk is not None and "risk_indicator_score" in risk:
        kpis["avg_risk_indicator_score"] = round(float(risk["risk_indicator_score"].mean()), 4)
        kpis["upper_indicator_band"] = int((risk["risk_indicator_band"] == "upper").sum())
        kpis["middle_indicator_band"] = int((risk["risk_indicator_band"] == "middle").sum())

    return kpis


def customer_360(customer_id: int, data: dict) -> dict:
    """Page B: Full 360 profile for a single customer."""
    result = {"customer_id": customer_id}

    cust = data.get("customers")
    if cust is not None:
        row = cust[cust["customer_id"] == customer_id]
        if not row.empty:
            result["profile"] = row.iloc[0].to_dict()

    feat = data.get("features")
    if feat is not None:
        row = feat[feat["customer_id"] == customer_id]
        if not row.empty:
            result["features"] = row.iloc[0].to_dict()

    seg = data.get("segments")
    if seg is not None:
        row = seg[seg["customer_id"] == customer_id]
        if not row.empty:
            result["segment"] = row.iloc[0].to_dict()

    risk = data.get("risk_indicator")
    if risk is not None:
        row = risk[risk["customer_id"] == customer_id]
        if not row.empty:
            result["risk_indicator"] = row.iloc[0].to_dict()

    ownership = data.get("ownership")
    if ownership is not None:
        row = ownership[ownership["customer_id"] == customer_id]
        if not row.empty:
            result["product_ownership"] = row.iloc[0].to_dict()

    nba = data.get("nba")
    if nba is not None:
        row = nba[nba["customer_id"] == customer_id]
        if not row.empty:
            result["nba"] = row.iloc[0].to_dict()

    tx = data.get("transactions")
    if tx is not None:
        cust_tx = tx[tx["customer_id"] == customer_id].copy()
        cust_tx["date"] = pd.to_datetime(cust_tx["date"])
        result["transaction_count"] = int(len(cust_tx))
        result["transaction_summary"] = (
            cust_tx.groupby("type")["amount"]
            .agg(["count","sum","mean"])
            .round(0)
            .to_dict(orient="index")
        )
        result["recent_transactions"] = (
            cust_tx.sort_values("date", ascending=False)
            .head(10)[["date","amount","type"]]
            .astype({"date": str})
            .to_dict(orient="records")
        )

    return result


def segment_summary(data: dict) -> pd.DataFrame:
    """Page C / D: Per-segment summary of key metrics."""
    feat = data.get("features")
    seg  = data.get("segments")
    risk = data.get("risk_indicator")

    if feat is None or seg is None:
        return pd.DataFrame()

    merged = feat.merge(seg[["customer_id","segment_label","cluster"]], on="customer_id", how="left")

    if risk is not None and "risk_indicator_score" in risk:
        merged = merged.merge(
            risk[["customer_id", "risk_indicator_score"]],
            on="customer_id",
            how="left",
        )

    agg_cols = {
        "customer_id":        "count",
        "recency_days":       "mean",
        "frequency_total":    "mean",
        "monetary_avg":       "mean",
        "monetary_total":     "mean",
        "engagement_score":   "mean",
        "satisfaction_score": "mean",
        "nps_score":          "mean",
        "active_products":    "mean",
    }
    if "risk_indicator_score" in merged.columns:
        agg_cols["risk_indicator_score"] = "mean"

    available_agg = {k: v for k, v in agg_cols.items() if k in merged.columns}

    summary = (
        merged.groupby("segment_label")
        .agg(available_agg)
        .round(3)
        .rename(columns={"customer_id": "customer_count"})
    )
    return summary


def engagement_indicator_analytics(data: dict) -> dict:
    """Summarize the exploratory relative engagement indicator."""
    risk = data.get("risk_indicator")
    seg   = data.get("segments")

    if risk is None or "risk_indicator_score" not in risk:
        return {}

    result = {
        "score_type": "exploratory_relative_indicator",
        "score_is_probability": False,
        "risk_tiers": {
            "upper": int((risk["risk_indicator_band"] == "upper").sum()),
            "middle": int((risk["risk_indicator_band"] == "middle").sum()),
            "lower": int((risk["risk_indicator_band"] == "lower").sum()),
        },
    }

    if seg is not None:
        merged = risk.merge(seg[["customer_id","segment_label"]], on="customer_id", how="left")
        result["churn_by_segment"] = (
            merged.groupby("segment_label")["risk_indicator_score"]
            .agg(["mean","count"])
            .round(4)
            .to_dict(orient="index")
        )

    return result


def nba_summary(data: dict) -> dict:
    """Page E: NBA action distribution."""
    nba = data.get("nba")
    if nba is None:
        return {}

    return {
        "action_distribution": nba["action_code"].value_counts().to_dict(),
        "segment_distribution": (
            nba.groupby(["segment_label","action_code"]).size()
               .reset_index(name="count")
               .to_dict(orient="records")
        ) if "segment_label" in nba.columns else [],
    }

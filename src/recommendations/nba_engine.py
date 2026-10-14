"""
src/recommendations/nba_engine.py
─────────────────────────────────────────────────────────────
Next-Best-Action (NBA) Engine — Module 4

Deterministic review rules use an exploratory relative engagement indicator,
transaction recency, satisfaction, and complaint signals. They do not infer
churn probabilities or product suitability.

Actions are review prompts, not automated contact or sales instructions.

IMPORTANT:
  - All recommendations require human review before action
  - Recommendations are informational only, not automated approvals
  - No loan/credit eligibility decisions are made here
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from src.config import settings

# ─────────────────────────────────────────────
# ACTION CODES (deterministic, documentable)
# ─────────────────────────────────────────────

ACTIONS = {
    "ENGAGEMENT_REVIEW":  "Review Customer Engagement — Exploratory indicator elevated",
    "REENGAGEMENT":       "Review Recent Activity — No recent transaction observed",
    "SERVICE_REVIEW":     "Service Quality Review — Complaints or low satisfaction",
    "MONITOR":            "Monitor — Insufficient evidence for a specific action",
}

REVIEW_INDICATOR_THRESHOLD = 2 / 3
DORMANT_DAYS = 60
SAT_LOW = 3


# ─────────────────────────────────────────────
# RULE ENGINE
# ─────────────────────────────────────────────

def _generate_actions(row: pd.Series) -> list[dict]:
    """
    Generate a ranked list of candidate actions for a single customer.
    Returns list of dicts sorted by priority (highest first).
    """
    candidates = []

    risk_score = row.get("risk_indicator_score", None)
    recency     = row.get("recency_days", None)
    sat_score   = row.get("satisfaction_score", None)
    has_complaint = row.get("has_complaints", False)

    if risk_score is not None and pd.notna(risk_score) and risk_score >= REVIEW_INDICATOR_THRESHOLD:
        candidates.append({
            "action_code": "ENGAGEMENT_REVIEW",
            "priority": 80,
            "evidence": [
                f"exploratory_risk_indicator={risk_score:.2f}",
                "This is a relative indicator, not a churn probability.",
            ],
            "reason": "Observed engagement signals warrant human review; no churn outcome is predicted.",
        })

    if recency is not None and recency >= DORMANT_DAYS:
        candidates.append({
            "action_code": "REENGAGEMENT",
            "priority": 70,
            "evidence": [f"recency_days={recency:.0f} >= {DORMANT_DAYS}"],
            "reason": f"No transaction for {recency:.0f} days. Review whether support is needed before contacting.",
        })

    if (sat_score is not None and sat_score <= SAT_LOW) or has_complaint:
        evidence = []
        if sat_score is not None and sat_score <= SAT_LOW:
            evidence.append(f"satisfaction_score={sat_score} <= {SAT_LOW}")
        if has_complaint:
            evidence.append("has_open_complaints=True")
        candidates.append({
            "action_code": "SERVICE_REVIEW",
            "priority":    60,
            "evidence":    evidence,
            "reason": "Low satisfaction or recorded complaint signal. A service review is appropriate.",
        })

    if not candidates:
        candidates.append({
            "action_code": "MONITOR",
            "priority":    0,
            "evidence":    ["No threshold exceeded"],
            "reason":      "No specific action warranted at this time. Continue monitoring.",
        })

    # Sort by priority descending
    candidates.sort(key=lambda x: x["priority"], reverse=True)
    return candidates


# ─────────────────────────────────────────────
# MAIN NBA COMPUTATION
# ─────────────────────────────────────────────

def run_nba_engine(
    feat: pd.DataFrame | None = None,
    segments: pd.DataFrame | None = None,
    risk_scores: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Run the NBA engine for all customers.
    Returns DataFrame with one recommended action per customer.
    """
    logger.info("=" * 60)
    logger.info("MODULE 4: NEXT-BEST-ACTION ENGINE")
    logger.info("=" * 60)

    # ── Load inputs ───────────────────────────────────────────────────
    if feat is None:
        from src.features.engineering import load_customer_features
        feat = load_customer_features()

    if segments is None and Path("data/processed/customer_segments.parquet").exists():
        segments = pd.read_parquet("data/processed/customer_segments.parquet")

    if risk_scores is None and Path("data/processed/engagement_indicators.parquet").exists():
        saved_scores = pd.read_parquet("data/processed/engagement_indicators.parquet")
        if "risk_indicator_score" in saved_scores:
            risk_scores = saved_scores

    # ── Merge all signals ──────────────────────────────────────────────
    df = feat[["customer_id", "recency_days", "satisfaction_score",
               "has_complaints", "engagement_score",
               "credit_card", "personal_loan", "investment_account",
               "insurance_product"]].copy()

    if segments is not None:
        df = df.merge(
            segments[["customer_id", "segment_label", "cluster"]],
            on="customer_id", how="left"
        )

    if risk_scores is not None:
        df = df.merge(
            risk_scores[["customer_id", "risk_indicator_score", "risk_indicator_band"]],
            on="customer_id", how="left"
        )

    logger.info(f"  Unified signal table: {df.shape}")

    # ── Generate NBA for each customer ────────────────────────────────
    records = []
    for _, row in df.iterrows():
        cid = row["customer_id"]
        candidates = _generate_actions(row)
        top_action = candidates[0]
        all_actions = candidates[:3]  # store top 3 for dashboard

        records.append({
            "customer_id":         cid,
            "segment_label":       row.get("segment_label", "Unknown"),
            "risk_indicator_score": row.get("risk_indicator_score", None),
            "recency_days":        row.get("recency_days", None),
            "action_code":         top_action["action_code"],
            "action_description":  ACTIONS.get(top_action["action_code"], ""),
            "reason":              top_action["reason"],
            "evidence":            json.dumps(top_action["evidence"]),
            "priority_score":      top_action["priority"],
            "requires_human_review": True,
            "engine_version":      "rule_v2",
            "recommended_at":      datetime.now().isoformat(),
            "all_actions_json":    json.dumps([
                {"action_code": a["action_code"], "reason": a["reason"]}
                for a in all_actions
            ]),
        })

    nba_df = pd.DataFrame(records)

    # ── Summary ───────────────────────────────────────────────────────
    logger.info("\nNBA Action Distribution:")
    dist = nba_df["action_code"].value_counts()
    for action, count in dist.items():
        logger.info(f"  {action:<30} {count:>8,} customers ({count/len(nba_df):.1%})")

    # ── Save ──────────────────────────────────────────────────────────
    out_path = Path("data/processed/recommended_actions.parquet")
    nba_df.to_parquet(out_path, index=False)
    logger.info(f"\nNBA recommendations saved -> {out_path}")

    logger.info("\nNBA engine complete.")
    return nba_df


def get_customer_nba(customer_id: int, nba_df: pd.DataFrame | None = None) -> dict:
    """Return the NBA recommendation for a single customer."""
    if nba_df is None:
        nba_df = pd.read_parquet("data/processed/recommended_actions.parquet")
    if "engine_version" not in nba_df or not nba_df["engine_version"].eq("rule_v2").all():
        return {"error": "Next-best-action data is stale; run the current pipeline first"}

    row = nba_df[nba_df["customer_id"] == customer_id]
    if row.empty:
        return {"error": f"Customer {customer_id} not found"}

    row = row.iloc[0]
    return {
        "customer_id":         int(row["customer_id"]),
        "segment":             row.get("segment_label", "Unknown"),
        "risk_indicator_score": (
            round(float(row.get("risk_indicator_score")), 3)
            if pd.notna(row.get("risk_indicator_score")) else None
        ),
        "risk_score_is_probability": False,
        "recommended_action":  row["action_description"],
        "reason":              row["reason"],
        "evidence":            json.loads(row["evidence"]) if row["evidence"] else [],
        "priority_score":      int(row["priority_score"]),
        "requires_human_review": True,
        "limitations":         [
            "No validated churn probability is available.",
            "No future product-uptake labels or product eligibility data are available.",
            "No verified business outcome tracking available",
        ],
        "engine_version":      row["engine_version"],
        "recommended_at":      row["recommended_at"],
    }


if __name__ == "__main__":
    nba = run_nba_engine()
    print("\nAction distribution:")
    print(nba["action_code"].value_counts().to_string())

    # Demo: show a sample recommendation
    sample = get_customer_nba(int(nba["customer_id"].iloc[0]), nba)
    print("\nSample recommendation:")
    print(json.dumps(sample, indent=2))

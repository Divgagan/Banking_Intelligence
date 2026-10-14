import json

import pandas as pd
import pytest

from src.config import settings
from src.llm.assistant import _run_tool, answer_question
from src.models.churn import build_exploratory_risk_scores, run_exploratory_risk_indicator
from src.models.propensity import build_product_ownership_summary
from src.recommendations.nba_engine import _generate_actions


def _engagement_features() -> pd.DataFrame:
    return pd.DataFrame({
        "customer_id": [101, 102, 103],
        "recency_days": [2, 30, 90],
        "frequency_total": [100, 10, 1],
        "engagement_score": [0.9, 0.5, 0.1],
        "satisfaction_score": [5, 4, 2],
        "has_complaints": [False, False, True],
    })


def test_relative_indicator_orders_observed_signals_without_churn_label():
    scores = build_exploratory_risk_scores(_engagement_features())

    assert scores["risk_indicator_score"].is_monotonic_increasing
    assert scores["score_type"].eq("exploratory_relative_indicator").all()
    assert scores["risk_indicator_score"].between(0, 1).all()
    assert "churn_label" not in scores


def test_churn_pipeline_writes_no_supervised_metrics(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings, "processed_data_dir", tmp_path / "processed")

    report = run_exploratory_risk_indicator(_engagement_features())
    saved = json.loads((tmp_path / "reports" / "churn_report.json").read_text())
    scores = pd.read_parquet(tmp_path / "processed" / "engagement_indicators.parquet")

    assert report["supervised_model_trained"] is False
    assert report["score_is_probability"] is False
    assert "models" not in saved
    assert len(scores) == 3


def test_product_report_does_not_score_future_uptake(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings, "processed_data_dir", tmp_path / "processed")
    features = _engagement_features().assign(
        credit_card=[True, False, False],
        personal_loan=[False, False, True],
    )

    summary = build_product_ownership_summary(features)
    report = json.loads((tmp_path / "reports" / "propensity_report.json").read_text())

    assert report["future_uptake_target_available"] is False
    assert "credit_card_relevance_score" not in summary
    assert summary["score_type"].eq("historical_ownership_only").all()


def test_next_best_action_never_creates_product_offer_from_score():
    row = pd.Series({
        "risk_indicator_score": 0.9,
        "recency_days": 120,
        "satisfaction_score": 2,
        "has_complaints": True,
        "credit_card": False,
    })

    actions = _generate_actions(row)

    assert actions[0]["action_code"] == "ENGAGEMENT_REVIEW"
    assert all(not action["action_code"].startswith("OFFER_") for action in actions)


def test_customer_specific_llm_request_is_blocked_before_provider_call():
    result = answer_question(
        "Show customer profile 123456",
        data={},
        api_key="not-used",
        model_name="not-used",
    )

    assert result["mode"] == "privacy_blocked"
    assert result["data"] is None


def test_llm_tools_reject_arbitrary_sql_and_arguments():
    with pytest.raises(ValueError, match="unapproved"):
        _run_tool("run_sql", '{"query":"SELECT * FROM customers"}', {})
    with pytest.raises(ValueError):
        _run_tool("get_business_kpis", '{"query":"SELECT 1"}', {})

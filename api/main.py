"""
api/main.py
─────────────────────────────────────────────────────────────
FastAPI Backend — Banking Customer Intelligence Platform

Endpoints:
  GET  /health                    → Service health check
  GET  /api/v1/customers/{id}     → Customer 360 profile
  GET  /api/v1/customers/{id}/nba → NBA recommendation
  GET  /api/v1/kpis/executive     → Executive KPIs
  GET  /api/v1/segments           → Segment summary
  GET  /api/v1/churn/top          → Top N high-risk customers
  POST /api/v1/chat               → LLM analytics assistant

Run with:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

app = FastAPI(
    title="Banking Customer Intelligence API",
    description="AI-Powered customer analytics, churn prediction, and NBA engine",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# DATA LOADING (singleton pattern)
# ─────────────────────────────────────────────
_data_cache: dict | None = None


def get_data():
    global _data_cache
    if _data_cache is None:
        from src.analytics.kpis import load_all_data
        _data_cache = load_all_data()
    return _data_cache


# ─────────────────────────────────────────────
# HEALTH
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "healthy", "version": "1.0.0"}


# ─────────────────────────────────────────────
# CUSTOMER ENDPOINTS
# ─────────────────────────────────────────────
@app.get("/api/v1/customers/{customer_id}")
def get_customer(customer_id: int):
    """Full 360-degree customer profile."""
    from src.analytics.kpis import customer_360
    data = get_data()
    result = customer_360(customer_id, data)
    if not result.get("profile") and not result.get("features"):
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")

    # Convert numpy/pandas types to JSON-safe
    return _serialize(result)


@app.get("/api/v1/customers/{customer_id}/nba")
def get_customer_nba(customer_id: int):
    """Next-Best-Action recommendation for a specific customer."""
    from src.recommendations.nba_engine import get_customer_nba as _get_nba
    result = _get_nba(customer_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/api/v1/customers/{customer_id}/segment")
def get_customer_segment(customer_id: int):
    """Return a customer's current descriptive cluster assignment."""
    data = get_data()
    segments = data.get("segments")
    if segments is None:
        raise HTTPException(status_code=404, detail="Segment assignments are not available")
    row = segments[segments["customer_id"] == customer_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
    return _serialize(row.iloc[0].to_dict())


@app.get("/api/v1/customers/{customer_id}/risk")
def get_customer_risk_indicator(customer_id: int):
    """Return a transparent relative engagement indicator, never a churn probability."""
    scores = get_data().get("risk_indicator")
    if scores is None:
        raise HTTPException(status_code=404, detail="Exploratory indicators are not available")
    row = scores[scores["customer_id"] == customer_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
    return {
        **_serialize(row.iloc[0].to_dict()),
        "score_is_probability": False,
    }


@app.get("/api/v1/customers/{customer_id}/ownership")
@app.get("/api/v1/customers/{customer_id}/propensity", include_in_schema=False)
def get_customer_product_ownership(customer_id: int):
    """Return descriptive ownership facts without estimating future uptake."""
    features = get_data().get("features")
    if features is None:
        raise HTTPException(status_code=404, detail="Customer feature data is not available")
    row = features[features["customer_id"] == customer_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
    product_columns = [
        name for name in (
            "savings_account", "credit_card", "personal_loan",
            "investment_account", "insurance_product",
        ) if name in row.columns
    ]
    record = row.iloc[0]
    return {
        "customer_id": customer_id,
        "current_ownership": {name: bool(record[name]) for name in product_columns},
        "future_uptake_prediction_available": False,
        "limitation": "No future uptake labels or product eligibility data are available.",
    }


# ─────────────────────────────────────────────
# KPI ENDPOINTS
# ─────────────────────────────────────────────
@app.get("/api/v1/kpis/executive")
def executive_kpis():
    """Executive dashboard KPIs."""
    from src.analytics.kpis import executive_kpis as _kpis
    data = get_data()
    return _serialize(_kpis(data))


@app.get("/api/v1/segments")
def get_segments():
    """Segment profiling summary."""
    from src.analytics.kpis import segment_summary
    data = get_data()
    result = segment_summary(data)
    if result.empty:
        return {"segments": []}
    return {"segments": json.loads(result.to_json(orient="index"))}


# ─────────────────────────────────────────────
# CHURN ENDPOINTS
# ─────────────────────────────────────────────
@app.get("/api/v1/churn/top")
@app.get("/api/v1/risk/indicators/top")
def top_churn_customers(n: int = Query(20, ge=1, le=500)):
    """Top N customers by exploratory relative engagement indicator."""
    data = get_data()
    risk = data.get("risk_indicator")
    feat  = data.get("features")
    if risk is None:
        raise HTTPException(status_code=404, detail="Exploratory indicators are not available")

    merged = risk.merge(
        feat[["customer_id", "frequency_total", "monetary_total", "satisfaction_score"]],
        on="customer_id", how="left"
    ) if feat is not None else risk

    top = merged.nlargest(n, "risk_indicator_score")
    return {
        "score_type": "exploratory_relative_indicator",
        "score_is_probability": False,
        "disclaimer": "This is not a predicted churn probability or verified churn outcome.",
        "count": len(top),
        "customers": json.loads(top.to_json(orient="records")),
    }


# ─────────────────────────────────────────────
# CHAT (LLM ANALYTICS ASSISTANT)
# ─────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        message = value.strip()
        if not message:
            raise ValueError("message must not be blank")
        return message


class ChatResponse(BaseModel):
    reply: str
    data: dict | None = None
    mode: str
    disclaimer: str = (
        "Answers use approved read-only analytics tools. Exploratory indicators are not probabilities; "
        "recommendations require human review."
    )


@app.post("/api/v1/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Use validated OpenAI tool calls; never accept arbitrary SQL or tool names."""
    from src.config import settings
    from src.llm.assistant import AssistantProviderError, answer_question

    try:
        result = answer_question(
            req.message,
            get_data(),
            settings.openai_api_key,
            settings.openai_model,
        )
    except AssistantProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except (ValueError, LookupError) as exc:
        raise HTTPException(status_code=422, detail="The assistant requested invalid or unavailable analytics.") from exc
    return ChatResponse(**result)


# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────
import numpy as np

def _serialize(obj):
    """Recursively convert numpy/pandas types to JSON-safe Python types."""
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_serialize(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, "isoformat"):
        return obj.isoformat()
    elif obj is np.nan or (isinstance(obj, float) and obj != obj):
        return None
    return obj

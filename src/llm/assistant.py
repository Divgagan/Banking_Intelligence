"""Grounded, aggregate-only read-only analytics tools and OpenAI tool calling."""
from __future__ import annotations

import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, ConfigDict, Field

from src.analytics.kpis import executive_kpis, segment_summary


class _ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SegmentArguments(_ToolArguments):
    segment_label: str | None = None


class ModelArguments(_ToolArguments):
    model_name: str = Field(pattern="^(churn|segmentation|propensity)$")


class AssistantProviderError(RuntimeError):
    """Raised when the configured language-model provider is unavailable."""


def _data_period(data: dict[str, pd.DataFrame]) -> dict[str, str] | None:
    transactions = data.get("transactions")
    if transactions is None or transactions.empty or "date" not in transactions:
        return None
    dates = pd.to_datetime(transactions["date"], errors="coerce").dropna()
    if dates.empty:
        return None
    return {"start": dates.min().date().isoformat(), "end": dates.max().date().isoformat()}


def _json_default(value: Any) -> Any:
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return value.isoformat()
    raise TypeError(f"Unsupported value for JSON output: {type(value).__name__}")


def _segment_tool(args: SegmentArguments, data: dict[str, pd.DataFrame]) -> dict[str, Any]:
    summary = segment_summary(data)
    if args.segment_label is not None:
        if args.segment_label not in summary.index:
            raise LookupError(f"Segment {args.segment_label!r} was not found")
        summary = summary.loc[[args.segment_label]]
    return {
        "segments": json.loads(summary.reset_index().to_json(orient="records")),
        "data_period": _data_period(data),
    }


def _kpi_tool(data: dict[str, pd.DataFrame]) -> dict[str, Any]:
    kpis = executive_kpis(data)
    for key in ("gender_dist", "income_dist", "acquisition_dist"):
        counts = kpis.get(key)
        if isinstance(counts, dict):
            kpis[key] = {
                label: count for label, count in counts.items()
                if isinstance(count, (int, float)) and count >= 10
            }
    return {"kpis": kpis, "data_period": _data_period(data)}


def _product_tool(data: dict[str, pd.DataFrame]) -> dict[str, Any]:
    features = data.get("features")
    if features is None:
        return {"status": "unavailable", "reason": "Customer feature data is unavailable."}
    products = (
        "savings_account", "credit_card", "personal_loan",
        "investment_account", "insurance_product",
    )
    summaries = {}
    for name in products:
        if name in features:
            owned = features[name].fillna(False).astype(bool)
            summaries[name] = {
                "owners": int(owned.sum()),
                "ownership_rate": float(owned.mean()),
            }
    return {
        "status": "current_ownership_summary",
        "products": summaries,
        "future_uptake_prediction_available": False,
        "data_period": _data_period(data),
    }


def _evaluation_tool(args: ModelArguments) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    path = project_root / "reports" / f"{args.model_name}_report.json"
    if not path.exists():
        return {"model_name": args.model_name, "status": "unavailable"}
    return {
        "model_name": args.model_name,
        "report": json.loads(path.read_text(encoding="utf-8")),
    }


_ARGUMENT_MODELS: dict[str, type[BaseModel]] = {
    "get_segment_summary": SegmentArguments,
    "get_business_kpis": _ToolArguments,
    "get_product_ownership_summary": _ToolArguments,
    "get_model_evaluation": ModelArguments,
}

_TOOL_DESCRIPTIONS = {
    "get_segment_summary": "Get aggregate customer segment summaries, optionally for one label.",
    "get_business_kpis": "Get measured aggregate customer and transaction KPIs.",
    "get_product_ownership_summary": "Get aggregate current product ownership rates, not future propensity.",
    "get_model_evaluation": "Get the aggregate status/evaluation report for an approved model name.",
}


def _tool_specs() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": _TOOL_DESCRIPTIONS[name],
                "parameters": model.model_json_schema(),
            },
        }
        for name, model in _ARGUMENT_MODELS.items()
    ]


def _run_tool(name: str, raw_arguments: str, data: dict[str, pd.DataFrame]) -> dict[str, Any]:
    if name == "get_business_kpis":
        _ToolArguments.model_validate_json(raw_arguments)
        return _kpi_tool(data)
    if name == "get_product_ownership_summary":
        _ToolArguments.model_validate_json(raw_arguments)
        return _product_tool(data)
    if name == "get_segment_summary":
        return _segment_tool(SegmentArguments.model_validate_json(raw_arguments), data)
    if name == "get_model_evaluation":
        return _evaluation_tool(ModelArguments.model_validate_json(raw_arguments))
    raise ValueError("The model requested an unapproved analytics tool")


def answer_question(
    message: str,
    data: dict[str, pd.DataFrame],
    api_key: str,
    model_name: str,
) -> dict[str, Any]:
    """Answer aggregate analytics questions without sending individual records to the LLM."""
    sensitive_input = re.search(
        r"(?:\b\d{6,}\b|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|\b\d{3}[- .]\d{3}[- .]\d{4}\b)",
        message,
    )
    customer_level_request = re.search(
        r"\b(?:customer\s*(?:(?:id|number|profile)\b\s*)?(?:[:#]?\s*)\d+|"
        r"customer\s*(?:profile|id|number)|this customer|specific customer|"
        r"which customers|who are the customers|next.best.action)\b",
        message,
        re.IGNORECASE,
    )
    if sensitive_input or customer_level_request:
        return {
            "reply": (
                "Customer-level details and identifiers are never forwarded to the external LLM. "
                "Use the local customer API for individual records and actions."
            ),
            "data": None,
            "mode": "privacy_blocked",
        }
    if not api_key:
        return {
            "reply": (
                "The LLM assistant is not configured. Set OPENAI_API_KEY to enable aggregate "
                "natural-language tool calling; local read-only analytics endpoints remain available."
            ),
            "data": None,
            "mode": "not_configured",
        }

    client = OpenAI(api_key=api_key)
    tools = _tool_specs()
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a banking analytics assistant. Use only approved aggregate, read-only "
                "tools for dataset facts, numbers, segment summaries, ownership, and model status. "
                "Never request or disclose customer-level records, invent missing values, run SQL, "
                "recommend an offer from ownership alone, or describe exploratory indicators as "
                "churn probabilities. Explain evidence, data period, and limitations."
            ),
        },
        {"role": "user", "content": message},
    ]

    last_data: dict[str, Any] | None = None
    try:
        for _ in range(3):
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools,
                tool_choice="required",
            )
            assistant_message = response.choices[0].message
            calls = assistant_message.tool_calls or []
            if not calls:
                return {
                    "reply": assistant_message.content or "The assistant returned no text.",
                    "data": last_data,
                    "mode": "openai_tools",
                }
            messages.append({
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": [call.model_dump() for call in calls],
            })
            for call in calls:
                tool_result = _run_tool(call.function.name, call.function.arguments, data)
                last_data = tool_result
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(tool_result, default=_json_default),
                })
        raise AssistantProviderError("The assistant exceeded the allowed tool-call steps.")
    except OpenAIError as exc:
        raise AssistantProviderError("The configured language-model provider request failed.") from exc

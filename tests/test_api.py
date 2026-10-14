import pandas as pd
import pytest
from pydantic import ValidationError

from api import main
from src.config import settings


def test_risk_endpoint_labels_relative_score_not_probability(monkeypatch):
    scores = pd.DataFrame({
        "customer_id": [11],
        "risk_indicator_score": [0.8],
        "risk_indicator_band": ["upper"],
        "score_type": ["exploratory_relative_indicator"],
    })
    monkeypatch.setattr(main, "get_data", lambda: {"risk_indicator": scores})

    result = main.get_customer_risk_indicator(11)

    assert result["risk_indicator_score"] == 0.8
    assert result["score_is_probability"] is False


def test_chat_rejects_blank_text_and_blocks_customer_identifier(monkeypatch):
    with pytest.raises(ValidationError):
        main.ChatRequest(message="   ")

    monkeypatch.setattr(main, "get_data", lambda: {})
    monkeypatch.setattr(settings, "openai_api_key", "")
    response = main.chat(main.ChatRequest(message="Show customer 123456"))

    assert response.mode == "privacy_blocked"
    assert response.data is None

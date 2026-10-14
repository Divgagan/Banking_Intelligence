import pandas as pd

from src.models.segmentation import prepare_features


def test_categorical_features_use_one_hot_columns():
    frame = pd.DataFrame({
        "customer_id": [1, 2, 3, 4],
        "recency_days": [2, 5, 20, 40],
        "income_bracket": ["low", "mid", "high", "low"],
        "recency_bucket": ["recent", "recent", "moderate", "dormant"],
    })

    _, matrix, _, feature_columns = prepare_features(frame)

    assert matrix.shape[0] == 4
    assert any(column.startswith("income_bracket_") for column in feature_columns)
    assert not any(column.endswith("_enc") for column in feature_columns)

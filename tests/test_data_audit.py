import json

import pandas as pd

from scripts.phase1_inspect import inspect_dataset


def test_data_audit_writes_column_dictionary_without_sample_records(tmp_path):
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "reports"
    raw_dir.mkdir()
    pd.DataFrame({
        "customer_id": [1, 2],
        "age": [30, 45],
    }).to_csv(raw_dir / "customer_data.csv", index=False)
    pd.DataFrame({
        "customer_id": [1, 2],
        "date": ["2023-01-01", "bad-date"],
        "amount": [25.0, -1.0],
        "type": ["Payment", "Withdrawal"],
    }).to_csv(raw_dir / "transactions_data.csv", index=False)

    report = inspect_dataset(raw_dir, output_dir)
    dictionary = pd.read_csv(output_dir / "data_dictionary.csv")
    saved_report = json.loads((output_dir / "data_quality_report.json").read_text())

    assert report["relationships"]["orphan_transaction_customer_ids"] == 0
    assert report["source_files"]["transactions"]["invalid_dates"] == 1
    assert report["source_files"]["transactions"]["invalid_or_nonpositive_amounts"] == 1
    assert set(dictionary["column"]) == {
        "customer_id", "age", "date", "amount", "type",
    }
    assert "licence" in saved_report["provenance_and_licence"]


def test_audit_fails_explicitly_when_required_files_are_missing(tmp_path):
    try:
        inspect_dataset(tmp_path, tmp_path / "reports")
    except FileNotFoundError as exc:
        assert "customer_data.csv" in str(exc)
        assert "transactions_data.csv" in str(exc)
    else:
        raise AssertionError("Expected missing source files to raise FileNotFoundError")

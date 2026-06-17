import pandas as pd
from forecasting import ingest


def test_parse_eia_response_shapes_dataframe(sample_eia_payload):
    df = ingest.parse_eia_response(sample_eia_payload)
    assert list(df.columns) == ["period", "respondent", "value"]
    assert len(df) == 2
    assert str(df["period"].dtype) == "datetime64[ns, UTC]"
    assert df["value"].dtype == "float64"
    assert df.iloc[0]["period"] == pd.Timestamp("2024-01-01T00:00", tz="UTC")
    assert df.iloc[0]["value"] == 85000.0


def test_parse_eia_response_empty(sample_eia_payload):
    empty = {"response": {"total": "0", "data": []}}
    df = ingest.parse_eia_response(empty)
    assert list(df.columns) == ["period", "respondent", "value"]
    assert len(df) == 0

import pandas as pd
from forecasting import ingest


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_fetch_demand_paginates(monkeypatch):
    """Con total=3 y length=2, debe hacer 2 requests y concatenar 3 filas."""
    pages = [
        {"response": {"total": "3", "data": [
            {"period": "2024-01-01T00", "respondent": "PJM", "type": "D", "value": "10"},
            {"period": "2024-01-01T01", "respondent": "PJM", "type": "D", "value": "11"},
        ]}},
        {"response": {"total": "3", "data": [
            {"period": "2024-01-01T02", "respondent": "PJM", "type": "D", "value": "12"},
        ]}},
    ]
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(params["offset"])
        return FakeResponse(pages[len(calls) - 1])

    monkeypatch.setattr(ingest.requests, "get", fake_get)

    df = ingest.fetch_demand(
        respondent="PJM",
        start="2024-01-01T00",
        end="2024-01-01T02",
        api_key="FAKE",
        page_length=2,
    )
    assert len(df) == 3
    assert calls == [0, 2]
    assert df.iloc[2]["value"] == 12.0


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
    assert str(df["period"].dtype) == "datetime64[ns, UTC]"

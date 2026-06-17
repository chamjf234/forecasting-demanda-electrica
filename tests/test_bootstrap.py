import pandas as pd
from forecasting import bootstrap, store


def test_backfill_demand_persists_history(tmp_history_path):
    def fake_fetch(respondent, start, end, api_key):
        return pd.DataFrame(
            {
                "period": pd.to_datetime(
                    ["2024-01-01T00:00", "2024-01-01T01:00"], utc=True
                ),
                "respondent": [respondent, respondent],
                "value": [100.0, 110.0],
            }
        )

    counts = bootstrap.backfill_demand(
        path=tmp_history_path,
        respondent="PJM",
        start="2024-01-01T00",
        end="2024-01-01T01",
        api_key="FAKE",
        now=pd.Timestamp("2024-06-01T12:00", tz="UTC"),
        fetch_fn=fake_fetch,
    )

    df = store.read_demand_history(tmp_history_path)
    assert len(df) == 2
    assert counts["inserted"] == 2

import pandas as pd
from forecasting import store
from forecasting.config import DEMAND_COLUMNS


def test_read_missing_returns_empty_with_schema(tmp_history_path):
    df = store.read_demand_history(tmp_history_path)
    assert list(df.columns) == DEMAND_COLUMNS
    assert len(df) == 0


def _now():
    return pd.Timestamp("2024-06-01T12:00", tz="UTC")


def test_upsert_into_empty_inserts_rows(tmp_history_path, sample_observations):
    result = store.upsert_demand(tmp_history_path, sample_observations, now=_now())
    df = store.read_demand_history(tmp_history_path)
    assert len(df) == 2
    assert result == {"inserted": 2, "revised": 0, "unchanged": 0}
    row = df.sort_values("period").iloc[0]
    assert row["value_first_reported"] == 100.0
    assert row["value_current"] == 100.0
    assert row["first_seen_at"] == _now()


def test_upsert_same_values_is_idempotent(tmp_history_path, sample_observations):
    store.upsert_demand(tmp_history_path, sample_observations, now=_now())
    result = store.upsert_demand(tmp_history_path, sample_observations, now=_now())
    df = store.read_demand_history(tmp_history_path)
    assert len(df) == 2
    assert result == {"inserted": 0, "revised": 0, "unchanged": 2}


def test_upsert_revision_preserves_first_reported(tmp_history_path, sample_observations):
    store.upsert_demand(tmp_history_path, sample_observations, now=_now())
    revised = sample_observations.copy()
    revised.loc[0, "value"] = 105.0
    later = pd.Timestamp("2024-06-02T12:00", tz="UTC")
    result = store.upsert_demand(tmp_history_path, revised, now=later)

    df = store.read_demand_history(tmp_history_path).sort_values("period")
    row = df.iloc[0]
    assert row["value_first_reported"] == 100.0
    assert row["value_current"] == 105.0
    assert row["last_updated_at"] == later
    assert row["first_seen_at"] == _now()
    assert result == {"inserted": 0, "revised": 1, "unchanged": 1}

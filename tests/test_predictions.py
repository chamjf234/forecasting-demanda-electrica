import pandas as pd
import pytest
from forecasting import predictions
from forecasting.predictions import PREDICTION_COLUMNS


@pytest.fixture
def tmp_predictions_path(tmp_path):
    return tmp_path / "predictions.parquet"


def _rows(made_at, model, n=3, base=100.0):
    made = pd.Timestamp(made_at, tz="UTC")
    targets = pd.date_range(made + pd.Timedelta(hours=1), periods=n, freq="h")
    return pd.DataFrame(
        {
            "forecast_made_at": made,
            "model": model,
            "target_period": targets,
            "horizon": range(1, n + 1),
            "prediction": [base + i for i in range(n)],
            "q10": [None] * n,
            "q90": [None] * n,
        }
    )


def test_read_missing_returns_empty_with_schema(tmp_predictions_path):
    df = predictions.read_predictions(tmp_predictions_path)
    assert list(df.columns) == PREDICTION_COLUMNS
    assert len(df) == 0


def test_upsert_inserts(tmp_predictions_path):
    res = predictions.upsert_predictions(tmp_predictions_path, _rows("2026-01-01T00", "naive"))
    df = predictions.read_predictions(tmp_predictions_path)
    assert len(df) == 3
    assert res == {"inserted": 3, "updated": 0}


def test_upsert_is_idempotent(tmp_predictions_path):
    predictions.upsert_predictions(tmp_predictions_path, _rows("2026-01-01T00", "naive"))
    res = predictions.upsert_predictions(tmp_predictions_path, _rows("2026-01-01T00", "naive"))
    df = predictions.read_predictions(tmp_predictions_path)
    assert len(df) == 3
    assert res == {"inserted": 0, "updated": 3}


def test_upsert_different_runs_accumulate(tmp_predictions_path):
    predictions.upsert_predictions(tmp_predictions_path, _rows("2026-01-01T00", "naive"))
    predictions.upsert_predictions(tmp_predictions_path, _rows("2026-01-02T00", "naive"))
    df = predictions.read_predictions(tmp_predictions_path)
    assert len(df) == 6

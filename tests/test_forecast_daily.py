import numpy as np
import pandas as pd
import pytest
from forecasting import forecast_daily
from forecasting.predictions import PREDICTION_COLUMNS


@pytest.fixture
def demand_history():
    """Histórico estilo store (con value_current) de 2 semanas."""
    n = 24 * 14
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "period": idx,
            "respondent": "PJM",
            "value_first_reported": np.arange(n, dtype=float),
            "value_current": np.arange(n, dtype=float),
            "first_seen_at": idx,
            "last_updated_at": idx,
        }
    )


class RecordingModel:
    """Modelo fake que registra qué histórico recibió y devuelve ceros."""
    name = "fake"

    def __init__(self):
        self.last_seen_max_period = None

    def predict(self, history, horizon):
        self.last_seen_max_period = history["period"].max()
        future = pd.date_range(
            history["period"].iloc[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h"
        )
        return pd.DataFrame({"period": future, "model": self.name, "prediction": 0.0})


class ExplodingModel:
    name = "boom"

    def predict(self, history, horizon):
        raise RuntimeError("kaboom")


def test_run_forecast_respects_as_of_cut(demand_history):
    model = RecordingModel()
    as_of = pd.Timestamp("2025-01-07T00:00", tz="UTC")
    preds, failures = forecast_daily.run_forecast(
        demand_history, [model], as_of=as_of, horizon=24
    )
    assert model.last_seen_max_period <= as_of
    assert failures == []
    assert list(preds.columns) == PREDICTION_COLUMNS
    assert len(preds) == 24
    assert (preds["forecast_made_at"] == as_of).all()
    assert preds["horizon"].tolist() == list(range(1, 25))


def test_run_forecast_skips_failing_model(demand_history):
    good, bad = RecordingModel(), ExplodingModel()
    as_of = pd.Timestamp("2025-01-07T00:00", tz="UTC")
    preds, failures = forecast_daily.run_forecast(
        demand_history, [good, bad], as_of=as_of, horizon=24
    )
    assert len(preds) == 24
    assert (preds["model"] == "fake").all()
    assert [name for name, _ in failures] == ["boom"]


def test_forecast_and_store_persists(tmp_path, demand_history):
    model = RecordingModel()
    as_of = pd.Timestamp("2025-01-07T00:00", tz="UTC")
    path = tmp_path / "predictions.parquet"
    summary = forecast_daily.forecast_and_store(
        demand_history, [model], as_of=as_of, horizon=24, predictions_path=path
    )
    from forecasting import predictions
    saved = predictions.read_predictions(path)
    assert len(saved) == 24
    assert summary["stored"]["inserted"] == 24
    assert summary["failures"] == []

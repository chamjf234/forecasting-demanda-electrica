import numpy as np
import pandas as pd
import pytest
from forecasting.models.sarimax import SarimaxForecaster
from forecasting.models.base import FORECAST_COLUMNS


@pytest.mark.slow
def test_sarimax_output_contract(synthetic_hourly_history):
    model = SarimaxForecaster()
    out = model.predict(synthetic_hourly_history, horizon=24)
    assert list(out.columns) == FORECAST_COLUMNS
    assert len(out) == 24
    assert (out["model"] == "sarimax").all()
    assert np.isfinite(out["prediction"]).all()
    last = synthetic_hourly_history["period"].iloc[-1]
    assert out["period"].iloc[0] == last + pd.Timedelta(hours=1)
    assert out["period"].iloc[-1] == last + pd.Timedelta(hours=24)


def test_sarimax_truncates_to_train_window():
    # con un histórico más largo que la ventana, solo se conservan las últimas N horas
    idx = pd.date_range("2025-01-01", periods=1000, freq="h", tz="UTC")
    hist = pd.DataFrame({"period": idx, "value": np.arange(1000, dtype=float)})
    model = SarimaxForecaster(train_window=200)
    sliced = model._train_slice(hist)
    assert len(sliced) == 200
    assert sliced["period"].iloc[-1] == hist["period"].iloc[-1]   # conserva lo más reciente
    assert sliced["period"].iloc[0] == hist["period"].iloc[800]   # arranca 200 atrás


def test_sarimax_train_slice_keeps_short_history():
    idx = pd.date_range("2025-01-01", periods=50, freq="h", tz="UTC")
    hist = pd.DataFrame({"period": idx, "value": np.arange(50, dtype=float)})
    sliced = SarimaxForecaster(train_window=200)._train_slice(hist)
    assert len(sliced) == 50  # más corto que la ventana -> sin cambios

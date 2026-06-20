import numpy as np
import pandas as pd
from forecasting.models.lightgbm_model import LightGBMForecaster
from forecasting.models.base import FORECAST_COLUMNS


def test_lightgbm_output_contract(synthetic_hourly_history):
    model = LightGBMForecaster()
    out = model.predict(synthetic_hourly_history, horizon=24)
    assert list(out.columns) == FORECAST_COLUMNS
    assert len(out) == 24
    assert (out["model"] == "lightgbm").all()
    assert np.isfinite(out["prediction"]).all()
    last = synthetic_hourly_history["period"].iloc[-1]
    assert out["period"].iloc[0] == last + pd.Timedelta(hours=1)


def test_lightgbm_learns_seasonal_signal(synthetic_hourly_history):
    model = LightGBMForecaster()
    out = model.predict(synthetic_hourly_history, horizon=24)
    values = synthetic_hourly_history["value"].to_numpy()
    n = len(values)
    week_ago = np.array([values[n - 168 + i] for i in range(24)])
    corr = np.corrcoef(out["prediction"].to_numpy(), week_ago)[0, 1]
    assert corr > 0.9


def test_lightgbm_raises_when_history_too_short():
    short = pd.DataFrame(
        {"period": pd.date_range("2025-01-01", periods=50, freq="h", tz="UTC"),
         "value": range(50)}
    )
    try:
        LightGBMForecaster().predict(short, horizon=24)
        assert False, "esperaba ValueError"
    except ValueError:
        pass

import numpy as np
import pandas as pd
import pytest
from forecasting.models.chronos_model import ChronosForecaster
from forecasting.models.base import FORECAST_COLUMNS


@pytest.mark.slow
def test_chronos_output_contract(synthetic_hourly_history):
    model = ChronosForecaster(model_id="amazon/chronos-bolt-tiny")
    out = model.predict(synthetic_hourly_history, horizon=24)
    for col in FORECAST_COLUMNS:
        assert col in out.columns
    assert {"q10", "q90"}.issubset(out.columns)
    assert len(out) == 24
    assert (out["model"] == "chronos").all()
    assert np.isfinite(out["prediction"]).all()
    assert (out["q10"] <= out["prediction"] + 1e-6).mean() > 0.9
    assert (out["prediction"] <= out["q90"] + 1e-6).mean() > 0.9
    last = synthetic_hourly_history["period"].iloc[-1]
    assert out["period"].iloc[0] == last + pd.Timedelta(hours=1)

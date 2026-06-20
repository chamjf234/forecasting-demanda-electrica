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

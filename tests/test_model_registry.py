import numpy as np
import pytest
from forecasting import models
from forecasting.models.base import FORECAST_COLUMNS


def test_registry_groups():
    daily = {m.name for m in models.daily_models()}
    weekly = {m.name for m in models.weekly_models()}
    assert daily == {"seasonal_naive", "lightgbm", "chronos"}
    assert weekly == {"sarimax"}
    assert {m.name for m in models.all_models()} == daily | weekly


@pytest.mark.parametrize("model_name", ["seasonal_naive", "lightgbm"])
def test_fast_models_obey_contract(model_name, synthetic_hourly_history):
    model = next(m for m in models.all_models() if m.name == model_name)
    out = model.predict(synthetic_hourly_history, horizon=24)
    for col in FORECAST_COLUMNS:
        assert col in out.columns
    assert len(out) == 24
    assert (out["model"] == model_name).all()
    assert np.isfinite(out["prediction"]).all()

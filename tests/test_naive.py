import pandas as pd
from forecasting.models.naive import SeasonalNaive
from forecasting.models.base import FORECAST_COLUMNS


def test_naive_output_contract(synthetic_hourly_history):
    model = SeasonalNaive()
    out = model.predict(synthetic_hourly_history, horizon=24)
    assert list(out.columns) == FORECAST_COLUMNS
    assert len(out) == 24
    assert (out["model"] == "seasonal_naive").all()
    last = synthetic_hourly_history["period"].iloc[-1]
    assert out["period"].iloc[0] == last + pd.Timedelta(hours=1)
    assert out["period"].iloc[-1] == last + pd.Timedelta(hours=24)


def test_naive_repeats_value_one_week_earlier(synthetic_hourly_history):
    model = SeasonalNaive()
    out = model.predict(synthetic_hourly_history, horizon=24)
    values = synthetic_hourly_history["value"].to_numpy()
    n = len(values)
    for i in range(24):
        assert abs(out["prediction"].iloc[i] - values[n - 168 + i]) < 1e-9


def test_naive_raises_when_history_too_short():
    short = pd.DataFrame(
        {"period": pd.date_range("2025-01-01", periods=10, freq="h", tz="UTC"),
         "value": range(10)}
    )
    model = SeasonalNaive()
    try:
        model.predict(short, horizon=24)
        assert False, "esperaba ValueError"
    except ValueError:
        pass

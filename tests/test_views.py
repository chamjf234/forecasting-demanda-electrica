import numpy as np
import pandas as pd
from forecasting import views
from forecasting.predictions import PREDICTION_COLUMNS


def _history(values, start="2025-01-01"):
    n = len(values)
    idx = pd.date_range(start, periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {"period": idx, "respondent": "PJM",
         "value_first_reported": values, "value_current": values,
         "first_seen_at": idx, "last_updated_at": idx}
    )


def _preds(made_at, model, target_periods, preds):
    return pd.DataFrame(
        {"forecast_made_at": pd.Timestamp(made_at, tz="UTC"), "model": model,
         "target_period": target_periods, "horizon": range(1, len(preds) + 1),
         "prediction": preds, "q10": np.nan, "q90": np.nan}
    )[PREDICTION_COLUMNS]


def test_latest_forecast_keeps_only_most_recent_run_per_model():
    tp1 = pd.date_range("2025-01-02", periods=2, freq="h", tz="UTC")
    tp2 = pd.date_range("2025-01-03", periods=2, freq="h", tz="UTC")
    preds = pd.concat([
        _preds("2025-01-02T00", "naive", tp1, [1.0, 2.0]),
        _preds("2025-01-03T00", "naive", tp2, [3.0, 4.0]),
    ], ignore_index=True)
    out = views.latest_forecast(preds)
    assert (out["forecast_made_at"] == pd.Timestamp("2025-01-03T00", tz="UTC")).all()
    assert len(out) == 2


def test_predicted_vs_actual_joins_on_target():
    hist = _history([100.0, 110.0, 120.0])
    tp = hist["period"].iloc[1:3].to_numpy()
    preds = _preds("2025-01-01T00", "naive", tp, [111.0, 119.0])
    out = views.predicted_vs_actual(preds, hist)
    assert set(["target_period", "model", "prediction", "actual"]).issubset(out.columns)
    assert len(out) == 2
    assert out.sort_values("target_period")["actual"].tolist() == [110.0, 120.0]


def test_rolling_error_per_model():
    hist = _history([100.0, 100.0, 100.0, 100.0])
    tp = hist["period"].to_numpy()
    preds = _preds("2025-01-01T00", "naive", tp, [110.0, 110.0, 110.0, 110.0])
    pva = views.predicted_vs_actual(preds, hist)
    out = views.rolling_error(pva, window=2)
    assert set(["target_period", "model", "rolling_mae"]).issubset(out.columns)
    assert out["rolling_mae"].dropna().iloc[-1] == 10.0

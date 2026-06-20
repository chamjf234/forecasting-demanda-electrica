import numpy as np
import pandas as pd
from forecasting import backtest
from forecasting.predictions import PREDICTION_COLUMNS


def _history(values):
    n = len(values)
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "period": idx,
            "respondent": "PJM",
            "value_first_reported": values,
            "value_current": values,
            "first_seen_at": idx,
            "last_updated_at": idx,
        }
    )


def _predictions(target_periods, model, preds):
    made = pd.Timestamp("2025-01-01T00", tz="UTC")
    return pd.DataFrame(
        {
            "forecast_made_at": made,
            "model": model,
            "target_period": target_periods,
            "horizon": range(1, len(preds) + 1),
            "prediction": preds,
            "q10": np.nan,
            "q90": np.nan,
        }
    )[PREDICTION_COLUMNS]


def test_evaluate_predictions_per_model():
    hist = _history([100.0, 110.0, 120.0, 130.0])
    tp = hist["period"].iloc[1:4].to_numpy()
    preds = pd.concat(
        [
            _predictions(tp, "A", [110.0, 120.0, 130.0]),
            _predictions(tp, "B", [120.0, 130.0, 140.0]),
        ],
        ignore_index=True,
    )
    out = backtest.evaluate_predictions(preds, hist).set_index("model")
    assert out.loc["A", "mae"] == 0.0
    assert out.loc["B", "mae"] == 10.0
    assert out.loc["A", "n"] == 3
    assert out.loc["B", "n"] == 3


def test_evaluate_predictions_inner_join_drops_unmatched():
    hist = _history([100.0, 110.0])
    future = pd.date_range("2025-02-01", periods=2, freq="h", tz="UTC")
    preds = _predictions(future, "A", [1.0, 2.0])
    out = backtest.evaluate_predictions(preds, hist)
    assert len(out) == 0

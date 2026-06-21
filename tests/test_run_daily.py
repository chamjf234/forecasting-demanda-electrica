import numpy as np
import pandas as pd
from forecasting import run_daily, store, predictions


def _seed_history(path, n=24 * 8):
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    obs = pd.DataFrame({"period": idx, "respondent": "PJM", "value": np.arange(n, dtype=float)})
    store.upsert_demand(path, obs, now=pd.Timestamp("2025-01-10", tz="UTC"))


def test_run_daily_ingests_then_forecasts(tmp_path):
    hist_path = tmp_path / "demand_history.parquet"
    preds_path = tmp_path / "predictions.parquet"
    _seed_history(hist_path)

    def fake_fetch(respondent, start, end, api_key):
        new_idx = pd.date_range("2025-01-09", periods=24, freq="h", tz="UTC")
        return pd.DataFrame({"period": new_idx, "respondent": respondent,
                             "value": np.arange(100, 124, dtype=float)})

    class DummyModel:
        name = "dummy"
        def predict(self, history, horizon):
            fut = pd.date_range(history["period"].iloc[-1] + pd.Timedelta(hours=1),
                                periods=horizon, freq="h")
            return pd.DataFrame({"period": fut, "model": self.name, "prediction": 0.0})

    summary = run_daily.run_daily(
        history_path=hist_path, predictions_path=preds_path,
        models=[DummyModel()], api_key="FAKE",
        now=pd.Timestamp("2025-01-09T23:00", tz="UTC"),
        recent_days=2, fetch_fn=fake_fetch, horizon=24,
    )
    hist = store.read_demand_history(hist_path)
    assert hist["period"].max() == pd.Timestamp("2025-01-09T23:00", tz="UTC")
    saved = predictions.read_predictions(preds_path)
    assert len(saved) == 24
    assert summary["ingested"]["inserted"] >= 1
    assert summary["forecast"]["failures"] == []

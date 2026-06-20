"""SARIMAX vía dynamic harmonic regression: Fourier (24h,168h) como exógenas + AutoARIMA."""
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

from forecasting.models.base import FORECAST_COLUMNS, future_periods

SEASONAL_PERIODS = (24, 168)
N_HARMONICS = (3, 3)


def _fourier_terms(periods: pd.DatetimeIndex, origin: pd.Timestamp) -> pd.DataFrame:
    hours = ((periods - origin) / pd.Timedelta(hours=1)).to_numpy(dtype=float)
    cols = {}
    for period, k_max in zip(SEASONAL_PERIODS, N_HARMONICS):
        for k in range(1, k_max + 1):
            cols[f"sin_{period}_{k}"] = np.sin(2 * np.pi * k * hours / period)
            cols[f"cos_{period}_{k}"] = np.cos(2 * np.pi * k * hours / period)
    return pd.DataFrame(cols)


class SarimaxForecaster:
    name = "sarimax"

    def predict(self, history: pd.DataFrame, horizon: int) -> pd.DataFrame:
        origin = history["period"].iloc[0]
        hist_idx = pd.DatetimeIndex(history["period"])
        future = future_periods(history, horizon)

        X_hist = _fourier_terms(hist_idx, origin)
        X_fut = _fourier_terms(future, origin)

        train = pd.DataFrame(
            {"unique_id": "PJM", "ds": hist_idx.tz_localize(None), "y": history["value"].to_numpy()}
        )
        train = pd.concat([train, X_hist], axis=1)

        X_df = pd.DataFrame({"unique_id": "PJM", "ds": future.tz_localize(None)})
        X_df = pd.concat([X_df, X_fut], axis=1)

        sf = StatsForecast(models=[AutoARIMA(seasonal=False)], freq="h")
        fc = sf.forecast(df=train, h=horizon, X_df=X_df)

        return pd.DataFrame(
            {"period": future, "model": self.name, "prediction": fc["AutoARIMA"].to_numpy()}
        )[FORECAST_COLUMNS]

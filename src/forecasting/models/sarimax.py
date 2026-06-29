"""SARIMAX vía dynamic harmonic regression: Fourier (24h,168h) como exógenas + AutoARIMA."""
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

from forecasting.models.base import FORECAST_COLUMNS, future_periods

SEASONAL_PERIODS = (24, 168)
N_HARMONICS = (3, 3)

# Ventana de entrenamiento por defecto (horas). AutoARIMA sobre años de datos
# horarios consume demasiada memoria (mata el runner gratuito de CI). Para un
# forecast a 24 h, la dinámica reciente es lo que importa: 90 días bastan y el
# ajuste baja de "se queda sin RAM" a ~10 s.
DEFAULT_TRAIN_WINDOW = 24 * 90


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

    def __init__(self, train_window: int = DEFAULT_TRAIN_WINDOW):
        self.train_window = train_window

    def _train_slice(self, history: pd.DataFrame) -> pd.DataFrame:
        """Acota el entrenamiento a las últimas `train_window` horas (memoria/tiempo)."""
        if self.train_window and len(history) > self.train_window:
            return history.tail(self.train_window).reset_index(drop=True)
        return history.reset_index(drop=True)

    def predict(self, history: pd.DataFrame, horizon: int) -> pd.DataFrame:
        history = self._train_slice(history)
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

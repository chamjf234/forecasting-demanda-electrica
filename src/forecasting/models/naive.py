"""Seasonal naive: el valor de hace exactamente una semana (168 h)."""
import pandas as pd

from forecasting.models.base import FORECAST_COLUMNS, future_periods

WEEK_HOURS = 168


class SeasonalNaive:
    name = "seasonal_naive"

    def predict(self, history: pd.DataFrame, horizon: int) -> pd.DataFrame:
        if len(history) < WEEK_HOURS:
            raise ValueError(
                f"seasonal_naive necesita >= {WEEK_HOURS} h de histórico, "
                f"recibió {len(history)}"
            )
        values = history["value"].to_numpy()
        n = len(values)
        preds = [values[n - WEEK_HOURS + i] for i in range(horizon)]
        return pd.DataFrame(
            {
                "period": future_periods(history, horizon),
                "model": self.name,
                "prediction": preds,
            }
        )[FORECAST_COLUMNS]

"""Contrato común de los modelos de forecasting y helpers compartidos."""
from __future__ import annotations

import pandas as pd

# Columnas mínimas que todo modelo devuelve. Los probabilísticos pueden añadir más.
FORECAST_COLUMNS = ["period", "model", "prediction"]


def future_periods(history: pd.DataFrame, horizon: int) -> pd.DatetimeIndex:
    """Las próximas `horizon` horas (UTC) después de la última del histórico."""
    last = history["period"].iloc[-1]
    return pd.date_range(
        start=last + pd.Timedelta(hours=1), periods=horizon, freq="h"
    )

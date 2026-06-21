"""Orquestación del forecast diario: corte as-of, corre modelos, persiste predicciones."""
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from forecasting import predictions, series, store
from forecasting.config import DEMAND_HISTORY_PATH, PREDICTIONS_PATH
from forecasting.predictions import PREDICTION_COLUMNS


def _as_series(history: pd.DataFrame) -> pd.DataFrame:
    """Convierte el histórico del store a la serie limpia (period, value) que usan los modelos.

    Regulariza a grilla horaria contigua: los datos de la EIA tienen huecos que
    romperían a los modelos basados en lags/posición (ver forecasting.series).
    """
    raw = pd.DataFrame(
        {"period": history["period"], "value": history["value_current"]}
    )
    return series.regularize_hourly(raw)


def run_forecast(history: pd.DataFrame, models, as_of: pd.Timestamp, horizon: int = 24):
    """Corre cada modelo sobre el histórico cortado en as_of.

    Devuelve (predictions_df, failures) donde failures es lista de (model_name, error_str).
    Robustez: un modelo que lanza excepción se omite y se registra.
    """
    series = _as_series(history)
    sliced = series[series["period"] <= as_of].reset_index(drop=True)
    last_period = sliced["period"].iloc[-1]

    frames = []
    failures = []
    for model in models:
        try:
            fc = model.predict(sliced, horizon)
        except Exception as exc:  # noqa: BLE001 — un modelo no debe tumbar el loop
            failures.append((model.name, str(exc)))
            continue
        rows = pd.DataFrame(
            {
                "forecast_made_at": as_of,
                "model": model.name,
                "target_period": fc["period"].to_numpy(),
                "horizon": (
                    (fc["period"] - last_period) / pd.Timedelta(hours=1)
                ).astype("int64").to_numpy(),
                "prediction": fc["prediction"].to_numpy(),
                "q10": fc["q10"].to_numpy() if "q10" in fc.columns else float("nan"),
                "q90": fc["q90"].to_numpy() if "q90" in fc.columns else float("nan"),
            }
        )
        frames.append(rows)

    if frames:
        preds = pd.concat(frames, ignore_index=True)[PREDICTION_COLUMNS]
    else:
        preds = predictions._empty()
    return preds, failures


def forecast_and_store(
    history, models, as_of, horizon=24, predictions_path=PREDICTIONS_PATH
) -> dict:
    """run_forecast + persistencia. Devuelve resumen {stored, failures}."""
    preds, failures = run_forecast(history, models, as_of, horizon)
    stored = predictions.upsert_predictions(predictions_path, preds)
    return {"stored": stored, "failures": failures}


if __name__ == "__main__":
    from forecasting.models import daily_models

    load_dotenv()
    history = store.read_demand_history(DEMAND_HISTORY_PATH)
    as_of = history["period"].max()
    summary = forecast_and_store(history, daily_models(), as_of=as_of, horizon=24)
    print(f"Forecast as_of={as_of}: {summary}")

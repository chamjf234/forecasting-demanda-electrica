"""Evaluación de forecasts en dos regímenes: forward-test y backtest histórico."""
import pandas as pd

from forecasting import metrics


def _metrics_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """df con columnas model/actual/prediction -> MAE/sMAPE/n por modelo."""
    rows = []
    for model, g in df.groupby("model"):
        rows.append(
            {
                "model": model,
                "mae": metrics.mae(g["actual"], g["prediction"]),
                "smape": metrics.smape(g["actual"], g["prediction"]),
                "n": len(g),
            }
        )
    return pd.DataFrame(rows)


def evaluate_predictions(
    predictions: pd.DataFrame, history: pd.DataFrame, actual_col: str = "value_current"
) -> pd.DataFrame:
    """Forward-test: une predicciones guardadas con los reales y mide error por modelo.

    Une por target_period == period. Inner join: predicciones sin real (futuro aún
    no observado) se descartan.
    """
    actuals = history[["period", actual_col]].rename(
        columns={"period": "target_period", actual_col: "actual"}
    )
    merged = predictions.merge(actuals, on="target_period", how="inner")
    return _metrics_by_model(merged)

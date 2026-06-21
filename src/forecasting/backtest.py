"""Evaluación de forecasts en dos regímenes: forward-test y backtest histórico."""
import pandas as pd

from forecasting import metrics, series


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


def rolling_origin_backtest(
    history: pd.DataFrame,
    models,
    origins,
    horizon: int = 24,
    actual_col: str = "value_current",
) -> pd.DataFrame:
    """Backtest histórico (optimista): para cada origin corta el histórico, corre los
    modelos y compara contra el valor real (revisado). Agrega MAE/sMAPE por modelo.
    """
    # Serie regularizada (huecos interpolados) como entrada de los modelos;
    # los reales para evaluar se mantienen sobre los datos observados originales.
    model_input = series.regularize_hourly(
        pd.DataFrame({"period": history["period"], "value": history[actual_col]})
    )
    actual_lookup = dict(zip(history["period"], history[actual_col]))

    records = []
    for origin in origins:
        sliced = model_input[model_input["period"] <= origin].reset_index(drop=True)
        for model in models:
            try:
                fc = model.predict(sliced, horizon)
            except Exception:  # noqa: BLE001 — un modelo no debe tumbar el backtest
                continue
            for target, pred in zip(fc["period"], fc["prediction"]):
                if target in actual_lookup:
                    records.append(
                        {"model": model.name, "actual": actual_lookup[target], "prediction": pred}
                    )

    return _metrics_by_model(pd.DataFrame(records, columns=["model", "actual", "prediction"]))

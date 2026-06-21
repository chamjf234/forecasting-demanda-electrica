"""Preparación de datos para el dashboard (funciones puras, testeables)."""
import pandas as pd


def latest_forecast(predictions: pd.DataFrame) -> pd.DataFrame:
    """Las filas del forecast más reciente por modelo (panel 'forecast actual')."""
    if predictions.empty:
        return predictions
    latest = predictions.groupby("model")["forecast_made_at"].transform("max")
    return predictions[predictions["forecast_made_at"] == latest].reset_index(drop=True)


def predicted_vs_actual(
    predictions: pd.DataFrame, history: pd.DataFrame, actual_col: str = "value_current"
) -> pd.DataFrame:
    """Une predicciones con el valor real por target_period (para graficar)."""
    actuals = history[["period", actual_col]].rename(
        columns={"period": "target_period", actual_col: "actual"}
    )
    merged = predictions.merge(actuals, on="target_period", how="inner")
    return merged[["target_period", "model", "prediction", "actual"]]


def rolling_error(predicted_vs_actual_df: pd.DataFrame, window: int = 168) -> pd.DataFrame:
    """MAE rodante por modelo a lo largo del tiempo (degradación visible)."""
    df = predicted_vs_actual_df.copy()
    df["abs_error"] = (df["actual"] - df["prediction"]).abs()
    df = df.sort_values(["model", "target_period"])
    df["rolling_mae"] = (
        df.groupby("model")["abs_error"]
        .rolling(window, min_periods=window)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return df[["target_period", "model", "rolling_mae"]].reset_index(drop=True)

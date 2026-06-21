"""Regularización de la serie temporal antes de alimentar a los modelos.

Los datos de la EIA tienen huecos ocasionales (horas faltantes) y valores NaN.
Los modelos asumen una serie horaria contigua (el seasonal naive usa posiciones,
LightGBM busca lags por timestamp exacto). Regularizar a una grilla horaria
completa e interpolar los huecos evita errores y mantiene la semántica temporal.
"""
import pandas as pd


def regularize_hourly(df: pd.DataFrame, value_col: str = "value") -> pd.DataFrame:
    """Reindexa a una grilla horaria UTC completa (min..max) e interpola huecos.

    Devuelve un DataFrame con columnas [period, value_col] contiguo por hora.
    """
    if df.empty:
        return df[["period", value_col]]

    s = df.set_index("period")[value_col].sort_index()
    full = pd.date_range(s.index.min(), s.index.max(), freq="h")  # hereda el tz UTC
    # interpolación por tiempo: rellena huecos internos y NaN; los extremos ya existen.
    s = s.reindex(full).interpolate(method="time")

    out = s.reset_index()
    out.columns = ["period", value_col]
    return out

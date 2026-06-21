"""Limpieza y regularización de la serie temporal antes de alimentar a los modelos.

Los datos de la EIA del mundo real traen dos problemas:
  1. Huecos: horas faltantes (p.ej. en cambios de horario). Los modelos asumen una
     serie horaria contigua (el seasonal naive usa posiciones; LightGBM busca lags
     por timestamp exacto), así que un hueco rompe o desalinea.
  2. Valores basura: centinelas (~2^31 ≈ 2.1e9 que la EIA usa como "missing"),
     ceros y outliers absurdos que corrompen tanto el entrenamiento como la
     evaluación (un "valor real" de 2.1e9 infla el error de todos los modelos).

`regularize_hourly` resuelve ambos: marca como faltantes los valores implausibles,
reindexa a una grilla horaria completa, e interpola. Devuelve una serie limpia,
contigua y lista para modelar/evaluar.
"""
import pandas as pd

# Un valor se considera implausible (centinela/outlier) si es <= 0 o si supera
# este múltiplo de la mediana robusta. La demanda eléctrica vive en una banda
# estrecha y positiva, así que un múltiplo generoso (5x) descarta basura
# (millones/miles de millones) sin tocar los picos reales.
MAX_MEDIAN_RATIO = 5.0


def regularize_hourly(df: pd.DataFrame, value_col: str = "value") -> pd.DataFrame:
    """Limpia outliers, reindexa a grilla horaria UTC completa e interpola huecos.

    Devuelve un DataFrame con columnas [period, value_col] contiguo por hora y
    sin valores implausibles.
    """
    if df.empty:
        return df[["period", value_col]]

    s = df.set_index("period")[value_col].sort_index()

    # 1. Marca valores implausibles como faltantes (NaN) usando la mediana robusta.
    positive = s[s > 0]
    if not positive.empty:
        med = positive.median()
        implausible = (s <= 0) | (s > MAX_MEDIAN_RATIO * med)
        s = s.mask(implausible)

    # 2. Reindexa a la grilla horaria completa (rellena huecos como NaN).
    full = pd.date_range(s.index.min(), s.index.max(), freq="h")  # hereda el tz UTC

    # 3. Interpola por tiempo; ffill/bfill cubre cualquier NaN en los extremos.
    s = s.reindex(full).interpolate(method="time", limit_direction="both").ffill().bfill()

    out = s.reset_index()
    out.columns = ["period", value_col]
    return out

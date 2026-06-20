"""Métricas de error de forecasting (puras, sin estado)."""
import numpy as np


def mae(y_true, y_pred) -> float:
    """Mean Absolute Error (en las unidades de la serie, p.ej. MWh)."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(yt - yp)))


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE en % (0 = perfecto). Robusto a denominador cero."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    denom = np.abs(yt) + np.abs(yp)
    # División segura: solo divide donde denom != 0; el resto queda en 0 (sin warning).
    contrib = np.zeros_like(denom)
    np.divide(2 * np.abs(yp - yt), denom, out=contrib, where=denom != 0)
    return float(np.mean(contrib) * 100)

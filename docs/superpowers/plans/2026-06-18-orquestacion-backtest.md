# Orquestación + Backtest — Implementation Plan (Plan 3 de 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Orquestar el forecast diario (corre los modelos sobre el histórico cortado *as-of* y persiste las predicciones) y evaluar la calidad con dos regímenes: forward-test (predicciones en vivo vs reales) y backtest histórico rolling-origin (datos revisados, optimista).

**Architecture:** Cuatro módulos nuevos con responsabilidad única. `metrics` (MAE/sMAPE, puro). `predictions` (persistencia de predicciones en Parquet, idempotente). `forecast_daily` (orquesta: corta as-of → corre modelos robustamente → arma filas de predicción → persiste). `backtest` (une predicciones con reales y calcula métricas por modelo, en los dos regímenes). Reusa `store` (histórico) y `models` (registry) de los Planes 1-2.

**Tech Stack:** Python, pandas, numpy. Sobre la fundación de Planes 1-2.

---

## Contexto para quien ejecuta (leer antes de empezar)

- **Repo:** `C:\Users\asus\Documents\forecasting-demanda-electrica`. Rama de feature creada por el controlador. Venv: `.venv/Scripts/python.exe -m pytest ...`.
- **Spec:** `docs/superpowers/specs/2026-06-15-forecasting-demanda-electrica-design.md` §4, §4.1, §5.
- **El corte *as-of* (anti-leakage, CLAVE):** al pronosticar en un momento `as_of`, el modelo solo puede ver datos con `period <= as_of`. `forecast_daily` corta el histórico ahí antes de pasarlo al modelo. Nunca se predice con información del futuro.
- **Dos regímenes de evaluación (spec §4.1):**
  - *Forward-test (honesto):* evalúa las predicciones que el loop guardó en vivo (`data/predictions.parquet`) contra los valores reales del histórico. Es point-in-time real.
  - *Backtest histórico (optimista):* re-corre los modelos sobre el histórico viejo con rolling-origin y evalúa contra `value_current` (datos ya revisados). Llena el dashboard desde el día 1 pero es ligeramente optimista. Se etiqueta como tal.
- **Contrato de modelo (de Plan 2):** cada modelo tiene `.name` y `.predict(history, horizon) -> DataFrame` con columnas `period`/`model`/`prediction` (+ opcional `q10`/`q90`). `history` es un DataFrame con `period` (UTC horaria contigua) y `value`.
- **Histórico (de Plan 1):** `store.read_demand_history(path)` devuelve columnas `period`, `respondent`, `value_first_reported`, `value_current`, `first_seen_at`, `last_updated_at`. Para alimentar modelos se usa `value_current` como `value`.

---

## File Structure

```
src/forecasting/
├── metrics.py          # mae, smape (puros)
├── predictions.py      # PREDICTION_COLUMNS, read_predictions, upsert_predictions
├── forecast_daily.py   # run_forecast, forecast_and_store, __main__ (entrypoint del cron)
└── backtest.py         # evaluate_predictions (forward), rolling_origin_backtest (histórico)
tests/
├── test_metrics.py
├── test_predictions.py
├── test_forecast_daily.py
└── test_backtest.py
config.py                # añade PREDICTIONS_PATH
```

---

## Task 1: `metrics` — MAE y sMAPE

**Files:**
- Create: `src/forecasting/metrics.py`
- Test: `tests/test_metrics.py`

sMAPE (symmetric MAPE) se elige sobre MAPE clásico porque evita la división por cero y es estándar en comparación de forecasts. MAE da el error en MWh (interpretable); sMAPE da el error relativo en %.

- [ ] **Step 1: Escribir el test que falla `tests/test_metrics.py`**
```python
import numpy as np
from forecasting import metrics


def test_mae_zero_when_perfect():
    assert metrics.mae([1, 2, 3], [1, 2, 3]) == 0.0


def test_mae_known_value():
    # errores |0-1|, |0-3| -> media 2
    assert metrics.mae([0, 0], [1, 3]) == 2.0


def test_smape_zero_when_perfect():
    assert metrics.smape([10, 20], [10, 20]) == 0.0


def test_smape_known_value():
    # y=100, yhat=110 -> 2*10/210 = 0.0952..; *100
    val = metrics.smape([100.0], [110.0])
    assert abs(val - (2 * 10 / 210) * 100) < 1e-9


def test_smape_handles_zero_denominator():
    # ambos 0 -> contribución 0, no NaN
    assert metrics.smape([0.0, 100.0], [0.0, 100.0]) == 0.0
```

- [ ] **Step 2: Correr y verificar que falla**

Run: `.venv/Scripts/python.exe -m pytest tests/test_metrics.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'forecasting.metrics'`).

- [ ] **Step 3: Implementar `src/forecasting/metrics.py`**
```python
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
    # donde el denominador es 0 (ambos valores 0), la contribución es 0, no NaN.
    contrib = np.where(denom == 0, 0.0, 2 * np.abs(yp - yt) / denom)
    return float(np.mean(contrib) * 100)
```

- [ ] **Step 4: Correr y verificar que pasa**

Run: `.venv/Scripts/python.exe -m pytest tests/test_metrics.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**
```bash
git add src/forecasting/metrics.py tests/test_metrics.py
git commit -m "feat(metrics): MAE y sMAPE"
```

---

## Task 2: `predictions` — persistencia idempotente de predicciones

**Files:**
- Create: `src/forecasting/predictions.py`
- Modify: `src/forecasting/config.py`
- Test: `tests/test_predictions.py`

Cada predicción se identifica por `(forecast_made_at, model, target_period)`. `upsert` re-ejecutable: re-correr el mismo forecast no duplica filas.

- [ ] **Step 1: Añadir la ruta en `src/forecasting/config.py`**

Después de la línea `DEMAND_HISTORY_PATH = DATA_DIR / "demand_history.parquet"`, añadir:
```python
PREDICTIONS_PATH = DATA_DIR / "predictions.parquet"
```

- [ ] **Step 2: Escribir el test que falla `tests/test_predictions.py`**
```python
import pandas as pd
import pytest
from forecasting import predictions
from forecasting.predictions import PREDICTION_COLUMNS


@pytest.fixture
def tmp_predictions_path(tmp_path):
    return tmp_path / "predictions.parquet"


def _rows(made_at, model, n=3, base=100.0):
    made = pd.Timestamp(made_at, tz="UTC")
    targets = pd.date_range(made + pd.Timedelta(hours=1), periods=n, freq="h")
    return pd.DataFrame(
        {
            "forecast_made_at": made,
            "model": model,
            "target_period": targets,
            "horizon": range(1, n + 1),
            "prediction": [base + i for i in range(n)],
            "q10": [None] * n,
            "q90": [None] * n,
        }
    )


def test_read_missing_returns_empty_with_schema(tmp_predictions_path):
    df = predictions.read_predictions(tmp_predictions_path)
    assert list(df.columns) == PREDICTION_COLUMNS
    assert len(df) == 0


def test_upsert_inserts(tmp_predictions_path):
    res = predictions.upsert_predictions(tmp_predictions_path, _rows("2026-01-01T00", "naive"))
    df = predictions.read_predictions(tmp_predictions_path)
    assert len(df) == 3
    assert res == {"inserted": 3, "updated": 0}


def test_upsert_is_idempotent(tmp_predictions_path):
    predictions.upsert_predictions(tmp_predictions_path, _rows("2026-01-01T00", "naive"))
    res = predictions.upsert_predictions(tmp_predictions_path, _rows("2026-01-01T00", "naive"))
    df = predictions.read_predictions(tmp_predictions_path)
    assert len(df) == 3
    assert res == {"inserted": 0, "updated": 3}


def test_upsert_different_runs_accumulate(tmp_predictions_path):
    predictions.upsert_predictions(tmp_predictions_path, _rows("2026-01-01T00", "naive"))
    predictions.upsert_predictions(tmp_predictions_path, _rows("2026-01-02T00", "naive"))
    df = predictions.read_predictions(tmp_predictions_path)
    assert len(df) == 6
```

- [ ] **Step 3: Correr y verificar que falla**

Run: `.venv/Scripts/python.exe -m pytest tests/test_predictions.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'forecasting.predictions'`).

- [ ] **Step 4: Implementar `src/forecasting/predictions.py`**
```python
"""Persistencia idempotente de predicciones en Parquet."""
from pathlib import Path

import pandas as pd

PREDICTION_COLUMNS = [
    "forecast_made_at",   # datetime64[ns, UTC] — corte as-of del forecast
    "model",              # str
    "target_period",      # datetime64[ns, UTC] — hora pronosticada
    "horizon",            # int — horas adelante (1..24)
    "prediction",         # float
    "q10",                # float, nullable (solo modelos probabilísticos)
    "q90",                # float, nullable
]

# Clave única de una predicción.
_KEY = ["forecast_made_at", "model", "target_period"]


def _empty() -> pd.DataFrame:
    df = pd.DataFrame(columns=PREDICTION_COLUMNS)
    df["forecast_made_at"] = pd.to_datetime(df["forecast_made_at"], utc=True)
    df["target_period"] = pd.to_datetime(df["target_period"], utc=True)
    df["horizon"] = df["horizon"].astype("int64")
    for col in ("prediction", "q10", "q90"):
        df[col] = df[col].astype("float64")
    df["model"] = df["model"].astype("object")
    return df


def read_predictions(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return _empty()
    return pd.read_parquet(path)


def upsert_predictions(path: Path, new_rows: pd.DataFrame) -> dict:
    """Funde predicciones nuevas; clave (forecast_made_at, model, target_period).

    Re-ejecutable: una predicción ya existente se reemplaza (no se duplica).
    Devuelve {"inserted", "updated"}.
    """
    path = Path(path)
    new_rows = new_rows.drop_duplicates(subset=_KEY, keep="last")[PREDICTION_COLUMNS]
    existing = read_predictions(path)

    existing_keys = set(map(tuple, existing[_KEY].to_numpy())) if len(existing) else set()
    new_keys = set(map(tuple, new_rows[_KEY].to_numpy()))
    inserted = len(new_keys - existing_keys)
    updated = len(new_keys & existing_keys)

    # Conserva las filas existentes cuya clave NO está en el batch nuevo.
    if len(existing):
        keep_mask = ~existing.set_index(_KEY).index.isin(new_keys)
        kept = existing[keep_mask]
    else:
        kept = existing

    merged = (
        pd.concat([kept, new_rows], ignore_index=True)
        .sort_values(_KEY)
        .reset_index(drop=True)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(path, index=False)
    return {"inserted": inserted, "updated": updated}
```

- [ ] **Step 5: Correr y verificar que pasa**

Run: `.venv/Scripts/python.exe -m pytest tests/test_predictions.py -v`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**
```bash
git add src/forecasting/predictions.py src/forecasting/config.py tests/test_predictions.py
git commit -m "feat(predictions): persistencia idempotente de predicciones"
```

---

## Task 3: `forecast_daily` — orquestación con corte as-of, robusta

**Files:**
- Create: `src/forecasting/forecast_daily.py`
- Test: `tests/test_forecast_daily.py`

`run_forecast` corta el histórico en `as_of`, corre cada modelo, y arma filas de predicción. Si un modelo falla, se omite y se reporta (un fallo no tumba el loop). `forecast_and_store` añade la persistencia. El `__main__` es el entrypoint del cron diario.

- [ ] **Step 1: Escribir el test que falla `tests/test_forecast_daily.py`**
```python
import numpy as np
import pandas as pd
import pytest
from forecasting import forecast_daily
from forecasting.predictions import PREDICTION_COLUMNS


@pytest.fixture
def demand_history():
    """Histórico estilo store (con value_current) de 2 semanas."""
    n = 24 * 14
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "period": idx,
            "respondent": "PJM",
            "value_first_reported": np.arange(n, dtype=float),
            "value_current": np.arange(n, dtype=float),
            "first_seen_at": idx,
            "last_updated_at": idx,
        }
    )


class RecordingModel:
    """Modelo fake que registra qué histórico recibió y devuelve ceros."""
    name = "fake"

    def __init__(self):
        self.last_seen_max_period = None

    def predict(self, history, horizon):
        self.last_seen_max_period = history["period"].max()
        future = pd.date_range(
            history["period"].iloc[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h"
        )
        return pd.DataFrame({"period": future, "model": self.name, "prediction": 0.0})


class ExplodingModel:
    name = "boom"

    def predict(self, history, horizon):
        raise RuntimeError("kaboom")


def test_run_forecast_respects_as_of_cut(demand_history):
    model = RecordingModel()
    as_of = pd.Timestamp("2025-01-07T00:00", tz="UTC")
    preds, failures = forecast_daily.run_forecast(
        demand_history, [model], as_of=as_of, horizon=24
    )
    # el modelo nunca vio datos posteriores al corte
    assert model.last_seen_max_period <= as_of
    assert failures == []
    assert list(preds.columns) == PREDICTION_COLUMNS
    assert len(preds) == 24
    assert (preds["forecast_made_at"] == as_of).all()
    assert preds["horizon"].tolist() == list(range(1, 25))


def test_run_forecast_skips_failing_model(demand_history):
    good, bad = RecordingModel(), ExplodingModel()
    as_of = pd.Timestamp("2025-01-07T00:00", tz="UTC")
    preds, failures = forecast_daily.run_forecast(
        demand_history, [good, bad], as_of=as_of, horizon=24
    )
    assert len(preds) == 24                 # solo el bueno
    assert (preds["model"] == "fake").all()
    assert [name for name, _ in failures] == ["boom"]


def test_forecast_and_store_persists(tmp_path, demand_history):
    model = RecordingModel()
    as_of = pd.Timestamp("2025-01-07T00:00", tz="UTC")
    path = tmp_path / "predictions.parquet"
    summary = forecast_daily.forecast_and_store(
        demand_history, [model], as_of=as_of, horizon=24, predictions_path=path
    )
    from forecasting import predictions
    saved = predictions.read_predictions(path)
    assert len(saved) == 24
    assert summary["stored"]["inserted"] == 24
    assert summary["failures"] == []
```

- [ ] **Step 2: Correr y verificar que falla**

Run: `.venv/Scripts/python.exe -m pytest tests/test_forecast_daily.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'forecasting.forecast_daily'`).

- [ ] **Step 3: Implementar `src/forecasting/forecast_daily.py`**
```python
"""Orquestación del forecast diario: corte as-of, corre modelos, persiste predicciones."""
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from forecasting import predictions, store
from forecasting.config import DEMAND_HISTORY_PATH, PREDICTIONS_PATH
from forecasting.predictions import PREDICTION_COLUMNS


def _as_series(history: pd.DataFrame) -> pd.DataFrame:
    """Convierte el histórico del store a la serie limpia (period, value) que usan los modelos."""
    return pd.DataFrame(
        {"period": history["period"], "value": history["value_current"]}
    )


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
    # Entrypoint del cron diario: corre los modelos diarios con as_of = ahora.
    from forecasting.models import daily_models

    load_dotenv()
    os.environ.setdefault("EIA_API_KEY", "")  # no se usa aquí, pero mantiene consistencia
    history = store.read_demand_history(DEMAND_HISTORY_PATH)
    as_of = history["period"].max()  # último dato disponible
    summary = forecast_and_store(history, daily_models(), as_of=as_of, horizon=24)
    print(f"Forecast as_of={as_of}: {summary}")
```

- [ ] **Step 4: Correr y verificar que pasa**

Run: `.venv/Scripts/python.exe -m pytest tests/test_forecast_daily.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Correr la suite completa (rápida)**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS.

- [ ] **Step 6: Commit**
```bash
git add src/forecasting/forecast_daily.py tests/test_forecast_daily.py
git commit -m "feat(forecast_daily): orquestación con corte as-of, robusta a fallos de modelo"
```

---

## Task 4: `backtest` — forward-test (predicciones en vivo vs reales)

**Files:**
- Create: `src/forecasting/backtest.py`
- Test: `tests/test_backtest.py`

`evaluate_predictions` une las predicciones guardadas con los valores reales del histórico (por `target_period`) y calcula MAE/sMAPE por modelo. Es el régimen forward-test (honesto).

- [ ] **Step 1: Escribir el test que falla `tests/test_backtest.py`**
```python
import numpy as np
import pandas as pd
from forecasting import backtest
from forecasting.predictions import PREDICTION_COLUMNS


def _history(values):
    n = len(values)
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "period": idx,
            "respondent": "PJM",
            "value_first_reported": values,
            "value_current": values,
            "first_seen_at": idx,
            "last_updated_at": idx,
        }
    )


def _predictions(target_periods, model, preds):
    made = pd.Timestamp("2025-01-01T00", tz="UTC")
    return pd.DataFrame(
        {
            "forecast_made_at": made,
            "model": model,
            "target_period": target_periods,
            "horizon": range(1, len(preds) + 1),
            "prediction": preds,
            "q10": np.nan,
            "q90": np.nan,
        }
    )[PREDICTION_COLUMNS]


def test_evaluate_predictions_per_model():
    hist = _history([100.0, 110.0, 120.0, 130.0])
    tp = hist["period"].iloc[1:4].to_numpy()  # horas 1,2,3
    # modelo A perfecto; modelo B con error constante de 10
    preds = pd.concat(
        [
            _predictions(tp, "A", [110.0, 120.0, 130.0]),
            _predictions(tp, "B", [120.0, 130.0, 140.0]),
        ],
        ignore_index=True,
    )
    out = backtest.evaluate_predictions(preds, hist).set_index("model")
    assert out.loc["A", "mae"] == 0.0
    assert out.loc["B", "mae"] == 10.0
    assert out.loc["A", "n"] == 3
    assert out.loc["B", "n"] == 3


def test_evaluate_predictions_inner_join_drops_unmatched():
    hist = _history([100.0, 110.0])
    # predicción para una hora que no existe en el histórico -> se descarta
    future = pd.date_range("2025-02-01", periods=2, freq="h", tz="UTC")
    preds = _predictions(future, "A", [1.0, 2.0])
    out = backtest.evaluate_predictions(preds, hist)
    assert len(out) == 0
```

- [ ] **Step 2: Correr y verificar que falla**

Run: `.venv/Scripts/python.exe -m pytest tests/test_backtest.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'forecasting.backtest'`).

- [ ] **Step 3: Implementar `src/forecasting/backtest.py`** (parte 1; la parte 2 se añade en Task 5)
```python
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
```

- [ ] **Step 4: Correr y verificar que pasa**

Run: `.venv/Scripts/python.exe -m pytest tests/test_backtest.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**
```bash
git add src/forecasting/backtest.py tests/test_backtest.py
git commit -m "feat(backtest): forward-test (predicciones en vivo vs reales)"
```

---

## Task 5: `backtest` — rolling-origin histórico

**Files:**
- Modify: `src/forecasting/backtest.py`
- Test: `tests/test_backtest.py`

`rolling_origin_backtest` re-corre los modelos sobre el histórico viejo desde varios orígenes, evaluando contra `value_current` (revisado → optimista). Es el régimen que llena el dashboard desde el día 1.

- [ ] **Step 1: Añadir el test que falla a `tests/test_backtest.py`**
```python
import numpy as np
import pandas as pd
from forecasting import backtest


class PerfectModel:
    """Predice el valor real exacto leyendo de un lookup (para test determinista)."""
    name = "perfect"

    def __init__(self, lookup):
        self._lookup = lookup

    def predict(self, history, horizon):
        last = history["period"].iloc[-1]
        future = pd.date_range(last + pd.Timedelta(hours=1), periods=horizon, freq="h")
        return pd.DataFrame(
            {"period": future, "model": self.name,
             "prediction": [self._lookup[p] for p in future]}
        )


def test_rolling_origin_backtest_aggregates_over_origins():
    n = 24 * 10
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    values = np.arange(n, dtype=float)
    hist = pd.DataFrame(
        {"period": idx, "respondent": "PJM",
         "value_first_reported": values, "value_current": values,
         "first_seen_at": idx, "last_updated_at": idx}
    )
    lookup = dict(zip(idx, values))
    model = PerfectModel(lookup)
    origins = [idx[24 * 7], idx[24 * 8]]  # dos orígenes con futuro disponible
    out = backtest.rolling_origin_backtest(hist, [model], origins, horizon=24)
    row = out.set_index("model").loc["perfect"]
    assert row["mae"] == 0.0          # modelo perfecto
    assert row["n"] == 48             # 2 orígenes * 24 h
```

- [ ] **Step 2: Correr y verificar que falla**

Run: `.venv/Scripts/python.exe -m pytest tests/test_backtest.py::test_rolling_origin_backtest_aggregates_over_origins -v`
Expected: FAIL (`AttributeError: module 'forecasting.backtest' has no attribute 'rolling_origin_backtest'`).

- [ ] **Step 3: Añadir a `src/forecasting/backtest.py`**
```python
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
    series = pd.DataFrame(
        {"period": history["period"], "value": history[actual_col]}
    )
    actual_lookup = dict(zip(history["period"], history[actual_col]))

    records = []
    for origin in origins:
        sliced = series[series["period"] <= origin].reset_index(drop=True)
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
```

- [ ] **Step 4: Correr y verificar que pasa**

Run: `.venv/Scripts/python.exe -m pytest tests/test_backtest.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Correr TODA la suite (incl. slow), verificación end-to-end**

Run: `.venv/Scripts/python.exe -m pytest -q -m "slow or not slow"`
Expected: PASS (fundación + modelos + orquestación + backtest).

- [ ] **Step 6: Commit**
```bash
git add src/forecasting/backtest.py tests/test_backtest.py
git commit -m "feat(backtest): rolling-origin histórico (régimen optimista)"
```

---

## Self-Review

**1. Spec coverage (§4, §4.1, §5):**
- Corte as-of anti-leakage → `run_forecast` corta `period <= as_of` (Task 3). ✅
- Persistencia de predicciones con forecast_made_at → `predictions` (Task 2). ✅
- Loop robusto (un fallo no tumba todo) → try/except por modelo en `run_forecast` (Task 3) y en `rolling_origin_backtest` (Task 5). ✅
- Forward-test (régimen honesto) → `evaluate_predictions` (Task 4). ✅
- Backtest histórico rolling-origin (régimen optimista) → `rolling_origin_backtest` (Task 5). ✅
- Métricas MAE/sMAPE → `metrics` (Task 1), usadas por backtest. ✅
- Entrypoint del cron diario → `forecast_daily.__main__` (Task 3); el cron en sí es Plan 4. ✅
- *Fuera de este plan:* dashboard y GitHub Actions (Plan 4); usar `value_first_reported` además de `value_current` en evaluación es trivial vía el parámetro `actual_col`.

**2. Placeholder scan:** sin TBD/TODO; todo el código está completo. ✅

**3. Type/firma consistency:**
- `PREDICTION_COLUMNS` definido en `predictions` (Task 2), usado por `forecast_daily` (Task 3) y los tests de backtest (Tasks 4-5). ✅
- `metrics.mae`/`metrics.smape` (Task 1) usadas por `_metrics_by_model` en `backtest` (Tasks 4-5). ✅
- Modelos consumen `(period, value)` y devuelven `(period, model, prediction[, q10, q90])` — `run_forecast` y `rolling_origin_backtest` respetan ese contrato (Plan 2). ✅
- `forecast_and_store` usa `predictions.upsert_predictions` con la firma definida en Task 2. ✅
- `run_forecast` devuelve `(preds, failures)`; `forecast_and_store` lo consume; los tests usan esa forma. ✅

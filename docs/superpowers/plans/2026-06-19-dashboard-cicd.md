# Dashboard + CI/CD — Implementation Plan (Plan 4 de 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) o superpowers:executing-plans para las tareas de código (1-4). Las tareas de infraestructura (5-6) son guiadas/manuales con el usuario (requieren cuentas de GitHub/Streamlit). Steps con checkbox (`- [ ]`).

**Goal:** Volver el sistema "vivo": un dashboard público que muestra forecast actual, predicho-vs-real y error rodante; y GitHub Actions que ejecutan el loop diario (ingesta + forecast) y el job semanal (SARIMAX), commiteando el histórico versionado. Más la publicación del repo en GitHub y el enlace desde el Portfolio.

**Architecture:** Una capa de datos del dashboard (`views`, pura y testeable) que prepara los DataFrames; una app Streamlit delgada que solo renderiza. Un entrypoint de pipeline diario (`run_daily`) que ingiere datos frescos y corre el forecast. Workflows de GitHub Actions que orquestan en la nube (cron) y commitean `data/`. Reusa todo lo de Planes 1-3.

**Tech Stack:** Python, pandas, streamlit; GitHub Actions (cron + commit); Streamlit Community Cloud (deploy gratis).

---

## Contexto para quien ejecuta

- **Repo:** `C:\Users\asus\Documents\forecasting-demanda-electrica`. Rama de feature para las tareas de código. Venv: `.venv/Scripts/python.exe`.
- **Spec:** §6 (tecnología/despliegue) y §4 (loop). El dashboard es **secundario**; la prueba real del loop es el histórico versionado en git.
- **Ya existe (Planes 1-3):** `store` (histórico), `ingest.fetch_demand`, `bootstrap`, `models` (registry), `forecast_daily` (`forecast_and_store`, `__main__`), `predictions`, `backtest` (`evaluate_predictions`, `rolling_origin_backtest`), `metrics`.
- **Filosofía de testeo:** la lógica de preparación de datos del dashboard se testea (TDD). El render de Streamlit y los YAML de Actions no se unit-testean (se valida que parseen / smoke). La infra (repo, secrets, deploy) es manual guiada.

---

## File Structure

```
src/forecasting/
├── views.py          # NUEVO: prep de datos del dashboard (puro, testeable)
└── run_daily.py      # NUEVO: pipeline diario (ingesta reciente + forecast)
dashboard/
└── app.py            # NUEVO: Streamlit (render delgado)
.github/workflows/
├── tests.yml         # NUEVO: corre pytest en push/PR
├── daily.yml         # NUEVO: cron diario (ingesta + forecast + commit)
└── weekly.yml        # NUEVO: cron semanal (SARIMAX + commit)
tests/
├── test_views.py
└── test_run_daily.py
requirements.txt      # añade streamlit, pyyaml
```

---

## Task 1: `views` — capa de datos del dashboard

**Files:** Create `src/forecasting/views.py`, `tests/test_views.py`

Funciones puras que preparan lo que el dashboard grafica. Reciben los DataFrames de `predictions`/`store` y devuelven tablas listas para graficar.

- [ ] **Step 1: Escribir el test que falla `tests/test_views.py`**
```python
import numpy as np
import pandas as pd
from forecasting import views
from forecasting.predictions import PREDICTION_COLUMNS


def _history(values, start="2025-01-01"):
    n = len(values)
    idx = pd.date_range(start, periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {"period": idx, "respondent": "PJM",
         "value_first_reported": values, "value_current": values,
         "first_seen_at": idx, "last_updated_at": idx}
    )


def _preds(made_at, model, target_periods, preds):
    return pd.DataFrame(
        {"forecast_made_at": pd.Timestamp(made_at, tz="UTC"), "model": model,
         "target_period": target_periods, "horizon": range(1, len(preds) + 1),
         "prediction": preds, "q10": np.nan, "q90": np.nan}
    )[PREDICTION_COLUMNS]


def test_latest_forecast_keeps_only_most_recent_run_per_model():
    tp1 = pd.date_range("2025-01-02", periods=2, freq="h", tz="UTC")
    tp2 = pd.date_range("2025-01-03", periods=2, freq="h", tz="UTC")
    preds = pd.concat([
        _preds("2025-01-02T00", "naive", tp1, [1.0, 2.0]),
        _preds("2025-01-03T00", "naive", tp2, [3.0, 4.0]),  # más reciente
    ], ignore_index=True)
    out = views.latest_forecast(preds)
    assert (out["forecast_made_at"] == pd.Timestamp("2025-01-03T00", tz="UTC")).all()
    assert len(out) == 2


def test_predicted_vs_actual_joins_on_target():
    hist = _history([100.0, 110.0, 120.0])
    tp = hist["period"].iloc[1:3].to_numpy()
    preds = _preds("2025-01-01T00", "naive", tp, [111.0, 119.0])
    out = views.predicted_vs_actual(preds, hist)
    assert set(["target_period", "model", "prediction", "actual"]).issubset(out.columns)
    assert len(out) == 2
    assert out.sort_values("target_period")["actual"].tolist() == [110.0, 120.0]


def test_rolling_error_per_model():
    # 4 puntos, abs_error constante = 10 -> rolling mae (window=2) = 10
    hist = _history([100.0, 100.0, 100.0, 100.0])
    tp = hist["period"].to_numpy()
    preds = _preds("2025-01-01T00", "naive", tp, [110.0, 110.0, 110.0, 110.0])
    pva = views.predicted_vs_actual(preds, hist)
    out = views.rolling_error(pva, window=2)
    assert set(["target_period", "model", "rolling_mae"]).issubset(out.columns)
    # tras la primera ventana completa, rolling_mae == 10
    assert out["rolling_mae"].dropna().iloc[-1] == 10.0
```

- [ ] **Step 2:** Run `.venv/Scripts/python.exe -m pytest tests/test_views.py -v` → FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implementar `src/forecasting/views.py`**
```python
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
```

- [ ] **Step 4:** Run → PASS (3 tests).

- [ ] **Step 5: Commit**
```bash
git add src/forecasting/views.py tests/test_views.py
git commit -m "feat(views): capa de datos del dashboard (latest_forecast, predicted_vs_actual, rolling_error)"
```

---

## Task 2: `run_daily` — pipeline diario (ingesta reciente + forecast)

**Files:** Create `src/forecasting/run_daily.py`, `tests/test_run_daily.py`

El entrypoint que el cron diario invoca: trae los últimos días de la EIA, los funde al histórico, y corre el forecast. `fetch_fn` inyectable para testear sin red.

- [ ] **Step 1: Escribir el test que falla `tests/test_run_daily.py`**
```python
import numpy as np
import pandas as pd
from forecasting import run_daily, store, predictions


def _seed_history(path, n=24 * 8):
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    obs = pd.DataFrame({"period": idx, "respondent": "PJM", "value": np.arange(n, dtype=float)})
    store.upsert_demand(path, obs, now=pd.Timestamp("2025-01-10", tz="UTC"))


def test_run_daily_ingests_then_forecasts(tmp_path):
    hist_path = tmp_path / "demand_history.parquet"
    preds_path = tmp_path / "predictions.parquet"
    _seed_history(hist_path)

    # nuevos datos "frescos": 24 h más, contiguas
    def fake_fetch(respondent, start, end, api_key):
        new_idx = pd.date_range("2025-01-09", periods=24, freq="h", tz="UTC")
        return pd.DataFrame({"period": new_idx, "respondent": respondent,
                             "value": np.arange(100, 124, dtype=float)})

    class DummyModel:
        name = "dummy"
        def predict(self, history, horizon):
            fut = pd.date_range(history["period"].iloc[-1] + pd.Timedelta(hours=1),
                                periods=horizon, freq="h")
            return pd.DataFrame({"period": fut, "model": self.name, "prediction": 0.0})

    summary = run_daily.run_daily(
        history_path=hist_path, predictions_path=preds_path,
        models=[DummyModel()], api_key="FAKE",
        now=pd.Timestamp("2025-01-09T23:00", tz="UTC"),
        recent_days=2, fetch_fn=fake_fetch, horizon=24,
    )
    # el histórico creció con los datos frescos
    hist = store.read_demand_history(hist_path)
    assert hist["period"].max() == pd.Timestamp("2025-01-09T23:00", tz="UTC")
    # se guardaron predicciones
    saved = predictions.read_predictions(preds_path)
    assert len(saved) == 24
    assert summary["ingested"]["inserted"] >= 1
    assert summary["forecast"]["failures"] == []
```

- [ ] **Step 2:** Run → FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implementar `src/forecasting/run_daily.py`**
```python
"""Pipeline diario: ingesta de datos frescos de la EIA + forecast. Entrypoint del cron."""
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from forecasting import forecast_daily, ingest, store
from forecasting.config import DEMAND_HISTORY_PATH, PREDICTIONS_PATH, REGION
from forecasting.models import daily_models


def run_daily(
    history_path: Path,
    predictions_path: Path,
    models,
    api_key: str,
    now: pd.Timestamp,
    recent_days: int = 3,
    fetch_fn=ingest.fetch_demand,
    horizon: int = 24,
    respondent: str = REGION,
) -> dict:
    """Trae los últimos `recent_days` días, funde al histórico y corre el forecast."""
    start = (now - pd.Timedelta(days=recent_days)).strftime("%Y-%m-%dT%H")
    end = now.strftime("%Y-%m-%dT%H")
    fresh = fetch_fn(respondent=respondent, start=start, end=end, api_key=api_key)
    ingested = store.upsert_demand(history_path, fresh, now=now)

    history = store.read_demand_history(history_path)
    as_of = history["period"].max()
    forecast = forecast_daily.forecast_and_store(
        history, models, as_of=as_of, horizon=horizon, predictions_path=predictions_path
    )
    return {"ingested": ingested, "forecast": forecast}


if __name__ == "__main__":
    load_dotenv()
    api_key = os.environ["EIA_API_KEY"]
    summary = run_daily(
        history_path=DEMAND_HISTORY_PATH, predictions_path=PREDICTIONS_PATH,
        models=daily_models(), api_key=api_key, now=pd.Timestamp.now(tz="UTC"),
    )
    print(f"run_daily: {summary}")
```

- [ ] **Step 4:** Run → PASS. Then whole suite `.venv/Scripts/python.exe -m pytest -q` → green.

- [ ] **Step 5: Commit**
```bash
git add src/forecasting/run_daily.py tests/test_run_daily.py
git commit -m "feat(run_daily): pipeline diario (ingesta reciente + forecast)"
```

---

## Task 3: Dashboard Streamlit

**Files:** Create `dashboard/app.py`; modify `requirements.txt`; Test: a syntax-check in `tests/test_views.py` o `tests/test_dashboard.py`

App delgada: usa `views` + `store`/`predictions`/`backtest` para renderizar. No se unit-testea el render; se valida que el archivo parsee (catches errores de sintaxis).

- [ ] **Step 1: Añadir deps** en `requirements.txt`: `streamlit>=1.36` y `pyyaml>=6.0`. Instalar: `.venv/Scripts/python.exe -m pip install "streamlit>=1.36" "pyyaml>=6.0"`.

- [ ] **Step 2: Escribir test de smoke `tests/test_dashboard.py`** (parsea el archivo; falla si hay error de sintaxis)
```python
import ast
from pathlib import Path


def test_dashboard_app_parses():
    src = Path("dashboard/app.py").read_text(encoding="utf-8")
    ast.parse(src)  # lanza SyntaxError si el archivo está malformado
```

- [ ] **Step 3:** Run `.venv/Scripts/python.exe -m pytest tests/test_dashboard.py -v` → FAIL (`FileNotFoundError`: aún no existe `dashboard/app.py`).

- [ ] **Step 4: Crear `dashboard/app.py`**
```python
"""Dashboard del servicio de forecasting de demanda eléctrica (Streamlit)."""
import pandas as pd
import streamlit as st

from forecasting import backtest, predictions, store, views
from forecasting.config import DEMAND_HISTORY_PATH, PREDICTIONS_PATH, REGION

st.set_page_config(page_title="Forecasting demanda eléctrica", layout="wide")
st.title(f"⚡ Forecasting de demanda eléctrica — {REGION}")
st.caption(
    "Sistema vivo: cada día ingiere datos de la EIA, pronostica 24 h y verifica "
    "la predicción del día anterior. El histórico versionado en git es la prueba del loop."
)

history = store.read_demand_history(DEMAND_HISTORY_PATH)
preds = predictions.read_predictions(PREDICTIONS_PATH)

if history.empty or preds.empty:
    st.warning("Aún no hay datos suficientes. Corre el backfill y el primer forecast.")
    st.stop()

# --- Forecast actual (próximas 24 h) ---
st.header("Forecast actual (próximas 24 h)")
latest = views.latest_forecast(preds)
chart = latest.pivot_table(index="target_period", columns="model", values="prediction")
st.line_chart(chart)

# --- Predicho vs real ---
st.header("Predicho vs real (forward-test, point-in-time)")
pva = views.predicted_vs_actual(preds, history)
if not pva.empty:
    recent = pva.sort_values("target_period").tail(24 * 14)
    pivot = recent.pivot_table(index="target_period", columns="model", values="prediction")
    pivot["real"] = recent.groupby("target_period")["actual"].first()
    st.line_chart(pivot)

# --- Error rodante por modelo ---
st.header("Error rodante (MAE, ventana 7 días)")
roll = views.rolling_error(pva, window=24 * 7)
if not roll["rolling_mae"].dropna().empty:
    roll_pivot = roll.pivot_table(index="target_period", columns="model", values="rolling_mae")
    st.line_chart(roll_pivot)

# --- Métricas (dos regímenes) ---
st.header("Métricas por modelo")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Forward-test (honesto, point-in-time)")
    st.dataframe(backtest.evaluate_predictions(preds, history))
with col2:
    st.subheader("Backtest histórico (datos revisados, optimista)")
    st.caption("Ligeramente optimista: usa datos ya revisados por la EIA.")
    st.dataframe(backtest.evaluate_predictions(preds, history, actual_col="value_first_reported"))
```

- [ ] **Step 5:** Run `.venv/Scripts/python.exe -m pytest tests/test_dashboard.py -v` → PASS. (Opcional: lanzar el dashboard localmente con datos del smoke test del Plan 1: `.venv/Scripts/streamlit run dashboard/app.py` y verlo en el navegador.)

- [ ] **Step 6: Commit**
```bash
git add dashboard/app.py tests/test_dashboard.py requirements.txt
git commit -m "feat(dashboard): app Streamlit con forecast, predicho-vs-real, error rodante y métricas"
```

---

## Task 4: Workflows de GitHub Actions

**Files:** Create `.github/workflows/{tests,daily,weekly}.yml`; Test: `tests/test_workflows.py` (valida que los YAML parseen)

> Concepto (para aprender): un *workflow* de GitHub Actions es un archivo YAML que describe pasos que GitHub ejecuta en un runner (una VM efímera) ante un evento (`push`, `pull_request`, o `schedule`/cron). El cron usa sintaxis cron UTC. Los secrets (como la API key) se inyectan vía `${{ secrets.NOMBRE }}` y nunca se hardcodean.

- [ ] **Step 1: Escribir test que falla `tests/test_workflows.py`**
```python
from pathlib import Path
import yaml


def test_workflows_parse():
    wf_dir = Path(".github/workflows")
    files = list(wf_dir.glob("*.yml"))
    assert {f.name for f in files} >= {"tests.yml", "daily.yml", "weekly.yml"}
    for f in files:
        data = yaml.safe_load(f.read_text(encoding="utf-8"))
        assert "jobs" in data  # estructura mínima válida
```

- [ ] **Step 2:** Run → FAIL (no existen los workflows).

- [ ] **Step 3: Crear `.github/workflows/tests.yml`**
```yaml
name: tests
on:
  push:
  pull_request:
jobs:
  pytest:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Instalar dependencias
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install -e .
      - name: Correr tests (rápidos; se omiten los 'slow' que descargan modelos pesados)
        run: pytest -q -m "not slow"
```

- [ ] **Step 4: Crear `.github/workflows/daily.yml`**
```yaml
name: daily-forecast
on:
  schedule:
    - cron: "0 12 * * *"   # 12:00 UTC diario
  workflow_dispatch:        # permite correrlo a mano desde la UI de GitHub
jobs:
  forecast:
    runs-on: ubuntu-latest
    permissions:
      contents: write       # para commitear el histórico actualizado
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Instalar dependencias
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install -e .
      - name: Ingesta + forecast
        env:
          EIA_API_KEY: ${{ secrets.EIA_API_KEY }}
        run: python -m forecasting.run_daily
      - name: Commitear datos actualizados
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add data/
          git diff --staged --quiet || git commit -m "data: actualiza histórico y forecast ($(date -u +%Y-%m-%d))"
          git push
```

- [ ] **Step 5: Crear `.github/workflows/weekly.yml`** (SARIMAX aislado; corre el backtest histórico y lo deja como artefacto del log)
```yaml
name: weekly-sarimax
on:
  schedule:
    - cron: "0 6 * * 1"    # lunes 06:00 UTC
  workflow_dispatch:
jobs:
  sarimax:
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Instalar dependencias
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install -e .
      - name: Forecast con SARIMAX (modelos semanales)
        env:
          EIA_API_KEY: ${{ secrets.EIA_API_KEY }}
        run: |
          python -c "from forecasting import store, forecast_daily; from forecasting.models import weekly_models; from forecasting.config import DEMAND_HISTORY_PATH, PREDICTIONS_PATH; h = store.read_demand_history(DEMAND_HISTORY_PATH); print(forecast_daily.forecast_and_store(h, weekly_models(), as_of=h['period'].max(), predictions_path=PREDICTIONS_PATH))"
      - name: Commitear predicciones
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add data/
          git diff --staged --quiet || git commit -m "data: forecast semanal SARIMAX ($(date -u +%Y-%m-%d))"
          git push
```

- [ ] **Step 6:** Run `.venv/Scripts/python.exe -m pytest tests/test_workflows.py -v` → PASS. Correr la suite completa rápida `.venv/Scripts/python.exe -m pytest -q -m "not slow"` → green.

- [ ] **Step 7: Commit**
```bash
git add .github/workflows/tests.yml .github/workflows/daily.yml .github/workflows/weekly.yml tests/test_workflows.py
git commit -m "ci: workflows de tests, forecast diario y SARIMAX semanal"
```

---

## Task 5: Publicar el repo en GitHub (GUIADA — con el usuario)

> No es subagente: requiere la cuenta de GitHub del usuario y la API key como secret. Se hace con `gh` (GitHub CLI) explicando cada paso.

- [ ] **Step 1: Verificar `gh` y autenticación** — `gh auth status`. Si no está autenticado, guiar `gh auth login`.
- [ ] **Step 2: Crear el repo y hacer push** — desde la raíz del repo:
  ```bash
  gh repo create forecasting-demanda-electrica --public --source=. --remote=origin --description "Servicio vivo de forecasting de demanda eléctrica (EIA) — ML Engineer" --push
  ```
- [ ] **Step 3: Cargar el secret** (la API key NO se pega en el chat; el usuario la pone):
  ```bash
  gh secret set EIA_API_KEY
  ```
  (pide el valor de forma interactiva, sin que quede en el historial).
- [ ] **Step 4: Verificar que el workflow de tests corre** en GitHub (pestaña Actions) tras el push.
- [ ] **Step 5: (Opcional) Backfill inicial multi-año** localmente y commitear el histórico (la prueba versionada): `python -m forecasting.bootstrap 2021-01-01T00 <hoy>` → `git add data/ && git commit && git push`.

---

## Task 6: Deploy del dashboard + enlace en el Portfolio (GUIADA)

- [ ] **Step 1: Deploy en Streamlit Community Cloud** (gratis): conectar la cuenta de GitHub en https://share.streamlit.io, elegir el repo, archivo `dashboard/app.py`, branch `main`. (Streamlit Cloud instala `requirements.txt` automáticamente; añadir `pip install -e .` no es directo allí — si falla el import de `forecasting`, alternativa: añadir un `packages.txt`/ajustar `sys.path`, o mover el dashboard a usar el paquete instalado. Resolver en el momento según el error.)
- [ ] **Step 2: Añadir la URL pública** al README del repo (badge/enlace al dashboard) y al README del Portfolio (reemplazar "link de GitHub pendiente" por la URL real del repo + dashboard).
- [ ] **Step 3: Commit** del README actualizado en ambos repos.

---

## Self-Review

**1. Spec coverage (§4, §6):**
- Dashboard con forecast actual, predicho-vs-real, error rodante, métricas de dos regímenes → Task 1 (views) + Task 3 (app). ✅
- Ingesta diaria automática + forecast → Task 2 (run_daily) + Task 4 (daily.yml). ✅
- SARIMAX en job semanal aislado → Task 4 (weekly.yml) + `weekly_models()`. ✅
- Histórico versionado commiteado por el cron (prueba del loop vivo) → daily.yml/weekly.yml commitean `data/`. ✅
- Tests en CI → tests.yml. ✅
- Repo dedicado en GitHub + link desde Portfolio → Tasks 5-6. ✅
- Deploy free-tier → Task 6 (Streamlit Cloud). ✅

**2. Placeholder scan:** sin TBD. Las Tasks 5-6 son pasos de infra con comandos exactos; el único punto abierto declarado es el ajuste de import de `forecasting` en Streamlit Cloud, que se resuelve según el error real (no es un placeholder de código).

**3. Type/firma consistency:**
- `views` consume `predictions` (PREDICTION_COLUMNS) e `history` (store schema) y reusa el mismo join que `backtest`. ✅
- `run_daily` usa `ingest.fetch_demand(respondent,start,end,api_key)`, `store.upsert_demand(path,obs,now)`, `forecast_daily.forecast_and_store(history,models,as_of,horizon,predictions_path)` — firmas de Planes 1-3. ✅
- `app.py` usa `views.latest_forecast/predicted_vs_actual/rolling_error` (Task 1) y `backtest.evaluate_predictions` (Plan 3). ✅
- Workflows invocan `python -m forecasting.run_daily` (Task 2, tiene `__main__`) y `weekly_models()` (Plan 2). ✅

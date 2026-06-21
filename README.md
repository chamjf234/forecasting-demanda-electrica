# Servicio vivo de forecasting de demanda eléctrica

Pronóstico de **demanda eléctrica horaria** (región PJM, EE.UU.) servido como un
sistema que **se mantiene y monitorea solo**. Cada día un cron de GitHub Actions
ingiere datos frescos de la [EIA](https://www.eia.gov/opendata/), pronostica las
próximas 24 h con varios modelos y verifica la predicción del día anterior contra
el valor real — con un backtest a prueba de *data leakage*.

El foco es de **ML Engineering**: lo que se luce no es el modelo, sino que el
modelo vive dentro de un sistema automatizado, versionado y monitoreado. El
historial de commits de `data/` (generado por el cron) es la prueba de que el
loop corre día tras día.

## Cómo funciona

```
  EIA API (demanda horaria)
        │  ingesta diaria (GitHub Actions cron)
        ▼
  Histórico versionado en el repo (Parquet)
        │
        ├──► Forecast 24 h (modelos) ──► Predicciones guardadas (con corte as-of)
        │                                      │
        ▼                                      ▼
  Backtesting continuo  ◄──────────  Valor real del día siguiente
        │
        ▼
  Métricas de error + error rodante  ──►  Dashboard (Streamlit)
```

### Anti-leakage (lo más importante)
- **Corte *as-of*:** al pronosticar, el modelo solo ve datos `period <= as_of`. Nunca el futuro.
- **Revisiones de la EIA:** por cada hora se guardan `value_first_reported` y
  `value_current`, así las revisiones no corrompen métricas pasadas.
- **Dos regímenes de evaluación:** *forward-test* (predicciones en vivo vs reales,
  honesto) y *backtest histórico* (datos revisados, ligeramente optimista),
  etiquetados como tales.

## Modelos

| Nivel | Modelo | Qué demuestra |
|---|---|---|
| 0 | Seasonal naive | La vara mínima honesta |
| 1 | SARIMAX + Fourier (24h/168h) | Econometría clásica rigurosa (job semanal aislado) |
| 2 | LightGBM + calendario y lags | ML tabular de industria |
| 3 | Chronos-Bolt (Amazon) | Foundation model de series, zero-shot probabilístico |

## Setup (desarrollo)

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1   |  Unix: source .venv/bin/activate

# Dashboard / base:
pip install -r requirements.txt
# Modelos (torch, lightgbm, statsforecast, chronos) — para correr forecasts/tests:
pip install -r requirements-models.txt
pip install -e .          # instala el paquete `forecasting`

cp .env.example .env      # pon tu EIA_API_KEY (https://www.eia.gov/opendata/register.php)
```

### Uso
```bash
python -m forecasting.bootstrap 2021-01-01T00 2026-06-19T23   # backfill del histórico (una vez)
python -m forecasting.run_daily                                # ingesta reciente + forecast
streamlit run dashboard/app.py                                 # dashboard local
pytest -q                                                      # tests (omite los 'slow')
pytest -q -m "slow or not slow"                                # incluye SARIMAX y Chronos
```

## Arquitectura del código
- `src/forecasting/` — `store` (histórico + snapshot), `ingest` (EIA), `bootstrap`,
  `series` (regularización horaria), `models/` (4 modelos + registry),
  `forecast_daily` (orquestación as-of), `predictions`, `backtest`, `metrics`, `views`.
- `dashboard/app.py` — Streamlit (solo lee los Parquet versionados; sin dependencias pesadas).
- `.github/workflows/` — `tests` (CI), `daily` (ingesta + forecast), `weekly` (SARIMAX).
- `data/` — histórico y predicciones versionados (la prueba del loop vivo).

Diseño detallado y planes de implementación en `docs/superpowers/`.

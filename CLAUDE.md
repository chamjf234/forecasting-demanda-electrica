# Forecasting de demanda eléctrica — Contexto del proyecto

## Qué es
Servicio vivo que pronostica demanda eléctrica horaria (región PJM, datos EIA).
Cada día ingiere datos frescos, predice 24h con varios modelos y verifica la
predicción del día anterior contra el valor real. Enfoque ML Engineer: el loop
de MLOps y el backtest anti-leakage son el centro, no el modelo.

## Diseño
Ver `docs/superpowers/specs/2026-06-15-forecasting-demanda-electrica-design.md`.

## Disciplina de snapshot
La EIA revisa cifras hacia atrás. Por cada (hora, región) guardamos
`value_first_reported` y `value_current`. Nunca se predice con datos del futuro.

## Estructura
- `src/forecasting/` — código (store, ingest, bootstrap, models, ...)
- `data/` — histórico versionado en Parquet (SÍ se commitea; es la prueba del loop)
- `tests/` — pytest

## Convención
Todo en UTC. API key de EIA en `.env` (no se commitea).

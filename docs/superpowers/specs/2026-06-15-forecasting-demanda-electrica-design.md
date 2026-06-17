# Diseño — Servicio vivo de forecasting de demanda eléctrica

**Fecha:** 2026-06-15
**Autor:** Juan Fernando Rojas
**Estado:** Aprobado (diseño), pendiente plan de implementación

---

## 1. Resumen ejecutivo

Sistema de pronóstico de **demanda eléctrica horaria** que no es un notebook que
corre una vez, sino un **servicio vivo** que se mantiene y monitorea solo. Cada
día (1) ingiere datos frescos de la EIA, (2) pronostica las próximas 24 horas con
varios modelos, (3) compara la predicción del día anterior contra el valor real, y
(4) actualiza métricas de precisión en un dashboard público.

El proyecto está orientado a mostrar habilidades de **ML Engineer**: lo que se luce
no es el modelo, sino que el modelo **vive dentro de un sistema automatizado,
versionado y monitoreado**, con un backtest honesto y a prueba de *data leakage*.

**Diferenciador clave:** el dominio (forecasting de demanda eléctrica) está
saturado en portafolios. Lo que distingue a este proyecto es el **loop vivo de
MLOps** y la **disciplina anti-leakage del backtest**. El esfuerzo se concentra
ahí, no en tunear modelos.

---

## 2. Objetivo y criterios de éxito

**Objetivo:** demostrar el ciclo completo notebook → producción de un ML Engineer,
con un sistema que un reclutador pueda inspeccionar y ver que funcionó de forma
sostenida.

**Criterios de éxito (v1):**

- El histórico de predicciones y valores reales está **versionado en el repo** y el
  `git log` evidencia que el loop diario corrió día tras día (esto es la prueba del
  sistema vivo, independiente de que la app esté despierta).
- El backtest es **honesto**: sin leakage, con disciplina de snapshot *as-of* y
  manejo correcto de las revisiones de datos de la EIA. Se distinguen
  explícitamente dos regímenes de evaluación (ver §4.1): backtest histórico
  (datos revisados, ligeramente optimista) y forward-test en vivo (point-in-time).
- Hay al menos 3 modelos comparados de forma justa con la misma metodología de
  evaluación.
- Existe un dashboard que muestra forecast actual, histórico predicho-vs-real y
  error rodante por modelo.
- El pipeline diario es **robusto**: un fallo de un modelo no tumba todo el loop.

**No es criterio de éxito:** que el modelo fundacional (Chronos) gane. La pregunta
"¿le gana un modelo de 2024 a un LightGBM bien hecho y a un baseline ingenuo,
medido honestamente?" es interesante con cualquier resultado y demuestra criterio.

---

## 3. Datos

**Fuente:** API oficial de la **EIA (U.S. Energy Information Administration)**,
gratuita con API key. Endpoint de demanda eléctrica horaria por *balancing
authority*.

**Región (v1):** **PJM** (interconexión del este de EE.UU.). Grande, estable y
fuertemente estacional → ideal para un v1 robusto.

**Granularidad:** horaria.

**Estacionalidades presentes:** diaria (período 24), semanal (período 168) y anual.
Esto es central para el diseño de modelos (ver §5).

**Retos conocidos de la fuente:**

- La EIA publica con algunas horas de retraso → el sistema vive con datos casi en
  tiempo real, no instantáneos.
- La EIA **revisa hacia atrás** sus cifras de demanda → el "valor real" de un día
  cambia con el tiempo. Esto se maneja explícitamente en §4.

---

## 4. Arquitectura y flujo de datos

```
  EIA API (demanda horaria)
        │  ingesta diaria (GitHub Actions cron)
        ▼
  Histórico versionado en el repo (Parquet/DuckDB commiteado)
        │
        ├──► Forecast diario (modelos) ──► Predicciones guardadas (con as-of cutoff)
        │                                        │
        ▼                                        ▼
  Backtesting continuo  ◄────────────  Valor real del día siguiente
        │                              (se guarda primer-reportado Y actual)
        ▼
  Métricas de error + error rodante  ──►  Dashboard público
```

### 4.1 Arranque (bootstrap) y dos regímenes de evaluación

El sistema no puede arrancar en frío: LightGBM y SARIMAX necesitan datos para
entrenar, y el dashboard debe tener contenido desde el día 1. Por eso hay un
**script de bootstrap** que se corre una sola vez al inicializar:

- **Backfill** de varios años de demanda horaria de la EIA → da (a) datos de
  entrenamiento para los modelos y (b) material para un backtest histórico
  inmediato.

A partir de ahí coexisten **dos regímenes de evaluación distintos**, etiquetados
explícitamente en el dashboard y el README (esta honestidad metodológica es parte
de lo que se quiere lucir):

1. **Backtest histórico (datos revisados):** se evalúan los modelos sobre el
   histórico con validación *rolling-origin*. Es ligeramente **optimista**: los
   datos viejos ya están revisados, así que el modelo "ve" cifras más limpias de
   las que habría tenido en tiempo real. Llena el dashboard desde el inicio.
2. **Forward-test en vivo (point-in-time):** el loop diario captura snapshots
   *as-of* verdaderos (ver abajo). Es el régimen **honesto de verdad**, pero
   empieza vacío y crece día a día. Con el tiempo es la evidencia más fuerte.

### Disciplina de snapshot (anti-leakage) — núcleo del proyecto

Esta es la parte que separa un backtest honesto de uno con leakage invisible:

1. **Al predecir:** se registra la predicción junto con el **corte de datos
   *as-of*** ese momento (qué datos tenía disponibles el modelo cuando predijo).
   Nunca se predice con información del futuro.
2. **Al llegar el valor real:** se guardan **dos versiones** del valor real:
   - `valor_primer_reportado`: el primer dato que publicó la EIA.
   - `valor_actual`: el valor tras posibles revisiones.
   Así las revisiones de la EIA no corrompen retroactivamente las métricas, y se
   puede medir el error contra ambas versiones.

### Componentes (unidades con responsabilidad única)

- **`ingest`** — trae datos de la EIA y los añade al histórico versionado.
  Idempotente (re-correrlo no duplica filas).
- **`store`** — capa de lectura/escritura del histórico y de las predicciones
  (Parquet/DuckDB). Maneja la disciplina de snapshot.
- **`models`** — cada modelo expone la misma interfaz: `fit(historia) -> modelo`,
  `predict(modelo, horizonte) -> forecast`. Esto permite compararlos de forma justa
  y agregar/quitar modelos sin tocar el resto.
- **`forecast_daily`** — orquesta el loop diario: lee historia → corre modelos
  robustos → guarda predicciones. Con manejo de errores por modelo.
- **`backtest`** — une predicciones pasadas con valores reales y calcula métricas.
- **`dashboard`** — UI que lee el histórico y las métricas.

---

## 5. Modelado

Se comparan **cuatro niveles** de sofisticación con la misma metodología de
evaluación. La narrativa: del más ingenuo al más actual, ¿cuál gana medido
honestamente?

| Nivel | Modelo | Qué demuestra | ¿En el loop diario? |
|---|---|---|---|
| 0 | Seasonal naive | La vara mínima honesta | Sí (instantáneo) |
| 1 | SARIMAX + Fourier + exógenas | Econometría clásica rigurosa | **No** — job semanal aislado |
| 2 | LightGBM + features de calendario | ML tabular de industria | Sí (segundos) |
| 3 | Chronos-Bolt (foundation model) | Transfer learning / lo "actual" | Sí (zero-shot, instantáneo) |

### Nivel 0 — Seasonal naive
"Mañana a las 3pm será como la semana pasada a las 3pm." Difícil de superar en
series muy estacionales. Se incluye deliberadamente para evitar el error clásico de
presumir un modelo complejo sin demostrar que le gana a lo trivial.

### Nivel 1 — SARIMAX con términos de Fourier y exógenas
- **Por qué Fourier y no SARIMA puro:** SARIMA clásico maneja una sola
  estacionalidad; la serie tiene tres. Forzar `m=168` es inviable (no converge, es
  lentísimo). La forma canónica (*dynamic harmonic regression*, Hyndman) es modelar
  las estacionalidades múltiples con **términos de Fourier** como regresores, y
  dejar que el ARIMA modele solo la autocorrelación de corto plazo.
- **La parte X (exógenas) en v1:** calendario (festivos), que es determinista.
  (Temperatura → roadmap, ver §7.)
- **Librería:** `statsforecast` (Nixtla), con `AutoARIMA` optimizado en C, soporta
  exógenas y Fourier, pensado para correr desatendido. (`statsmodels` queda como
  opción para una exploración interpretativa de diagnósticos: ACF/PACF, residuos.)
- **Aislamiento:** corre en un **job semanal separado**, con manejo de errores y
  *fallback*, para que si no converge o se pasa de tiempo **no tumbe el forecast
  diario**. Sigue contando en el benchmark.

### Nivel 2 — LightGBM
Features temporales de calendario (hora, día de semana, festivo, lags, medias
móviles). Es lo que se usa de verdad en industria para forecasting tabular-izado.
Rápido, corre en CPU, interpretable. Probable favorito a ganar en una sola serie
limpia.

### Nivel 3 — Chronos-Bolt (Amazon)
- Transformer preentrenado en millones de series; predice **zero-shot**.
- **Por qué Chronos y no TimesFM:** se instala con `pip`, corre en CPU para
  horizontes cortos, muy bien documentado en HuggingFace, y la versión "Bolt" es
  rápida. TimesFM es comparable pero con setup más pesado.
- **Paralelo con el proyecto de visión del portafolio:** igual que se usó
  EfficientNet preentrenado para embeddings de imágenes, aquí se usa un transformer
  preentrenado para forecasting. Misma filosofía de *transfer learning*, dominio
  nuevo.
- Da forecasts **probabilísticos** (cuantiles): en v1 se muestran las bandas de
  incertidumbre en el dashboard, sin el estudio formal de calibración (roadmap).

### Métricas de evaluación
MAE y MAPE (o sMAPE) por modelo y por horizonte, calculadas sobre el histórico de
backtest. Error rodante (ventana móvil) para visualizar degradación en el tiempo.

---

## 6. Tecnología y despliegue (free-tier-first)

Arquitectado como producción, con cero costo recurrente. La misma arquitectura se
migra a cloud si más adelante hay créditos.

- **Repo:** **dedicado y standalone** (no subcarpeta del monorepo Portfolio). Los
  commits diarios del histórico no contaminan el portafolio y son la prueba viva
  del loop; secrets/CI/CD/deploy quedan acotados a este repo. El Portfolio solo lo
  enlaza desde su README.
- **Lenguaje:** Python.
- **Orquestación / cron:** GitHub Actions (cron gratis).
  - *Riesgo conocido:* GitHub deshabilita crons en repos sin actividad por 60 días.
    Mitigación: el commit diario del histórico mantiene el repo activo.
- **Almacenamiento del histórico:** Parquet o DuckDB **commiteado al repo** (es la
  prueba versionada del loop). Decisión final Parquet vs DuckDB se toma en el plan.
- **Modelos:** `statsforecast`, `lightgbm`, `chronos-forecasting` (HuggingFace),
  `pandas`, `numpy`.
- **Dashboard:** Streamlit (Streamlit Cloud o HuggingFace Spaces, free tier).
  - *Riesgo conocido:* el free tier duerme la app por inactividad (cold start).
    Mitigación: el dashboard es **secundario**; la prueba real es el histórico
    versionado. El cold start es aceptable para un demo de portafolio.

---

## 7. Fuera de alcance (roadmap explícito en el README)

Se documentan en el README para señalar visión sin riesgo de dejar el v1 a medias:

- **Exógenas de clima (NOAA):** segundo punto de falla; requiere el *pronóstico* de
  temperatura (no el observado) para evitar leakage, otra API y otro cron.
- **Estudio formal de calibración / conformal prediction:** en v1 se muestran las
  bandas de Chronos, pero sin el deep-dive de calibración.
- **Múltiples regiones.**
- **Detector de drift formal:** en v1 basta el error rodante, que da el 90% del
  valor.

---

## 8. Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| Sistema vivo que se rompe parece abandonado | El histórico versionado en git es la prueba; la app es secundaria |
| Cron de GitHub Actions se deshabilita a los 60 días | El commit diario mantiene el repo activo |
| App duerme en free tier (cold start) | Aceptable para demo; la prueba real es el histórico |
| EIA revisa datos → métricas corruptas | Guardar primer-reportado Y actual; predecir con corte as-of |
| SARIMAX no converge / se pasa de tiempo en cron | Aislado en job semanal con error handling y fallback |
| Chronos pierde contra LightGBM | Narrativa de criterio: "lo nuevo no siempre gana"; resultado interesante igual |
| Scope creep | v1 recortado; resto en roadmap explícito |

---

## 9. Estructura propuesta del proyecto

```
forecasting-demanda-electrica/
├── README.md                 # Narrativa, resultados, roadmap
├── CLAUDE.md                 # Contexto del subproyecto
├── docs/superpowers/specs/   # Este diseño
├── src/
│   ├── bootstrap.py          # Backfill inicial (una sola vez): histórico EIA
│   ├── ingest.py             # Ingesta EIA → histórico
│   ├── store.py              # Lectura/escritura + disciplina de snapshot
│   ├── models/               # Un módulo por modelo, interfaz común
│   ├── forecast_daily.py     # Orquesta el loop diario
│   └── backtest.py           # Une predicciones con reales, métricas
├── dashboard/                # App Streamlit
├── data/                     # Histórico versionado (Parquet/DuckDB)
└── .github/workflows/        # Crons: diario (forecast) y semanal (SARIMAX)
```

(Estructura tentativa; se concreta en el plan de implementación.)

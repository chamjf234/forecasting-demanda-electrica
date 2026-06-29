# ⚡ Servicio vivo de forecasting de demanda eléctrica

**🔴 Demo en vivo:** https://forecasting-demanda-electrica-vksx6ueygoucdhwzvoordf.streamlit.app/

Pronóstico de **demanda eléctrica horaria** (región **PJM**, EE.UU.) servido como un
sistema que **se mantiene y se monitorea solo**. Cada día, un cron de GitHub Actions
ingiere datos frescos de la [EIA](https://www.eia.gov/opendata/), pronostica las
próximas 24 h con cuatro modelos y verifica la predicción del día anterior contra el
valor real — con un backtest a prueba de *data leakage*.

> **Por qué este proyecto.** El foco es de **ML Engineering**, no de modelado: lo que
> se busca demostrar no es un modelo brillante, sino que el modelo **vive dentro de un
> sistema automatizado, versionado y monitoreado**. El historial de commits de la
> carpeta [`data/`](data/) — generado por el robot en la nube — es la prueba de que el
> loop corre día tras día. Esa es la diferencia entre entregar un notebook y entregar
> un servicio.

📓 **¿Quieres entender el código por dentro?** El notebook
[`notebooks/walkthrough.ipynb`](notebooks/walkthrough.ipynb) explica el proyecto
**celda por celda**: cada módulo, el porqué de cada decisión, los problemas reales y
demos ejecutables sobre los datos reales.

---

## Tabla de contenidos
- [Cómo funciona](#cómo-funciona)
- [Decisiones de diseño](#decisiones-de-diseño)
- [Los modelos](#los-modelos)
- [Dificultades y aprendizajes](#dificultades-y-aprendizajes)
- [Stack tecnológico](#stack-tecnológico)
- [Estructura del código](#estructura-del-código)
- [Setup y uso](#setup-y-uso)
- [Tests y CI/CD](#tests-y-cicd)

---

## Cómo funciona

```
  EIA API (demanda horaria)
        │  ingesta diaria (GitHub Actions cron)
        ▼
  Histórico versionado en el repo (Parquet)
        │
        ├──► Forecast 24 h (4 modelos) ──► Predicciones guardadas (con corte as-of)
        │                                        │
        ▼                                        ▼
  Backtest continuo  ◄──────────────────  Valor real del día siguiente
        │
        ▼
  Métricas + error rodante  ──►  Dashboard (Streamlit)
```

El sistema **no es un notebook que corre una vez**. Es un loop: cada día (1) trae
datos nuevos, (2) pronostica las próximas 24 h, (3) compara la predicción de *ayer*
contra lo que realmente pasó, y (4) actualiza métricas en un dashboard público.

---

## Decisiones de diseño

Las decisiones que de verdad definen el proyecto (el detalle, en el notebook y en
[`docs/superpowers/specs`](docs/superpowers/specs)):

| # | Decisión | Por qué |
|---|----------|---------|
| 1 | **Dos versiones de cada valor real** (`value_first_reported` y `value_current`) | La EIA **revisa** sus cifras hacia atrás. Comparar contra un valor que cambió después es *data leakage* silencioso. |
| 2 | **Interfaz común de modelos** `predict(history, horizon)` | Hace los 4 modelos intercambiables y comparables con el mismo código. |
| 3 | **Fourier en vez de SARIMA "puro"** | La serie tiene 3 estacionalidades (diaria, semanal, anual); SARIMA clásico solo maneja una. *Dynamic harmonic regression* las modela todas. |
| 4 | **Corte *as-of* al pronosticar** | El modelo solo ve datos `period <= as_of`. **Nunca el futuro** — la regla de oro anti-leakage. |
| 5 | **Dos regímenes de backtest, etiquetados** | *Forward-test* (predicciones en vivo vs reales, honesto) y *backtest histórico* (datos revisados, ligeramente optimista). La honestidad metodológica es parte del valor. |
| + | **Repo dedicado** (no subcarpeta de un monorepo) | Los commits diarios del robot no ensucian el historial del portafolio. |
| + | **Dependencias separadas** (liviano vs pesado) | El dashboard se despliega sin PyTorch; las libs de modelos solo se instalan en CI/cron/local. |

---

## Los modelos

Se comparan **cuatro niveles** de sofisticación con la misma metodología. La pregunta
interesante: *¿le gana un modelo fundacional de 2024 a un LightGBM bien hecho y a un
baseline ingenuo, medido honestamente?*

| Nivel | Modelo | Qué demuestra |
|-------|--------|---------------|
| 0 | **Seasonal naive** | La vara mínima honesta (el valor de hace una semana) |
| 1 | **SARIMAX + Fourier** (24 h/168 h) | Econometría clásica rigurosa — *job semanal aislado* |
| 2 | **LightGBM** + calendario y lags | ML tabular de industria (forecasting directo, sin recursión) |
| 3 | **Chronos-Bolt** (Amazon) | *Foundation model* de series, zero-shot y probabilístico |

**Resultado indicativo** (backtest rolling-origin sobre la serie real de PJM, datos
limpios): LightGBM ≈ **4.5 % sMAPE** le gana ~2× al seasonal naive (≈ 9.4 %). El
modelo fundacional aporta forecasts probabilísticos (bandas de incertidumbre) sin
entrenar en la serie. Los números se actualizan en vivo en el dashboard.

> El paralelo con *transfer learning* en visión: igual que se usa una CNN
> preentrenada (EfficientNet) para *embeddings* de imágenes, aquí se usa un
> transformer preentrenado (Chronos) para forecasting — misma idea, dominio nuevo.

---

## Dificultades y aprendizajes

Lo más valioso del proyecto fueron los problemas que **solo aparecieron al correr el
sistema con datos reales** — ninguno lo veían los tests sintéticos. El hilo conductor:
**"compila y pasa los tests" ≠ "funciona con datos reales"**.

### 🐛 1. Huecos horarios en los datos de la EIA
La EIA tiene **horas faltantes** (p. ej. en cambios de horario). Los modelos asumen
una serie horaria *contigua*: el seasonal naive usa posiciones ("hace 168 posiciones")
y LightGBM busca el lag por timestamp exacto. Un hueco hacía que LightGBM lanzara
`KeyError` y que el naive quedara desalineado.
**Solución:** [`series.regularize_hourly`](src/forecasting/series.py) reindexa a una
grilla horaria completa e interpola.

### 🐛 2. Valores basura (centinelas de la EIA)
Al verificar, LightGBM daba un MAE **absurdo (~48 %)** frente al baseline — algo que no
tiene sentido para demanda eléctrica. Investigando, el máximo de la serie era
**≈ 2.147 × 10⁹**: el valor 2³¹−1, el clásico "dato faltante" codificado como entero
máximo. ~72 horas traían estos centinelas (o ceros). Corrompían el entrenamiento *y*
la evaluación (un "valor real" de 2.1e9 infla el error de todos los modelos).
**Solución:** `regularize_hourly` marca como implausibles los valores `<= 0` o
`> 5 × mediana` y los interpola. Con datos limpios, LightGBM pasó de ~48 % a ~4.5 %.
*Esta fue la lección más importante: la calidad de datos del mundo real importa más
que el modelo.*

### 🐛 3. La API de Chronos cambió de nombre
El método se invocaba con `inputs=` (no `context=`, como sugería la documentación
antigua). Se detectó haciendo un *spike* de verificación —probar la API real antes de
escribir el adaptador— en vez de confiar en la memoria. **Verifica la API, no la
inventes.**

### 🐛 4. El deploy del dashboard fallaba
Streamlit Community Cloud (free tier) no podía instalar **PyTorch** (cientos de MB).
**Solución:** separar `requirements.txt` (liviano: solo lo que el dashboard necesita)
de `requirements-models.txt` (torch, lightgbm, statsforecast, chronos — solo para
CI/cron/local). El dashboard no importa ninguna librería pesada.

### 🐛 5. SARIMAX agotaba la memoria del runner (OOM en CI)
El job semanal de SARIMAX moría en GitHub Actions con *exit code 143* (SIGTERM):
`AutoARIMA` ajustando sobre ~48 000 horas (5 años) agotaba la RAM del runner gratuito.
**Solución:** acotar la ventana de entrenamiento de SARIMAX a 90 días — para un
forecast a 24 h, la dinámica reciente es lo que importa. El ajuste pasó de "sin
memoria" a ~9 s, sin perder calidad. Decisión de ingeniería clásica: el modelo no
necesita *toda* la historia, solo la suficiente.

---

## Stack tecnológico

- **Lenguaje/datos:** Python, pandas, NumPy, PyArrow (Parquet).
- **Modelos:** LightGBM, statsforecast (Nixtla, AutoARIMA), Chronos-Bolt + PyTorch.
- **Dashboard:** Streamlit (desplegado en Streamlit Community Cloud).
- **Automatización:** GitHub Actions (cron diario y semanal, CI de tests).
- **Datos:** [EIA Open Data API v2](https://www.eia.gov/opendata/) (demanda horaria por *balancing authority*).
- **Calidad:** pytest (TDD), `pip install -e .` (src layout).

Todo corre en **CPU** y sobre *free tiers* — costo recurrente cero.

---

## Estructura del código

```
src/forecasting/
├── config.py          # Constantes (región, rutas, schema)
├── store.py           # Histórico + disciplina de snapshot (anti-leakage)
├── ingest.py          # API de la EIA (parseo + paginación)
├── bootstrap.py       # Backfill inicial del histórico
├── series.py          # Limpieza y regularización (huecos, outliers)
├── models/            # 4 modelos (naive, sarimax, lightgbm, chronos) + registry
├── forecast_daily.py  # Orquestación: corte as-of, corre modelos, persiste
├── predictions.py     # Persistencia idempotente de predicciones
├── backtest.py        # Forward-test + backtest histórico rolling-origin
├── metrics.py         # MAE y sMAPE
├── views.py           # Preparación de datos para el dashboard
└── run_daily.py       # Pipeline diario (ingesta + forecast) — entrypoint del cron
dashboard/app.py       # App Streamlit (solo lee Parquet; sin deps pesadas)
.github/workflows/     # tests (CI), daily (forecast), weekly (SARIMAX)
data/                  # Histórico y predicciones versionados (la prueba del loop)
docs/superpowers/      # Diseño (specs) y planes de implementación
notebooks/             # walkthrough.ipynb — explicación celda por celda
```

Cada módulo tiene **una sola responsabilidad** y una interfaz clara, de modo que se
puede entender y testear de forma independiente.

---

## Setup y uso

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1   |  Unix: source .venv/bin/activate

pip install -r requirements.txt              # base / dashboard
pip install -r requirements-models.txt       # modelos (torch, lightgbm, statsforecast, chronos)
pip install -e .                             # instala el paquete `forecasting`

cp .env.example .env                         # pon tu EIA_API_KEY (https://www.eia.gov/opendata/register.php)
```

```bash
# Backfill del histórico (una sola vez):
python -m forecasting.bootstrap 2021-01-01T00 2026-06-19T23
# Pipeline diario (ingesta reciente + forecast):
python -m forecasting.run_daily
# Dashboard local:
streamlit run dashboard/app.py
```

---

## Tests y CI/CD

```bash
pytest -q                          # tests rápidos (omite los 'slow')
pytest -q -m "slow or not slow"    # incluye SARIMAX y Chronos (más lentos)
```

Todo el proyecto se construyó con **TDD** (test primero, luego implementación). En
GitHub Actions corren tres workflows: **tests** (en cada push), **daily** (cron de
ingesta + forecast que commitea `data/`) y **weekly** (SARIMAX aislado). La API key se
inyecta como *secret*, nunca en el código.

# Fundación de Datos — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construir la capa de datos del servicio de forecasting: traer demanda eléctrica horaria de la EIA, versionarla con disciplina de snapshot (anti-leakage) y poder rellenar el histórico de años anteriores.

**Architecture:** Tres módulos con responsabilidad única. `store` lee/escribe el histórico en Parquet con upsert idempotente que rastrea revisiones (primer-valor-reportado vs valor-actual). `ingest` llama la API v2 de la EIA, pagina y normaliza la respuesta a un DataFrame. `bootstrap` orquesta un backfill único de varios años. Parquet como almacenamiento (eficiente, tipado); DuckDB se incorporará como motor de consulta en planes posteriores (backtest).

**Tech Stack:** Python 3.11+, pandas, pyarrow (Parquet), requests (HTTP), python-dotenv (API key), pytest.

---

## Contexto para quien ejecuta (lee esto antes de empezar)

- **Spec de referencia:** `docs/superpowers/specs/2026-06-15-forecasting-demanda-electrica-design.md`. Léelo: explica el *porqué* de cada decisión.
- **Este proyecto será un repo dedicado** (no subcarpeta del monorepo Portfolio). Hoy vive en `data-science/forecasting-demanda-electrica/` dentro del monorepo; la Tarea 0 lo convierte en repo independiente.
- **Disciplina de snapshot (el núcleo):** la EIA *revisa* sus cifras de demanda hacia atrás. Por eso, por cada (hora, región) guardamos DOS valores: `value_first_reported` (lo primero que publicó la EIA) y `value_current` (tras revisiones). Esto permite medir el error contra ambos y evita que las revisiones corrompan métricas pasadas.
- **API de la EIA:** endpoint `https://api.eia.gov/v2/electricity/rto/region-data/data/`. Facets relevantes: `respondent` (p.ej. `PJM`), `type` (`D` = demanda real). La API key va en la URL (`?api_key=...`), nunca en headers. Pagina con `length` (máx 5000) y `offset`. Periodos horarios vienen como string `"YYYY-MM-DDTHH"` en UTC y los valores como string.
- **Convención de timezone:** TODO se guarda en UTC. La EIA entrega UTC en este endpoint.

---

## File Structure

```
forecasting-demanda-electrica/
├── README.md                        # Skeleton; se completa en plan de dashboard/CI
├── CLAUDE.md                         # Contexto del subproyecto
├── pyproject.toml                    # Metadata + deps + config de pytest
├── requirements.txt                  # Deps pinneadas para CI
├── .gitignore                        # Ignora venv, .env, caches (NO el histórico)
├── .env.example                      # Plantilla: EIA_API_KEY=
├── docs/superpowers/                 # specs/ y plans/ (ya existen)
├── src/forecasting/
│   ├── __init__.py
│   ├── config.py                     # Constantes: región, schema, rutas, type codes
│   ├── store.py                      # read_demand_history, upsert_demand
│   ├── ingest.py                     # parse_eia_response, fetch_demand
│   └── bootstrap.py                  # backfill_demand + CLI
├── tests/
│   ├── __init__.py
│   ├── conftest.py                   # fixtures: tmp parquet path, sample EIA JSON
│   ├── test_store.py
│   ├── test_ingest.py
│   └── test_bootstrap.py
└── data/                             # Histórico versionado (Parquet). Se crea en runtime.
```

Cada archivo tiene una responsabilidad: `config` no tiene lógica, solo constantes; `store` no sabe de HTTP; `ingest` no sabe de Parquet; `bootstrap` solo orquesta. Esto permite testear `store` sin red y `ingest` con respuestas mockeadas.

---

## Task 0: Scaffolding del repo dedicado

**Files:**
- Create: `pyproject.toml`, `requirements.txt`, `.gitignore`, `.env.example`, `CLAUDE.md`, `README.md`
- Create: `src/forecasting/__init__.py`, `tests/__init__.py`, `tests/conftest.py`
- Create: `src/forecasting/config.py`

- [ ] **Step 1: Inicializar el repo dedicado**

Desde dentro de la carpeta del proyecto (que hoy está en el monorepo), conviértela en repo Git propio:

```bash
cd data-science/forecasting-demanda-electrica
git init
```

(El repo remoto en GitHub se crea en el plan de CI/CD. Por ahora basta el repo local.)

- [ ] **Step 2: Crear `.gitignore`**

```gitignore
# Entornos
.venv/
venv/
__pycache__/
*.pyc

# Secrets
.env

# Caches de herramientas
.pytest_cache/
.ruff_cache/

# Modelos de HuggingFace descargados (Chronos) — se bajan en runtime
.cache/

# NOTA: data/ NO se ignora. El histórico versionado es la prueba del loop vivo.
```

- [ ] **Step 3: Crear `requirements.txt`**

```
pandas>=2.2
pyarrow>=16.0
requests>=2.32
python-dotenv>=1.0
pytest>=8.0
```

- [ ] **Step 4: Crear `pyproject.toml`**

```toml
[project]
name = "forecasting-demanda-electrica"
version = "0.1.0"
description = "Servicio vivo de forecasting de demanda eléctrica horaria (EIA)"
requires-python = ">=3.11"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

`pythonpath = ["src"]` permite `import forecasting...` en los tests sin instalar el paquete. Es el patrón "src layout".

- [ ] **Step 5: Crear `.env.example`**

```
# Pide tu API key gratis en https://www.eia.gov/opendata/register.php
EIA_API_KEY=
```

- [ ] **Step 6: Crear `CLAUDE.md` del subproyecto**

```markdown
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
```

- [ ] **Step 7: Crear `README.md` skeleton**

```markdown
# Servicio vivo de forecasting de demanda eléctrica

Pronóstico de demanda eléctrica horaria (región PJM, datos de la EIA) servido
como un sistema que se mantiene y monitorea solo. (Documentación completa y
demo: pendientes en planes de orquestación y dashboard.)

## Estado
En construcción. Plan actual: fundación de datos (ingesta + histórico versionado).
```

- [ ] **Step 8: Crear `src/forecasting/__init__.py` y `tests/__init__.py` (vacíos)**

```python
# src/forecasting/__init__.py  (archivo vacío)
```

```python
# tests/__init__.py  (archivo vacío)
```

- [ ] **Step 9: Crear `src/forecasting/config.py`**

```python
"""Constantes compartidas. Sin lógica: solo configuración."""
from pathlib import Path

# Región objetivo (balancing authority de la EIA). Ver spec §3.
REGION = "PJM"

# Type codes de la EIA en el endpoint region-data:
#   "D"  = demanda real (lo que predecimos/verificamos)
#   "DF" = pronóstico día-adelante de la propia EIA (baseline futuro, roadmap)
DEMAND_TYPE = "D"

EIA_BASE_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"

# Rutas de datos versionados.
DATA_DIR = Path("data")
DEMAND_HISTORY_PATH = DATA_DIR / "demand_history.parquet"

# Schema del histórico de demanda. Todas las horas en UTC.
DEMAND_COLUMNS = [
    "period",                 # datetime64[ns, UTC] — hora de la observación
    "respondent",             # str — región (p.ej. "PJM")
    "value_first_reported",   # float — primer valor publicado por la EIA (MWh)
    "value_current",          # float — valor tras posibles revisiones (MWh)
    "first_seen_at",          # datetime64[ns, UTC] — cuándo lo vimos por 1a vez
    "last_updated_at",        # datetime64[ns, UTC] — última vez que cambió
]
```

- [ ] **Step 10: Instalar deps y verificar que pytest corre (sin tests aún)**

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1   |  bash: source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```
Expected: `no tests ran` (exit 5) — confirma que pytest está instalado y configurado.

- [ ] **Step 11: Commit**

```bash
git add pyproject.toml requirements.txt .gitignore .env.example CLAUDE.md README.md src/forecasting/__init__.py src/forecasting/config.py tests/__init__.py
git commit -m "chore: scaffolding del repo de forecasting de demanda eléctrica"
```

---

## Task 1: `store` — leer el histórico (con schema vacío si no existe)

**Files:**
- Create: `src/forecasting/store.py`
- Create: `tests/conftest.py`
- Test: `tests/test_store.py`

- [ ] **Step 1: Crear `tests/conftest.py` con fixtures base**

```python
import pandas as pd
import pytest


@pytest.fixture
def tmp_history_path(tmp_path):
    """Ruta a un parquet temporal para el histórico de demanda."""
    return tmp_path / "demand_history.parquet"


@pytest.fixture
def sample_observations():
    """Dos observaciones crudas (period UTC + value), como las produce ingest."""
    return pd.DataFrame(
        {
            "period": pd.to_datetime(
                ["2024-01-01T00:00", "2024-01-01T01:00"], utc=True
            ),
            "respondent": ["PJM", "PJM"],
            "value": [100.0, 110.0],
        }
    )
```

- [ ] **Step 2: Escribir el test que falla**

```python
# tests/test_store.py
import pandas as pd
from forecasting import store
from forecasting.config import DEMAND_COLUMNS


def test_read_missing_returns_empty_with_schema(tmp_history_path):
    df = store.read_demand_history(tmp_history_path)
    assert list(df.columns) == DEMAND_COLUMNS
    assert len(df) == 0
```

- [ ] **Step 3: Correr el test y verificar que falla**

Run: `pytest tests/test_store.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'forecasting.store'`

- [ ] **Step 4: Implementación mínima**

```python
# src/forecasting/store.py
"""Lectura/escritura del histórico de demanda con disciplina de snapshot."""
from pathlib import Path

import pandas as pd

from forecasting.config import DEMAND_COLUMNS


def _empty_history() -> pd.DataFrame:
    """DataFrame vacío con el schema correcto (tipos incluidos)."""
    df = pd.DataFrame(columns=DEMAND_COLUMNS)
    df["period"] = pd.to_datetime(df["period"], utc=True)
    df["first_seen_at"] = pd.to_datetime(df["first_seen_at"], utc=True)
    df["last_updated_at"] = pd.to_datetime(df["last_updated_at"], utc=True)
    df["value_first_reported"] = df["value_first_reported"].astype("float64")
    df["value_current"] = df["value_current"].astype("float64")
    df["respondent"] = df["respondent"].astype("object")
    return df


def read_demand_history(path: Path) -> pd.DataFrame:
    """Lee el histórico desde Parquet. Si no existe, devuelve vacío con schema."""
    path = Path(path)
    if not path.exists():
        return _empty_history()
    return pd.read_parquet(path)
```

- [ ] **Step 5: Correr el test y verificar que pasa**

Run: `pytest tests/test_store.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/forecasting/store.py tests/conftest.py tests/test_store.py
git commit -m "feat(store): leer histórico de demanda con schema vacío por defecto"
```

---

## Task 2: `store` — upsert idempotente con rastreo de revisiones

**Files:**
- Modify: `src/forecasting/store.py`
- Test: `tests/test_store.py`

Esta es la pieza central del anti-leakage. `upsert_demand` toma observaciones crudas (`period`, `respondent`, `value`) y las funde en el histórico:
- Periodo nuevo → inserta con `value_first_reported == value_current == value` y timestamps = ahora.
- Periodo existente con MISMO valor → no cambia nada (idempotente).
- Periodo existente con valor DISTINTO → actualiza `value_current` y `last_updated_at`; **preserva** `value_first_reported`.

- [ ] **Step 1: Escribir los tests que fallan**

```python
# tests/test_store.py  (añadir al final)
import pandas as pd
from forecasting import store


def _now():
    return pd.Timestamp("2024-06-01T12:00", tz="UTC")


def test_upsert_into_empty_inserts_rows(tmp_history_path, sample_observations):
    result = store.upsert_demand(tmp_history_path, sample_observations, now=_now())
    df = store.read_demand_history(tmp_history_path)
    assert len(df) == 2
    assert result == {"inserted": 2, "revised": 0, "unchanged": 0}
    row = df.sort_values("period").iloc[0]
    assert row["value_first_reported"] == 100.0
    assert row["value_current"] == 100.0
    assert row["first_seen_at"] == _now()


def test_upsert_same_values_is_idempotent(tmp_history_path, sample_observations):
    store.upsert_demand(tmp_history_path, sample_observations, now=_now())
    result = store.upsert_demand(tmp_history_path, sample_observations, now=_now())
    df = store.read_demand_history(tmp_history_path)
    assert len(df) == 2  # no duplica
    assert result == {"inserted": 0, "revised": 0, "unchanged": 2}


def test_upsert_revision_preserves_first_reported(tmp_history_path, sample_observations):
    store.upsert_demand(tmp_history_path, sample_observations, now=_now())
    revised = sample_observations.copy()
    revised.loc[0, "value"] = 105.0  # la EIA revisó la primera hora: 100 -> 105
    later = pd.Timestamp("2024-06-02T12:00", tz="UTC")
    result = store.upsert_demand(tmp_history_path, revised, now=later)

    df = store.read_demand_history(tmp_history_path).sort_values("period")
    row = df.iloc[0]
    assert row["value_first_reported"] == 100.0   # se preserva el original
    assert row["value_current"] == 105.0          # se actualiza al revisado
    assert row["last_updated_at"] == later
    assert row["first_seen_at"] == _now()         # no cambia
    assert result == {"inserted": 0, "revised": 1, "unchanged": 1}
```

- [ ] **Step 2: Correr y verificar que fallan**

Run: `pytest tests/test_store.py -v`
Expected: FAIL con `AttributeError: module 'forecasting.store' has no attribute 'upsert_demand'`

- [ ] **Step 3: Implementación**

```python
# src/forecasting/store.py  (añadir imports y función)
from pathlib import Path

import pandas as pd

from forecasting.config import DEMAND_COLUMNS


def upsert_demand(path: Path, observations: pd.DataFrame, now: pd.Timestamp) -> dict:
    """Funde observaciones crudas en el histórico con rastreo de revisiones.

    `observations` debe tener columnas: period (UTC), respondent, value.
    `now` es el timestamp UTC del momento de ingesta (inyectado para testeo).
    Devuelve conteos: {"inserted", "revised", "unchanged"}.
    """
    path = Path(path)
    history = read_demand_history(path)
    # Índice por clave (period, respondent) para lookups O(1).
    existing = {
        (r.period, r.respondent): r for r in history.itertuples(index=False)
    }

    rows = []
    counts = {"inserted": 0, "revised": 0, "unchanged": 0}
    for obs in observations.itertuples(index=False):
        key = (obs.period, obs.respondent)
        prev = existing.pop(key, None)
        if prev is None:
            counts["inserted"] += 1
            rows.append(
                {
                    "period": obs.period,
                    "respondent": obs.respondent,
                    "value_first_reported": float(obs.value),
                    "value_current": float(obs.value),
                    "first_seen_at": now,
                    "last_updated_at": now,
                }
            )
        elif float(obs.value) != prev.value_current:
            counts["revised"] += 1
            rows.append(
                {
                    "period": prev.period,
                    "respondent": prev.respondent,
                    "value_first_reported": prev.value_first_reported,
                    "value_current": float(obs.value),
                    "first_seen_at": prev.first_seen_at,
                    "last_updated_at": now,
                }
            )
        else:
            counts["unchanged"] += 1
            rows.append(prev._asdict())

    # Filas existentes que no aparecieron en esta tanda se conservan tal cual.
    for prev in existing.values():
        rows.append(prev._asdict())

    merged = pd.DataFrame(rows, columns=DEMAND_COLUMNS).sort_values(
        ["respondent", "period"]
    ).reset_index(drop=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(path, index=False)
    return counts
```

- [ ] **Step 4: Correr y verificar que pasan**

Run: `pytest tests/test_store.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/forecasting/store.py tests/test_store.py
git commit -m "feat(store): upsert idempotente con rastreo de revisiones (anti-leakage)"
```

---

## Task 3: `ingest` — parsear la respuesta JSON de la EIA

**Files:**
- Create: `src/forecasting/ingest.py`
- Modify: `tests/conftest.py`
- Test: `tests/test_ingest.py`

Separamos el parseo (puro, testeable sin red) del fetch (con HTTP). `parse_eia_response` toma el dict JSON de la EIA y devuelve un DataFrame con columnas `period` (UTC), `respondent`, `value` — justo lo que `upsert_demand` espera.

- [ ] **Step 1: Añadir fixture con un JSON de muestra de la EIA**

```python
# tests/conftest.py  (añadir al final)
@pytest.fixture
def sample_eia_payload():
    """Forma real (simplificada) de la respuesta v2 de la EIA. value viene como str."""
    return {
        "response": {
            "total": "2",
            "data": [
                {"period": "2024-01-01T00", "respondent": "PJM",
                 "type": "D", "value": "85000", "value-units": "megawatthours"},
                {"period": "2024-01-01T01", "respondent": "PJM",
                 "type": "D", "value": "83250", "value-units": "megawatthours"},
            ],
        }
    }
```

- [ ] **Step 2: Escribir el test que falla**

```python
# tests/test_ingest.py
import pandas as pd
from forecasting import ingest


def test_parse_eia_response_shapes_dataframe(sample_eia_payload):
    df = ingest.parse_eia_response(sample_eia_payload)
    assert list(df.columns) == ["period", "respondent", "value"]
    assert len(df) == 2
    assert str(df["period"].dtype) == "datetime64[ns, UTC]"
    assert df["value"].dtype == "float64"
    assert df.iloc[0]["period"] == pd.Timestamp("2024-01-01T00:00", tz="UTC")
    assert df.iloc[0]["value"] == 85000.0


def test_parse_eia_response_empty(sample_eia_payload):
    empty = {"response": {"total": "0", "data": []}}
    df = ingest.parse_eia_response(empty)
    assert list(df.columns) == ["period", "respondent", "value"]
    assert len(df) == 0
```

- [ ] **Step 3: Correr y verificar que falla**

Run: `pytest tests/test_ingest.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'forecasting.ingest'`

- [ ] **Step 4: Implementación**

```python
# src/forecasting/ingest.py
"""Ingesta de demanda horaria desde la API v2 de la EIA."""
import pandas as pd

PARSED_COLUMNS = ["period", "respondent", "value"]


def parse_eia_response(payload: dict) -> pd.DataFrame:
    """Normaliza el JSON de la EIA a un DataFrame (period UTC, respondent, value).

    La EIA devuelve `period` como "YYYY-MM-DDTHH" (UTC) y `value` como string.
    """
    records = payload.get("response", {}).get("data", [])
    if not records:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in PARSED_COLUMNS}).astype(
            {"value": "float64"}
        ).assign(period=lambda d: pd.to_datetime(d["period"], utc=True))

    df = pd.DataFrame(records)
    out = pd.DataFrame(
        {
            "period": pd.to_datetime(df["period"], utc=True, format="%Y-%m-%dT%H"),
            "respondent": df["respondent"].astype("object"),
            "value": df["value"].astype("float64"),
        }
    )
    return out[PARSED_COLUMNS]
```

- [ ] **Step 5: Correr y verificar que pasa**

Run: `pytest tests/test_ingest.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add src/forecasting/ingest.py tests/conftest.py tests/test_ingest.py
git commit -m "feat(ingest): parsear respuesta JSON de la EIA a DataFrame normalizado"
```

---

## Task 4: `ingest` — `fetch_demand` con paginación (HTTP mockeado)

**Files:**
- Modify: `src/forecasting/ingest.py`
- Test: `tests/test_ingest.py`

`fetch_demand` arma la URL, pagina con `offset`/`length` hasta traer todo el rango, y concatena los parseos. Se testea inyectando un cliente HTTP falso (no se toca la red real).

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/test_ingest.py  (añadir al final)
from forecasting import ingest


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_fetch_demand_paginates(monkeypatch):
    """Con total=3 y length=2, debe hacer 2 requests y concatenar 3 filas."""
    pages = [
        {"response": {"total": "3", "data": [
            {"period": "2024-01-01T00", "respondent": "PJM", "type": "D", "value": "10"},
            {"period": "2024-01-01T01", "respondent": "PJM", "type": "D", "value": "11"},
        ]}},
        {"response": {"total": "3", "data": [
            {"period": "2024-01-01T02", "respondent": "PJM", "type": "D", "value": "12"},
        ]}},
    ]
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(params["offset"])
        return FakeResponse(pages[len(calls) - 1])

    monkeypatch.setattr(ingest.requests, "get", fake_get)

    df = ingest.fetch_demand(
        respondent="PJM",
        start="2024-01-01T00",
        end="2024-01-01T02",
        api_key="FAKE",
        page_length=2,
    )
    assert len(df) == 3
    assert calls == [0, 2]  # paginó: offset 0 luego 2
    assert df.iloc[2]["value"] == 12.0
```

- [ ] **Step 2: Correr y verificar que falla**

Run: `pytest tests/test_ingest.py::test_fetch_demand_paginates -v`
Expected: FAIL con `AttributeError: module 'forecasting.ingest' has no attribute 'fetch_demand'`

- [ ] **Step 3: Implementación**

```python
# src/forecasting/ingest.py  (añadir imports arriba y función)
import requests

from forecasting.config import DEMAND_TYPE, EIA_BASE_URL

# ... parse_eia_response sin cambios ...


def fetch_demand(
    respondent: str,
    start: str,
    end: str,
    api_key: str,
    page_length: int = 5000,
) -> pd.DataFrame:
    """Trae demanda horaria (type=D) de la EIA para [start, end], paginando.

    start/end en formato "YYYY-MM-DDTHH" (UTC). Devuelve DataFrame normalizado.
    """
    frames = []
    offset = 0
    while True:
        params = {
            "api_key": api_key,
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": respondent,
            "facets[type][]": DEMAND_TYPE,
            "start": start,
            "end": end,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": offset,
            "length": page_length,
        }
        resp = requests.get(EIA_BASE_URL, params=params, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        frames.append(parse_eia_response(payload))

        total = int(payload.get("response", {}).get("total", 0))
        offset += page_length
        if offset >= total:
            break

    return pd.concat(frames, ignore_index=True)
```

- [ ] **Step 4: Correr y verificar que pasa**

Run: `pytest tests/test_ingest.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/forecasting/ingest.py tests/test_ingest.py
git commit -m "feat(ingest): fetch_demand con paginación contra la API de la EIA"
```

---

## Task 5: `bootstrap` — backfill del histórico (con CLI)

**Files:**
- Create: `src/forecasting/bootstrap.py`
- Test: `tests/test_bootstrap.py`

`backfill_demand` une `fetch_demand` + `upsert_demand`: trae un rango largo y lo persiste. Se testea inyectando un `fetch_fn` falso (no se toca red ni EIA real). Incluye un `__main__` para correrlo desde la terminal una sola vez.

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/test_bootstrap.py
import pandas as pd
from forecasting import bootstrap, store


def test_backfill_demand_persists_history(tmp_history_path):
    def fake_fetch(respondent, start, end, api_key):
        return pd.DataFrame(
            {
                "period": pd.to_datetime(
                    ["2024-01-01T00:00", "2024-01-01T01:00"], utc=True
                ),
                "respondent": [respondent, respondent],
                "value": [100.0, 110.0],
            }
        )

    counts = bootstrap.backfill_demand(
        path=tmp_history_path,
        respondent="PJM",
        start="2024-01-01T00",
        end="2024-01-01T01",
        api_key="FAKE",
        now=pd.Timestamp("2024-06-01T12:00", tz="UTC"),
        fetch_fn=fake_fetch,
    )

    df = store.read_demand_history(tmp_history_path)
    assert len(df) == 2
    assert counts["inserted"] == 2
```

- [ ] **Step 2: Correr y verificar que falla**

Run: `pytest tests/test_bootstrap.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'forecasting.bootstrap'`

- [ ] **Step 3: Implementación**

```python
# src/forecasting/bootstrap.py
"""Backfill único del histórico de demanda (datos viejos para entrenar/backtest)."""
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from forecasting import ingest, store
from forecasting.config import DEMAND_HISTORY_PATH, REGION


def backfill_demand(
    path: Path,
    respondent: str,
    start: str,
    end: str,
    api_key: str,
    now: pd.Timestamp,
    fetch_fn=ingest.fetch_demand,
) -> dict:
    """Trae demanda en [start, end] y la persiste vía upsert. `fetch_fn` inyectable."""
    observations = fetch_fn(
        respondent=respondent, start=start, end=end, api_key=api_key
    )
    return store.upsert_demand(path, observations, now=now)


if __name__ == "__main__":
    # Uso: python -m forecasting.bootstrap 2021-01-01T00 2026-06-01T00
    import sys

    load_dotenv()
    api_key = os.environ["EIA_API_KEY"]
    start, end = sys.argv[1], sys.argv[2]
    counts = backfill_demand(
        path=DEMAND_HISTORY_PATH,
        respondent=REGION,
        start=start,
        end=end,
        api_key=api_key,
        now=pd.Timestamp.now(tz="UTC"),
    )
    print(f"Backfill {REGION} [{start}..{end}]: {counts}")
```

- [ ] **Step 4: Correr y verificar que pasa**

Run: `pytest tests/test_bootstrap.py -v`
Expected: PASS

- [ ] **Step 5: Correr la suite completa**

Run: `pytest -q`
Expected: PASS (todos: store 4, ingest 3, bootstrap 1)

- [ ] **Step 6: Commit**

```bash
git add src/forecasting/bootstrap.py tests/test_bootstrap.py
git commit -m "feat(bootstrap): backfill del histórico de demanda con CLI"
```

- [ ] **Step 7: (Manual, con API key real) Smoke test del backfill**

Con una `EIA_API_KEY` real en `.env`, traer un rango corto y confirmar que se versiona:

```bash
python -m forecasting.bootstrap 2024-01-01T00 2024-01-07T23
```
Expected: imprime conteos con `inserted > 0` y crea `data/demand_history.parquet`.
Luego commitear el histórico (la prueba versionada):
```bash
git add data/demand_history.parquet
git commit -m "data: backfill inicial de demanda PJM (smoke test)"
```

---

## Self-Review

**1. Spec coverage (Plan 1 cubre la fundación de datos del spec §3, §4, §4.1):**
- EIA API, región PJM, granularidad horaria → Task 0 (config) + Task 4 (fetch). ✅
- Disciplina de snapshot (first_reported vs current) → Task 2. ✅
- Idempotencia de ingesta (spec §4 componentes) → Task 2 (test idempotente). ✅
- Backfill / bootstrap (spec §4.1) → Task 5. ✅
- Histórico versionado en repo (no ignorado) → Task 0 (.gitignore) + Task 5 Step 7. ✅
- Repo dedicado (spec §6) → Task 0 Step 1. ✅
- *Fuera de este plan (planes siguientes):* modelos (§5), forecast_daily/predictions store/backtest (§4, §5), dashboard y crons (§6). Documentado en la descomposición.

**2. Placeholder scan:** Sin "TBD"/"TODO". Todos los steps de código tienen código completo. ✅

**3. Type/firma consistency:**
- `parse_eia_response` / `fetch_demand` devuelven columnas `["period","respondent","value"]`; `upsert_demand` consume exactamente esas. ✅
- `read_demand_history` / `upsert_demand` usan `DEMAND_COLUMNS` de `config`. ✅
- `backfill_demand(fetch_fn=...)` y la firma de `fetch_demand(respondent,start,end,api_key)` coinciden (el test usa los mismos kwargs). ✅
- `now` se inyecta en `upsert_demand` y `backfill_demand` para tests deterministas. ✅

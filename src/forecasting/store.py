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


def upsert_demand(path: Path, observations: pd.DataFrame, now: pd.Timestamp) -> dict:
    """Funde observaciones crudas en el histórico con rastreo de revisiones.

    `observations` debe tener columnas: period (UTC), respondent, value.
    `now` es el timestamp UTC del momento de ingesta (inyectado para testeo).
    Devuelve conteos: {"inserted", "revised", "unchanged"}.
    """
    path = Path(path)
    history = read_demand_history(path)
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

    # Preservar filas existentes que no vinieron en el batch de observaciones
    for prev in existing.values():
        rows.append(prev._asdict())

    merged = pd.DataFrame(rows, columns=DEMAND_COLUMNS).sort_values(
        ["respondent", "period"]
    ).reset_index(drop=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(path, index=False)
    return counts

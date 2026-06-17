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

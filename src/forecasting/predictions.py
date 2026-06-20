"""Persistencia idempotente de predicciones en Parquet."""
from pathlib import Path

import pandas as pd

PREDICTION_COLUMNS = [
    "forecast_made_at",
    "model",
    "target_period",
    "horizon",
    "prediction",
    "q10",
    "q90",
]

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

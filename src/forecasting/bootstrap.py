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

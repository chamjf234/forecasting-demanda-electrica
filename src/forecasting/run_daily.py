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

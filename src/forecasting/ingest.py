"""Ingesta de demanda horaria desde la API v2 de la EIA."""
import pandas as pd
import requests

from forecasting.config import DEMAND_TYPE, EIA_BASE_URL

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
            # pandas 2+ infiere resolución "us" por defecto; forzamos "ns" para
            # coincidir con el dtype que usa store.upsert_demand en el histórico.
            "period": pd.to_datetime(df["period"], utc=True, format="%Y-%m-%dT%H").astype(
                "datetime64[ns, UTC]"
            ),
            "respondent": df["respondent"].astype("object"),
            "value": df["value"].astype("float64"),
        }
    )
    return out[PARSED_COLUMNS]


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

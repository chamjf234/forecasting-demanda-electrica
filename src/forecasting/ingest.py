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

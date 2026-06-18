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

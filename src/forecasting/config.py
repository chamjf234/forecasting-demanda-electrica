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

"""Registry de modelos de forecasting, agrupados por cadencia de ejecución (spec §5)."""
from forecasting.models.chronos_model import ChronosForecaster
from forecasting.models.lightgbm_model import LightGBMForecaster
from forecasting.models.naive import SeasonalNaive
from forecasting.models.sarimax import SarimaxForecaster


def daily_models():
    """Modelos rápidos/robustos que corren en el loop diario."""
    return [SeasonalNaive(), LightGBMForecaster(), ChronosForecaster()]


def weekly_models():
    """Modelos costosos, aislados en un job semanal (no tumban el loop diario)."""
    return [SarimaxForecaster()]


def all_models():
    return daily_models() + weekly_models()

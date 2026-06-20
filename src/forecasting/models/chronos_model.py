"""Chronos-Bolt (Amazon): transformer preentrenado, forecasting zero-shot probabilístico."""
import pandas as pd
import torch
from chronos import BaseChronosPipeline

from forecasting.models.base import FORECAST_COLUMNS, future_periods

QUANTILE_LEVELS = [0.1, 0.5, 0.9]


class ChronosForecaster:
    name = "chronos"

    def __init__(self, model_id: str = "amazon/chronos-bolt-small", context_length: int = 512):
        self.model_id = model_id
        self.context_length = context_length
        self._pipeline = None  # carga perezosa: solo al primer predict

    def _load(self):
        if self._pipeline is None:
            self._pipeline = BaseChronosPipeline.from_pretrained(
                self.model_id, device_map="cpu", torch_dtype=torch.float32
            )
        return self._pipeline

    def predict(self, history: pd.DataFrame, horizon: int) -> pd.DataFrame:
        pipeline = self._load()
        values = history["value"].to_numpy()[-self.context_length:]
        context = torch.tensor(values, dtype=torch.float32)

        # Nota: la API instalada (chronos-forecasting 2.3.0) nombra el parámetro
        # `inputs`, no `context` como en versiones previas de la librería.
        # El spike de verificación confirmó esto antes de escribir el código.
        quantiles, _mean = pipeline.predict_quantiles(
            inputs=context, prediction_length=horizon, quantile_levels=QUANTILE_LEVELS
        )
        q = quantiles[0].numpy()  # forma (horizon, 3): [q10, q50, q90]

        return pd.DataFrame(
            {
                "period": future_periods(history, horizon),
                "model": self.name,
                "prediction": q[:, 1],
                "q10": q[:, 0],
                "q90": q[:, 2],
            }
        )[FORECAST_COLUMNS + ["q10", "q90"]]

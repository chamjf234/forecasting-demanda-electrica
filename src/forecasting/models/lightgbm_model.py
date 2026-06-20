"""LightGBM con features de calendario + lags conocidos (24 h y 168 h)."""
import lightgbm as lgb
import pandas as pd

from forecasting.models.base import FORECAST_COLUMNS, future_periods

LAG_DAY = 24
LAG_WEEK = 168


def _calendar(idx: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "hour": idx.hour,
            "dayofweek": idx.dayofweek,
            "month": idx.month,
            "is_weekend": (idx.dayofweek >= 5).astype(int),
        }
    )


class LightGBMForecaster:
    name = "lightgbm"

    def __init__(self, **params):
        self.params = {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "random_state": 0,
            "verbose": -1,
            **params,
        }

    def _build_features(self, periods: pd.DatetimeIndex, lookup: dict) -> pd.DataFrame:
        feats = _calendar(periods)
        feats["lag_24"] = [lookup[p - pd.Timedelta(hours=LAG_DAY)] for p in periods]
        feats["lag_168"] = [lookup[p - pd.Timedelta(hours=LAG_WEEK)] for p in periods]
        return feats

    def predict(self, history: pd.DataFrame, horizon: int) -> pd.DataFrame:
        if horizon > LAG_DAY:
            raise ValueError(
                f"lightgbm asume horizon <= {LAG_DAY} (lag_24 conocido sin recursión)"
            )
        if len(history) < LAG_WEEK + 1:
            raise ValueError(
                f"lightgbm necesita > {LAG_WEEK} h de histórico, recibió {len(history)}"
            )
        series = history.set_index("period")["value"]
        lookup = series.to_dict()

        train_idx = series.index[LAG_WEEK:]
        X_train = self._build_features(train_idx, lookup)
        y_train = series.loc[train_idx].to_numpy()

        model = lgb.LGBMRegressor(**self.params)
        model.fit(X_train, y_train)

        future = future_periods(history, horizon)
        X_future = self._build_features(future, lookup)
        preds = model.predict(X_future)

        return pd.DataFrame(
            {"period": future, "model": self.name, "prediction": preds}
        )[FORECAST_COLUMNS]

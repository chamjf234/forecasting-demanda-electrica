import numpy as np
import pandas as pd
from forecasting import series


def test_regularize_fills_missing_hour():
    # serie con un hueco: falta la hora 02:00
    idx = pd.to_datetime(
        ["2025-01-01T00", "2025-01-01T01", "2025-01-01T03", "2025-01-01T04"], utc=True
    )
    df = pd.DataFrame({"period": idx, "value": [10.0, 20.0, 40.0, 50.0]})
    out = series.regularize_hourly(df)
    # ahora hay 5 horas contiguas
    assert len(out) == 5
    periods = out["period"].tolist()
    assert periods[2] == pd.Timestamp("2025-01-01T02", tz="UTC")
    # el hueco se interpola linealmente entre 20 y 40 -> 30
    assert out["value"].iloc[2] == 30.0


def test_regularize_contiguous_is_noop():
    # datos limpios, contiguos y positivos: regularize no debe cambiarlos
    idx = pd.date_range("2025-01-01", periods=10, freq="h", tz="UTC")
    values = (np.arange(10, dtype=float) + 100)  # 100..109, todos plausibles
    df = pd.DataFrame({"period": idx, "value": values})
    out = series.regularize_hourly(df)
    assert len(out) == 10
    assert out["value"].tolist() == values.tolist()


def test_regularize_interpolates_internal_nan():
    idx = pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC")
    df = pd.DataFrame({"period": idx, "value": [10.0, np.nan, 30.0, 40.0]})
    out = series.regularize_hourly(df)
    assert out["value"].iloc[1] == 20.0


def test_regularize_empty_returns_empty():
    df = pd.DataFrame({"period": pd.to_datetime([], utc=True), "value": []})
    out = series.regularize_hourly(df)
    assert len(out) == 0
    assert list(out.columns) == ["period", "value"]


def test_regularize_replaces_implausible_outlier():
    # valor centinela de la EIA (~2^31) tratado como faltante e interpolado
    idx = pd.date_range("2025-01-01", periods=5, freq="h", tz="UTC")
    df = pd.DataFrame({"period": idx, "value": [100.0, 110.0, 2_147_480_000.0, 130.0, 140.0]})
    out = series.regularize_hourly(df)
    assert out["value"].iloc[2] == 120.0  # interpolado entre 110 y 130


def test_regularize_replaces_nonpositive():
    idx = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
    df = pd.DataFrame({"period": idx, "value": [100.0, 0.0, 120.0]})
    out = series.regularize_hourly(df)
    assert out["value"].iloc[1] == 110.0  # 0 tratado como faltante e interpolado

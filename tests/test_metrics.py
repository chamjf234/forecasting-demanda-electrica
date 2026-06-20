import numpy as np
from forecasting import metrics


def test_mae_zero_when_perfect():
    assert metrics.mae([1, 2, 3], [1, 2, 3]) == 0.0


def test_mae_known_value():
    assert metrics.mae([0, 0], [1, 3]) == 2.0


def test_smape_zero_when_perfect():
    assert metrics.smape([10, 20], [10, 20]) == 0.0


def test_smape_known_value():
    val = metrics.smape([100.0], [110.0])
    assert abs(val - (2 * 10 / 210) * 100) < 1e-9


def test_smape_handles_zero_denominator():
    assert metrics.smape([0.0, 100.0], [0.0, 100.0]) == 0.0

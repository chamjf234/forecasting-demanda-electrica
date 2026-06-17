import pandas as pd
from forecasting import store
from forecasting.config import DEMAND_COLUMNS


def test_read_missing_returns_empty_with_schema(tmp_history_path):
    df = store.read_demand_history(tmp_history_path)
    assert list(df.columns) == DEMAND_COLUMNS
    assert len(df) == 0

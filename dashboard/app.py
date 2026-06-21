"""Dashboard del servicio de forecasting de demanda eléctrica (Streamlit)."""
import pandas as pd
import streamlit as st

from forecasting import backtest, predictions, store, views
from forecasting.config import DEMAND_HISTORY_PATH, PREDICTIONS_PATH, REGION

st.set_page_config(page_title="Forecasting demanda eléctrica", layout="wide")
st.title(f"⚡ Forecasting de demanda eléctrica — {REGION}")
st.caption(
    "Sistema vivo: cada día ingiere datos de la EIA, pronostica 24 h y verifica "
    "la predicción del día anterior. El histórico versionado en git es la prueba del loop."
)

history = store.read_demand_history(DEMAND_HISTORY_PATH)
preds = predictions.read_predictions(PREDICTIONS_PATH)

if history.empty or preds.empty:
    st.warning("Aún no hay datos suficientes. Corre el backfill y el primer forecast.")
    st.stop()

st.header("Forecast actual (próximas 24 h)")
latest = views.latest_forecast(preds)
chart = latest.pivot_table(index="target_period", columns="model", values="prediction")
st.line_chart(chart)

st.header("Predicho vs real (forward-test, point-in-time)")
pva = views.predicted_vs_actual(preds, history)
if not pva.empty:
    recent = pva.sort_values("target_period").tail(24 * 14)
    pivot = recent.pivot_table(index="target_period", columns="model", values="prediction")
    pivot["real"] = recent.groupby("target_period")["actual"].first()
    st.line_chart(pivot)

st.header("Error rodante (MAE, ventana 7 días)")
roll = views.rolling_error(pva, window=24 * 7)
if not roll["rolling_mae"].dropna().empty:
    roll_pivot = roll.pivot_table(index="target_period", columns="model", values="rolling_mae")
    st.line_chart(roll_pivot)

st.header("Métricas por modelo")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Forward-test (honesto, point-in-time)")
    st.dataframe(backtest.evaluate_predictions(preds, history))
with col2:
    st.subheader("Backtest histórico (datos revisados, optimista)")
    st.caption("Ligeramente optimista: usa datos ya revisados por la EIA.")
    st.dataframe(backtest.evaluate_predictions(preds, history, actual_col="value_first_reported"))

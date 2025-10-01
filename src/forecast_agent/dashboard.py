"""Streamlit dashboard to explore forecasts and ask questions."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from forecast_agent.agent import AgentConfig, ForecastAgent


st.set_page_config(page_title="Forecast Agent Dashboard", layout="wide")
st.title("Forecast Agent Dashboard")
st.caption(
    "Forecasts synthetic SaaS metrics and answers natural-language questions. "
    "All computations run locally."
)


def load_agent(data_path: str, horizon: int) -> ForecastAgent:
    cache_key = (Path(data_path).resolve(), horizon)
    return _load_agent_cached(cache_key)


@st.cache_resource(show_spinner=False)
def _load_agent_cached(cache_key: tuple[Path, int]) -> ForecastAgent:
    path, horizon = cache_key
    config = AgentConfig(data_path=str(path), horizon=horizon)
    return ForecastAgent(config)


with st.sidebar:
    st.header("Configuration")
    default_path = Path("saas_metrics.csv")
    data_path = st.text_input("Data file", value=str(default_path))
    horizon = st.slider("Forecast horizon (months)", min_value=3, max_value=24, value=12)

    agent: ForecastAgent | None = None
    if data_path:
        try:
            agent = load_agent(data_path, horizon)
        except Exception as exc:  # noqa: BLE001 - surfaced to user
            st.error(f"Failed to load data: {exc}")

if agent is None:
    st.stop()

metrics = agent.metrics
if not metrics:
    st.warning("No metrics available in the dataset.")
    st.stop()

default_index = metrics.index("monthly_recurring_revenue") if "monthly_recurring_revenue" in metrics else 0
selected_metric = st.sidebar.selectbox("Metric", metrics, index=default_index)

history = agent.data[[selected_metric]].copy()
history.index.name = "date"
history.reset_index(inplace=True)
history.rename(columns={selected_metric: "value"}, inplace=True)
history["source"] = "History"

forecast_df = agent.get_forecast(selected_metric).copy()
forecast_points = forecast_df[["date", "point"]].rename(columns={"point": "value"})
forecast_points["date"] = pd.to_datetime(forecast_points["date"])
forecast_points["source"] = "Forecast"

chart_df = pd.concat([history, forecast_points], ignore_index=True)
chart_df.set_index("date", inplace=True)
chart_df.sort_index(inplace=True)

col_chart, col_table = st.columns((2, 1))

title_metric = selected_metric.replace("_", " ").title()
with col_chart:
    st.subheader(f"{title_metric} Forecast")
    st.line_chart(chart_df, y="value", color="source")

with col_table:
    st.subheader("Forecast Table")
    st.dataframe(
        forecast_df.set_index("date"),
        use_container_width=True,
        column_config={
            "point": st.column_config.NumberColumn("Point", format="%0.2f"),
            "lower": st.column_config.NumberColumn("Lower", format="%0.2f"),
            "upper": st.column_config.NumberColumn("Upper", format="%0.2f"),
        },
    )

st.divider()

st.subheader("Ask the Agent")
with st.form("question_form", clear_on_submit=False):
    question = st.text_input("Question", placeholder="e.g. Are the forecasted results positive?")
    submitted = st.form_submit_button("Get answer")

if submitted and question:
    answer = agent.answer_question(question, metric=selected_metric)
    st.success(answer)
elif submitted:
    st.warning("Please enter a question.")

st.caption(
    "Assumptions: synthetic SaaS data, deterministic Holt-Winters forecasts, correlation-based explanations."
)

"""Forecast agent that combines forecasting and Q&A."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import pandas as pd

from forecast_agent.data_loader import load_saas_metrics
from forecast_agent.explain import ExplainabilityEngine
from forecast_agent.forecasting import ForecastResult, MetricForecaster
from forecast_agent.llm import get_llm_client


PRIMARY_METRIC = "monthly_recurring_revenue"
DOMAIN_KEYWORDS = {
    "forecast",
    "projection",
    "predict",
    "prediction",
    "trend",
    "increase",
    "decrease",
    "positive",
    "negative",
    "revenue",
    "customer",
    "customers",
    "marketing",
    "churn",
    "signup",
    "trial",
    "saas",
    "metric",
    "growth",
    "explain",
    "why",
    "confidence",
    "interval",
}

OUT_OF_SCOPE_MESSAGE = (
    "I can only answer questions about the SaaS metrics and forecasts in this dashboard. "
    "Try asking about revenue, customers, churn, or similar topics."
)


@dataclass
class AgentConfig:
    data_path: str = "saas_metrics.csv"
    metrics: Iterable[str] | None = None
    horizon: int = 12


class ForecastAgent:
    """Agent that combines deterministic forecasts with LLM-based answers."""

    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig()
        self.data = load_saas_metrics(self.config.data_path)
        self.metrics = list(self.config.metrics) if self.config.metrics else list(self.data.columns)
        if PRIMARY_METRIC not in self.metrics and PRIMARY_METRIC in self.data.columns:
            self.metrics.append(PRIMARY_METRIC)

        self.forecaster = MetricForecaster(self.data, self.metrics, horizon=self.config.horizon)
        self.forecasts: Dict[str, ForecastResult] = self.forecaster.forecast_all()
        self.explainer = ExplainabilityEngine(self.data)
        self.llm_client = get_llm_client()

    def get_forecast(self, metric: str | None = None) -> pd.DataFrame:
        metric = metric or PRIMARY_METRIC
        if metric not in self.forecasts:
            raise KeyError(f"Metric '{metric}' not found in forecast outputs")
        return self.forecasts[metric].forecast

    def answer_question(self, question: str, metric: Optional[str] = None) -> str:
        question_lower = question.lower()
        explicit_metric = metric is not None

        classification = self._classify_question(question)
        if classification is not None and not classification.get("is_domain", False):
            return OUT_OF_SCOPE_MESSAGE

        target_metric = metric or self._infer_metric(question_lower)
        if not explicit_metric and classification is not None:
            suggested_metric = classification.get("metric")
            if isinstance(suggested_metric, str):
                normalized = suggested_metric.strip().lower().replace(" ", "_")
                if normalized in self.metrics:
                    target_metric = normalized

        if not self._matches_domain(question_lower, target_metric, explicit_metric):
            return OUT_OF_SCOPE_MESSAGE

        forecast_df = self.get_forecast(target_metric)

        llm_answer = self._try_llm_answer(question, target_metric, forecast_df)
        if llm_answer:
            return llm_answer

        if "why" in question_lower:
            narrative = self.explainer.describe_trend(target_metric, forecast_df)
            drivers = " | ".join(narrative.drivers) if narrative.drivers else "Drivers are inconclusive"
            return (
                f"Forecast explanation for {target_metric}: {narrative.trend_description} "
                f"Key drivers: {drivers}."
            )

        if any(keyword in question_lower for keyword in ("positive", "good", "healthy")):
            return self._describe_positivity(target_metric, forecast_df)

        if any(keyword in question_lower for keyword in ("increase", "improve", "growth")):
            return self._describe_trend(target_metric, forecast_df)

        if "confidence" in question_lower or "range" in question_lower:
            last_row = forecast_df.iloc[-1]
            return (
                f"Latest forecast for {target_metric} is {last_row['point']:.2f} "
                f"with 95% interval [{last_row['lower']:.2f}, {last_row['upper']:.2f}]."
            )

        return self._summarise_forecast(target_metric, forecast_df)

    def _try_llm_answer(self, question: str, metric: str, forecast_df: pd.DataFrame) -> Optional[str]:
        client = self.llm_client
        if client is None or not client.available:
            return None

        context = self._build_llm_context(metric, forecast_df)
        answer = client.answer(question=question, context=context)
        if answer:
            sanitized = answer.strip()
            if sanitized:
                return sanitized
        return None

    def _build_llm_context(self, metric: str, forecast_df: pd.DataFrame) -> Dict[str, Any]:
        history = self.data[metric]
        latest_actual = float(history.iloc[-1])
        narrative = self.explainer.describe_trend(metric, forecast_df)
        positivity = self._describe_positivity(metric, forecast_df)
        trend = self._describe_trend(metric, forecast_df)
        last_row = forecast_df.iloc[-1]
        confidence_text = (
            f"Last month interval: [{last_row['lower']:.2f}, {last_row['upper']:.2f}] around {last_row['point']:.2f}."
        )
        forecast_records = []
        for row in forecast_df.to_dict(orient="records"):
            forecast_records.append(
                {
                    "date": str(row["date"]),
                    "point": float(row["point"]),
                    "lower": float(row["lower"]),
                    "upper": float(row["upper"]),
                }
            )

        return {
            "metric": metric,
            "history_start": history.index[0].strftime("%Y-%m-%d"),
            "history_end": history.index[-1].strftime("%Y-%m-%d"),
            "latest_actual": latest_actual,
            "forecast_horizon_months": len(forecast_records),
            "forecast_summary": {
                "trend": trend,
                "positivity": positivity,
                "confidence": confidence_text,
            },
            "drivers": narrative.drivers,
            "driver_summary": narrative.trend_description,
            "forecast_table": forecast_records,
            "dataset_note": (
                "Synthetic SaaS metrics for demonstration; numbers are illustrative. "
                "Follow instructions to label fictional assumptions."
            ),
        }

    def _classify_question(self, question: str) -> Optional[Dict[str, Any]]:
        client = self.llm_client
        if client is None or not client.available:
            return None
        return client.classify(question)

    def _infer_metric(self, question_lower: str) -> str:
        for metric in self.metrics:
            metric_phrase = metric.replace("_", " ")
            if metric in question_lower or metric_phrase in question_lower:
                return metric
        return PRIMARY_METRIC if PRIMARY_METRIC in self.metrics else self.metrics[0]

    def _matches_domain(self, question_lower: str, metric: str, explicit_metric: bool) -> bool:
        if explicit_metric:
            return True

        metric_tokens = metric.replace("_", " ").split()
        if any(token in question_lower for token in metric_tokens):
            return True

        if any(keyword in question_lower for keyword in DOMAIN_KEYWORDS):
            return True

        return False

    def _describe_positivity(self, metric: str, forecast_df: pd.DataFrame) -> str:
        min_point = forecast_df["point"].min()
        above_zero = min_point > 0
        avg_forecast = forecast_df["point"].mean()
        status = "remain positive" if above_zero else "dip below zero"
        return (
            f"Forecasted {metric} values {status}. Minimum projected value is {min_point:.2f} "
            f"and average across the horizon is {avg_forecast:.2f}."
        )

    def _describe_trend(self, metric: str, forecast_df: pd.DataFrame) -> str:
        first = forecast_df["point"].iloc[0]
        last = forecast_df["point"].iloc[-1]
        change = last - first
        pct_change = (change / first) * 100 if first != 0 else float("nan")
        direction = "increase" if change > 0 else "decrease" if change < 0 else "remain flat"
        pct_text = f" ({pct_change:.1f}% change)" if pd.notna(pct_change) else ""
        return f"{metric} is projected to {direction}{pct_text} over the next {len(forecast_df)} months."

    def _summarise_forecast(self, metric: str, forecast_df: pd.DataFrame) -> str:
        first_month = forecast_df.iloc[0]
        last_month = forecast_df.iloc[-1]
        return (
            f"Forecast for {metric}: starts at {first_month['point']:.2f} "
            f"and ends at {last_month['point']:.2f}. Interval for last point: "
            f"[{last_month['lower']:.2f}, {last_month['upper']:.2f}]."
        )

    def export_forecasts(self) -> Dict[str, pd.DataFrame]:
        return {metric: result.forecast for metric, result in self.forecasts.items()}

"""Explanation utilities for forecast outputs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


@dataclass
class MetricNarrative:
    metric: str
    trend_description: str
    drivers: List[str]


class ExplainabilityEngine:
    """Generate simple narratives about forecast behaviour."""

    def __init__(self, history: pd.DataFrame):
        self.history = history

    def describe_trend(self, metric: str, forecast_df: pd.DataFrame) -> MetricNarrative:
        history_series = self.history[metric]
        trend = self._describe_trend(history_series, forecast_df)
        drivers = self._top_drivers(metric)
        return MetricNarrative(metric=metric, trend_description=trend, drivers=drivers)

    def _describe_trend(self, history: pd.Series, forecast_df: pd.DataFrame) -> str:
        model = LinearRegression()
        index = np.arange(len(history)).reshape(-1, 1)
        model.fit(index, history.values)
        slope = model.coef_[0]

        change = forecast_df["point"].iloc[-1] - history.iloc[-1]
        pct_change = (change / history.iloc[-1]) * 100 if history.iloc[-1] != 0 else np.nan

        trend_direction = "stable"
        if abs(slope) > 0.001:
            trend_direction = "rising" if slope > 0 else "softening"

        forecast_direction = "higher" if change > 0 else "lower"
        if abs(change) < 1e-6:
            forecast_direction = "flat compared to"

        pct_text = f"({pct_change:.1f}% change)" if np.isfinite(pct_change) else ""
        return (
            f"Historical trend is {trend_direction}. Forecast points to {forecast_direction} "
            f"the latest actual {pct_text}."
        )

    def _top_drivers(self, metric: str, top_n: int = 3) -> List[str]:
        df = self.history
        if metric not in df.columns:
            return []

        features = df.drop(columns=[metric])
        if features.empty:
            return []

        correlations = features.corrwith(df[metric]).dropna()
        if correlations.empty:
            return []

        ranked = correlations.abs().sort_values(ascending=False).head(top_n)
        narratives: List[str] = []
        for feature in ranked.index:
            corr = correlations[feature]
            direction = "moves with" if corr > 0 else "moves against"
            narratives.append(f"{feature} {direction} {metric} (corr={corr:.2f})")
        return narratives

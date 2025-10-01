"""Forecasting utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing


@dataclass(frozen=True)
class ForecastResult:
    metric: str
    forecast: pd.DataFrame


class MetricForecaster:
    """Create forecasts for a set of numeric metrics."""

    def __init__(self, history: pd.DataFrame, metrics: Iterable[str], horizon: int = 12):
        self.history = history
        self.metrics = list(metrics)
        self.horizon = horizon
        self._models: Dict[str, object] = {}

    def forecast_all(self) -> Dict[str, ForecastResult]:
        """Forecast all configured metrics."""
        results: Dict[str, ForecastResult] = {}
        for metric in self.metrics:
            forecast_df = self._forecast_metric(metric)
            results[metric] = ForecastResult(metric=metric, forecast=forecast_df)
        return results

    def _forecast_metric(self, metric: str) -> pd.DataFrame:
        series = self.history[metric]
        future_index = pd.date_range(
            start=series.index[-1] + pd.offsets.MonthBegin(),
            periods=self.horizon,
            freq="MS",
        )

        # Try a seasonal exponential smoothing model first, fall back to linear regression if needed.
        try:
            model = ExponentialSmoothing(
                series,
                trend="add",
                seasonal="add",
                seasonal_periods=12,
            )
            fitted = model.fit(optimized=True)
            forecast_values = fitted.forecast(self.horizon)
            fitted_values = fitted.fittedvalues
            self._models[metric] = fitted
        except (ValueError, np.linalg.LinAlgError):
            forecast_values, fitted_values = self._linear_regression_forecast(series)

        residuals = series - fitted_values
        residual_std = residuals.std(ddof=1)
        interval = 1.96 * residual_std if np.isfinite(residual_std) else 0.0

        forecast_df = pd.DataFrame(
            {
                "date": future_index,
                "point": forecast_values.values,
                "lower": np.maximum(forecast_values.values - interval, 0.0),
                "upper": forecast_values.values + interval,
            }
        )
        return forecast_df

    def _linear_regression_forecast(self, series: pd.Series):
        index = np.arange(len(series)).reshape(-1, 1)
        model = LinearRegression().fit(index, series.values)
        future_index = np.arange(len(series), len(series) + self.horizon).reshape(-1, 1)
        forecast_values = model.predict(future_index)
        fitted_values = pd.Series(model.predict(index), index=series.index)
        # Pad future values to align with date index creation in caller
        future_dates = pd.date_range(
            start=series.index[-1] + pd.offsets.MonthBegin(),
            periods=self.horizon,
            freq="MS",
        )
        forecast_series = pd.Series(forecast_values, index=future_dates)
        return forecast_series, fitted_values

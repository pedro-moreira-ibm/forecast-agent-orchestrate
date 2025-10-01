"""FastAPI service exposing the forecasting agent for Watson Orchestrate."""
from __future__ import annotations

import logging
import os
import threading
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

from .agent import AgentConfig, ForecastAgent, PRIMARY_METRIC

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Forecast Agent Skill",
    version="0.1.0",
    description=(
        "REST interface for the SaaS forecasting agent so it can be wired into "
        "IBM Watson Orchestrate as a custom skill."
    ),
)

_AGENT_LOCK = threading.Lock()
_AGENT_INSTANCE: ForecastAgent | None = None


def _get_api_key() -> str:
    return os.getenv("FORECAST_AGENT_API_KEY", "").strip()


def _get_dashboard_url() -> str:
    return os.getenv("FORECAST_DASHBOARD_URL", "").strip()


def _build_agent_config() -> AgentConfig:
    base = AgentConfig()
    data_path = os.getenv("FORECAST_DATA_PATH") or base.data_path
    horizon_raw = os.getenv("FORECAST_HORIZON")
    horizon = base.horizon
    if horizon_raw:
        try:
            horizon_candidate = int(horizon_raw)
            if horizon_candidate > 0:
                horizon = horizon_candidate
            else:
                logger.warning(
                    "FORECAST_HORIZON must be positive. Received %s; keeping %s.",
                    horizon_raw,
                    base.horizon,
                )
        except ValueError:
            logger.warning(
                "Unable to parse FORECAST_HORIZON=%s. Falling back to %s.",
                horizon_raw,
                base.horizon,
            )
    return AgentConfig(data_path=data_path, metrics=base.metrics, horizon=horizon)


def _get_agent() -> ForecastAgent:
    global _AGENT_INSTANCE
    if _AGENT_INSTANCE is None:
        with _AGENT_LOCK:
            if _AGENT_INSTANCE is None:
                logger.info("Bootstrapping forecast agent instance for the API service")
                _AGENT_INSTANCE = ForecastAgent(_build_agent_config())
    return _AGENT_INSTANCE


class ForecastPoint(BaseModel):
    date: str
    point: float
    lower: float
    upper: float


class ForecastResponse(BaseModel):
    metric: str
    horizon: int
    points: list[ForecastPoint]


class MetricsResponse(BaseModel):
    metrics: list[str]
    primary_metric: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question about the forecasts")
    metric: Optional[str] = Field(default=None, description="Optional metric override")


class AskResponse(BaseModel):
    answer: str
    metric: str
    origin: str = "forecast-agent"


class DashboardResponse(BaseModel):
    configured: bool
    url: Optional[str] = None
    note: Optional[str] = None


class RefreshResponse(BaseModel):
    status: str
    horizon_months: int
    data_path: str


def enforce_api_key(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
) -> None:
    api_key = _get_api_key()
    if not api_key:
        return

    if x_api_key and x_api_key == api_key:
        return

    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token == api_key:
            return

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API key",
    )


@app.get("/health", dependencies=[Depends(enforce_api_key)])
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    dependencies=[Depends(enforce_api_key)],
)
def list_metrics(agent: ForecastAgent = Depends(_get_agent)) -> MetricsResponse:
    return MetricsResponse(metrics=list(agent.metrics), primary_metric=PRIMARY_METRIC)


@app.get(
    "/forecast/{metric}",
    response_model=ForecastResponse,
    dependencies=[Depends(enforce_api_key)],
)
def get_metric_forecast(metric: str, agent: ForecastAgent = Depends(_get_agent)) -> ForecastResponse:
    normalized = metric.strip().lower().replace(" ", "_")
    if normalized not in agent.forecasts:
        raise HTTPException(status_code=404, detail=f"Metric '{metric}' not found")

    forecast_df = agent.get_forecast(normalized)
    points = [
        ForecastPoint(
            date=str(row["date"]),
            point=float(row["point"]),
            lower=float(row["lower"]),
            upper=float(row["upper"]),
        )
        for row in forecast_df.to_dict(orient="records")
    ]

    return ForecastResponse(metric=normalized, horizon=len(points), points=points)


@app.post(
    "/ask",
    response_model=AskResponse,
    dependencies=[Depends(enforce_api_key)],
)
def ask(payload: AskRequest, agent: ForecastAgent = Depends(_get_agent)) -> AskResponse:
    metric = None
    if payload.metric:
        metric = payload.metric.strip().lower().replace(" ", "_")
        if metric not in agent.metrics:
            raise HTTPException(status_code=404, detail=f"Metric '{payload.metric}' not found")

    answer = agent.answer_question(payload.question, metric=metric)

    resolved_metric = metric or (PRIMARY_METRIC if PRIMARY_METRIC in agent.metrics else agent.metrics[0])
    return AskResponse(answer=answer, metric=resolved_metric)


@app.get(
    "/dashboard",
    response_model=DashboardResponse,
    dependencies=[Depends(enforce_api_key)],
)
def dashboard() -> DashboardResponse:
    url = _get_dashboard_url()
    if url:
        return DashboardResponse(configured=True, url=url)
    return DashboardResponse(
        configured=False,
        note="Set FORECAST_DASHBOARD_URL to expose a clickable Streamlit dashboard link.",
    )


@app.post(
    "/refresh",
    response_model=RefreshResponse,
    dependencies=[Depends(enforce_api_key)],
)
def refresh_agent() -> RefreshResponse:
    global _AGENT_INSTANCE
    config = _build_agent_config()
    with _AGENT_LOCK:
        _AGENT_INSTANCE = ForecastAgent(config)
    return RefreshResponse(status="refreshed", horizon_months=config.horizon, data_path=config.data_path)

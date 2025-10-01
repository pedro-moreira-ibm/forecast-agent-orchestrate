"""CLI entry points for the forecast agent."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import click
import pandas as pd

from .agent import AgentConfig, ForecastAgent, PRIMARY_METRIC


def _build_agent(data_path: Optional[str], horizon: int) -> ForecastAgent:
    base_config = AgentConfig()
    config = AgentConfig(
        data_path=data_path or base_config.data_path,
        metrics=base_config.metrics,
        horizon=horizon,
    )
    return ForecastAgent(config)


@click.group()
def cli() -> None:
    """Interact with the forecasting agent."""


@cli.command()
@click.argument("question", nargs=-1, required=True)
@click.option("--metric", default=None, help="Metric to focus on (defaults to primary metric).")
@click.option("--data-path", default=None, help="Path to the SaaS metrics CSV file.")
@click.option("--horizon", default=12, show_default=True, help="Forecast horizon in months.")
def ask(question: tuple[str, ...], metric: Optional[str], data_path: Optional[str], horizon: int) -> None:
    """Ask the agent a question about the forecast."""
    agent = _build_agent(data_path, horizon)
    joined_question = " ".join(question)
    answer = agent.answer_question(joined_question, metric=metric)
    click.echo(answer)


@cli.command()
@click.option("--metric", default=PRIMARY_METRIC, show_default=True, help="Metric to display.")
@click.option("--data-path", default=None, help="Path to the SaaS metrics CSV file.")
@click.option("--horizon", default=12, show_default=True, help="Forecast horizon in months.")
@click.option("--as-json", is_flag=True, default=False, help="Emit forecast as JSON.")
def forecast(metric: str, data_path: Optional[str], horizon: int, as_json: bool) -> None:
    """Print the forecast table for a metric."""
    agent = _build_agent(data_path, horizon)
    forecast_df = agent.get_forecast(metric)
    if as_json:
        click.echo(forecast_df.to_json(orient="records", date_format="iso"))
    else:
        click.echo(forecast_df.to_string(index=False, float_format="{:.2f}".format))


@cli.command()
@click.option("--data-path", default=None, help="Path to the SaaS metrics CSV file.")
@click.option("--horizon", default=12, show_default=True, help="Forecast horizon in months.")
@click.option("--output", type=click.Path(path_type=Path), default=None, help="Write all forecasts to a JSON file.")
def report(data_path: Optional[str], horizon: int, output: Optional[Path]) -> None:
    """Generate a report for all metrics."""
    agent = _build_agent(data_path, horizon)
    forecasts = {metric: df.to_dict(orient="records") for metric, df in agent.export_forecasts().items()}
    if output:
        output.write_text(json.dumps(forecasts, indent=2, default=str))
        click.echo(f"Saved forecasts to {output}")
    else:
        for metric, rows in forecasts.items():
            click.echo(f"\n=== {metric} ===")
            click.echo(pd.DataFrame(rows).to_string(index=False, float_format="{:.2f}".format))


if __name__ == "__main__":
    cli()

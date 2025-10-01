# Forecast Agent

A proof-of-concept forecasting agent that fits time-series models on SaaS metrics, produces 12-month projections, and answers natural-language questions (positivity, trends, explanations) about the forecasted values.

## Environment

The project uses Poetry for environment and dependency management. Python 3.12 is required.

```bash
poetry install
```

### Hugging Face access token (optional but recommended)

The dashboard can call the free Hugging Face router (`HuggingFaceTB/SmolLM3-3B`) to understand questions and craft answers backed by the local forecasts. Create a token on https://huggingface.co/settings/tokens (scope: **read**), then set it in your shell before launching the app:

```powershell
# PowerShell (current session)
$env:HF_TOKEN = "hf_your_token"

# Persist for future sessions
setx HF_TOKEN "hf_your_token"
```

If the token is absent, the agent falls back to the deterministic keyword guardrails and rule-based answers.

## CLI Usage

Interact via the CLI with the bundled `saas_metrics.csv` dataset (spanning Jan 2021 - Dec 2024):

```bash
# Display the monthly recurring revenue forecast
poetry run forecast-agent forecast --metric monthly_recurring_revenue

# Ask a natural language question
poetry run forecast-agent ask "Are the forecasted results positive?"

# Explain the forecast
poetry run forecast-agent ask "Why are the forecasted values like that?"

# Export forecasts for every tracked metric
poetry run forecast-agent report --output forecasts.json
```

## Dashboard (local and free)

Launch the Streamlit dashboard to explore charts and chat with the agent:

```bash
poetry run streamlit run src/forecast_agent/dashboard.py
```

Inside the app you can:

- Pick any metric and horizon.
- View historical vs. forecasted values on an interactive chart plus the forecast table.
- Submit natural-language questions; with an HF token the LLM will vet scope and compose answers using the supplied forecast context, otherwise the rule-based responses are used.

## How it works

- Historical data is read from `saas_metrics.csv` and coerced into a clean monthly time series.
- Each metric is modelled with Holt-Winters exponential smoothing; a linear-trend fallback is used if the seasonal model fails.
- Per-metric forecasts include simple 95% confidence bands derived from in-sample residuals.
- The Hugging Face model (when available) first classifies whether a question belongs to the SaaS forecasting scope and then crafts a concise answer using a JSON summary of the computed forecasts. Without the token, deterministic logic handles both steps.

## Assumptions

- The provided CSV is synthetic; forecasts and explanations are illustrative rather than production-grade.
- Seasonality is assumed to repeat annually (period=12). With only 48 observations, the agent cannot reliably capture long cycles or structural breaks.
- Explanations rely on historical correlations, not causal analysis. Correlated drivers may not reflect real-world causation.
- Forecast intervals are based on homoscedastic residuals; they will be overly optimistic if variability grows with scale.

## What you would need for reliable forecasts

1. **Longer history** - at least 3-5 years (60+ points) improves seasonal and trend estimates.
2. **Higher granularity** - weekly or daily series help catch short-term swings and enable hierarchical reconciliation.
3. **Segmentation** - cohorts (plan tier, region, acquisition channel) let you model heterogeneity and answer “why” with concrete segments.
4. **External signals** - marketing calendars, pricing changes, economic indicators give the models real drivers beyond autocorrelation.
5. **Event tracking** - labeled interventions (product launches, outages) allow the agent to explain structural shifts instead of attributing them to noise.

When you can supply the above data, the forecasting stack should graduate to richer models (SARIMAX with exogenous regressors, gradient-boosted trees on engineered features, causal impact analysis) and the explanation layer can promote feature importances, scenario deltas, or segment roll-ups with confidence.
## Watson Orchestrate integration

1. **Install & run the API service**
   - `poetry install`
   - Start the REST layer with `poetry run uvicorn forecast_agent.service:app --host 0.0.0.0 --port 8000 --env-file env.orchestrate.example` (copy the file and adjust if needed)
   - Expose the port that IBM Watson Orchestrate can reach (through your ingress, API gateway, or tunnel).
2. **Provide the required environment variables** before launching the service:
   - `FORECAST_AGENT_API_KEY` — set this to `u315UbYulLICUbw_ty6FNCQtXkBoaedth_qsHynNtqs3` so only your Watson Orchestrate instance can call the agent.
   - `FORECAST_DASHBOARD_URL` — public URL of the Streamlit dashboard you want the skill to surface. Share the link you plan to expose so I can double-check formatting if needed.
   - Optional overrides: `FORECAST_DATA_PATH` for a custom CSV location, `FORECAST_HORIZON` to change the number of projected months.
   - Edit `env.orchestrate.example` (or supply your own env file) so the API key and instance URL are loaded automatically when uvicorn starts.
3. **Register the skill** inside Orchestrate:
   - Upload `docs/orchestrate_skill.yaml` as a new custom skill definition.
   - Update the server URL in the dialog to the public host/port of your deployment.
   - Point the connection at your Watson Orchestrate instance: https://api.us-south.watson-orchestrate.cloud.ibm.com/instances/1d97775c-2501-47d1-9cba-1c77bd101291.
   - Store the API key secret in Orchestrate’s connection designer; it must match `FORECAST_AGENT_API_KEY`.
4. **Wire actions to phrases**
   - Map “ask” flows to the `POST /ask` operation so users can pose questions in chat.
   - Map report-style or card responses to `GET /forecast/{metric}` (returns the full horizon) and `GET /dashboard` for the dashboard link.
   - Use `POST /refresh` in admin recipes when you update the underlying CSV.
5. **Test the conversation**
   - In Orchestrate Studio, trigger phrases like “Show me the MRR forecast” or “Why is revenue trending up?” and ensure the responses render correctly.
   - Provide the dashboard URL as a button or link element in the response template.

The FastAPI app mirrors the CLI/dash logic, so anything you can ask locally will now be available to Orchestrate once the service is reachable.


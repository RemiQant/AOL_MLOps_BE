# Jakarta PM2.5 forecasting: MLOps backend

Forecasts tomorrow's daily PM2.5 for Jakarta and serves the results, model metrics and drift checks as a read-only JSON API. The design decisions are in `system-design.md`, and the task status is in `TODO.md`.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Linux/macOS: .venv/bin/python
python -m pytest tests                                    # 49 tests
```

## Pipeline

```bash
python -m src.train      # tests -> validate 3 models -> log to MLflow -> promote the champion
python -m src.monitor    # final-holdout evaluation + drift check -> artifacts/champion/monitoring.json
uvicorn src.api:app      # serve the API from artifacts/ (docs at /docs)
```

Training is separate from serving. The API only reads `artifacts/champion/*.json`, which are committed, and never retrains. MLflow runs are stored in `mlruns/mlflow.db` (git-ignored).

## Evaluation

- **Task:** forecast the next day's PM2.5 (µg/m³). The input at day *t* is the history up to and including day *t*.
- **Split (chronological):** the final **7 days (2026-02-12 to 2026-02-18) are a holdout** that is never used for fitting, tuning or selection (a test asserts it). Everything earlier is train and validation.
- **Validation:** walk-forward, one day ahead, over the last 90 days before the holdout. The same days are used for all models.
- **Models:** a naive baseline (yesterday's value), SARIMA (order chosen by AIC on pre-validation data), and a small LSTM (14-day window, fixed seed).
- **Metrics:** MAE and RMSE.
- **Promotion:** the lowest validation RMSE wins, but it must beat naive and the tests must pass. Otherwise `artifacts/champion/` is left untouched. The deployed version is the MLflow run ID in `champion.json`.
- **Holdout reporting:** reported separately from validation, in two clearly separate modes. Rolling one-step-ahead predicts each day from the actuals before it. Fixed-origin 7-day predicts all 7 days from the last training day.

Current validation RMSE (µg/m³): naive 16.05, SARIMA 15.21 (champion), LSTM 15.25. SARIMA and the LSTM are statistically tied, because the LSTM varies by about ±0.4 RMSE across seeds. See `GET /performance` for the live numbers.

## Monitoring (offline demonstration)

- **Drift:** two-sample KS test of the holdout against the training period, plus a mean shift. Drift is flagged only if KS p < 0.05 **and** the shift is at least 0.5 reference standard deviations. Only `pm2_5` is monitored, because the models are univariate.
- **Degradation:** champion holdout RMSE against its validation RMSE, flagged above a ratio of 1.5. This is reported separately from drift, since inputs can move without the errors growing, and the reverse.
- **Limits:** seven observations cannot support robust drift detection or model comparison, so the results are indicative only. `src/monitor.py` accepts any window length, so a longer rolling window works without changes. The data are CAMS model output, not ground sensors.

## API (read-only)

| Endpoint | Returns |
|---|---|
| `GET /health` | liveness |
| `GET /model` | active model, version (MLflow run ID), params, periods, dataset hash, git commit |
| `GET /forecast` | next-day forecast and the latest observation, with units |
| `GET /history?days=120` | recent observed PM2.5 |
| `GET /performance` | validation metrics for all models, and holdout metrics in both modes, kept separate |
| `GET /monitoring` | drift result, degradation check, daily holdout errors |

If `artifacts/` is missing, endpoints return 503 telling you which command to run.

**Environment variables** (all optional): `ARTIFACT_DIR` overrides the artifacts folder, `CORS_ORIGINS` is a comma-separated list of allowed frontend origins (default `*`), and `PORT` is used by the container (default 8000). No secrets are needed or committed.

## CI/CD and deployment

- **GitHub Actions** (`.github/workflows/ci.yml`) runs on pushes to `main` and on pull requests:
  1. `test`: install dependencies and run pytest. This includes `tests/test_artifacts.py`, which fails if the committed artifacts are stale against the dataset.
  2. `container`: build the Docker image, start it, and call every endpoint.
  3. `live`: after a push, if the repository variable `APP_URL` is set, wait until the deployed API serves `/forecast`.
- **Docker:** `docker build -t pm25-api . && docker run -p 8000:8000 pm25-api`. The image holds only the API, the dataset and the artifacts, with no torch or mlflow.
- **Deploying:** point any host that builds from a Dockerfile (Render, Railway, Fly.io, ...) at this repo's `main` branch, and require the `test` and `container` checks to pass before merging (branch protection). Then set `APP_URL`.
- **Updating the model:** run `python -m src.train` and `python -m src.monitor` locally, then commit the new `artifacts/champion/*.json`. The container job is the redeploy gate.

---

## Dataset: Air Quality Dataset for Jakarta

Covers 2022-08-01 to 2026-02-18. Air quality values are retrieved from the Open-Meteo Air Quality service using the cams_global domain, and city metadata is derived from the GeoNames city list. Author: Nitiraj Kulkarni.

- `dataset/city_info.csv`: city identifiers, coordinates, admin areas, and population.
- `dataset/air_quality_historical.csv`: the historical time series.
- `dataset/data_dictionary.csv`: each column, its unit, and its meaning.

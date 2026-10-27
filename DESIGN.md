# Project Metaprompt: Jakarta PM2.5 Forecasting — End-to-End MLOps

## 1. Role & Objective

Act as a senior Machine Learning Engineer and MLOps Engineer. Build a simple, production-minded end-to-end MLOps project for forecasting daily PM2.5 concentrations in Jakarta using the provided `air_quality_historical.csv` dataset.

The goal is to demonstrate the complete ML lifecycle: data preprocessing, time-series modeling, evaluation, experiment tracking, model versioning, automated testing, data drift monitoring, CI/CD, deployment, and a simple interactive website.

**Core principles:**
- Keep the architecture simple and maintainable.
- Prioritize correctness, reproducibility, and measurable model performance.
- Avoid unnecessary abstractions, microservices, and infrastructure.
- Implement incrementally, with clear checkpoints for review.
- Do not claim a feature works until it has been implemented and tested.

## 2. Inspect Before Implementing

Before writing code:
1. Inspect the existing repository, dataset, dependencies, and environment.
2. Inspect `air_quality_historical.csv` rather than assuming its schema.
3. Verify the timestamp column, date range, sampling frequency, missing values, duplicates, and feature distributions.
4. Confirm the PM2.5 target column and determine whether resampling or missing-value handling is necessary.
5. Create or update `system-design.md` with the architecture, decisions, project structure, and implementation checkpoints.
6. Create a concise `TODO.md` with testable acceptance criteria for each checkpoint.

Do not overwrite existing work unnecessarily. If a repository already exists, extend its current structure instead of rebuilding it from scratch.

## 3. Forecasting Task

Use daily PM2.5 forecasting as the primary task.

- Primary target: predict PM2.5 one day ahead.
- Dataset: `air_quality_historical.csv`.
- Preserve chronological ordering.
- Handle missing values without using future information.
- Prevent target leakage in preprocessing, feature engineering, and evaluation.
- Document the forecasting assumptions and prediction horizon.

If the dataset supports it, evaluate relevant exogenous variables such as PM10, ozone, temperature-related indicators, and other air-quality measurements. Only use features that would actually be available at the time of prediction.

Do not assume every available column should be used.

## 4. Models

Implement and compare three approaches:

1. **Naive baseline:** predict the next day's PM2.5 using the most recently observed PM2.5 value.
2. **SARIMA:** a statistical time-series model. Select a reasonable configuration using training/validation data only. Consider weekly seasonality only if supported by the data.
3. **LSTM:** a small PyTorch model using sliding windows of historical observations. Keep the architecture and tuning effort modest.

Use the same forecasting horizon and comparable evaluation periods for all models.

For the initial version, prioritize a reliable SARIMA implementation and baseline. Add LSTM after the evaluation pipeline works.

Do not assume LSTM will outperform SARIMA. Select the champion model based on validation results, not model complexity.

## 5. Data Splitting and Evaluation

Use chronological splitting, never random train/test splitting.

- Training and validation: all data before the final seven days.
- Final holdout: the last seven chronological observations, equivalent to `df.tail(7)` after sorting and validating the timestamps.
- Use a time-series validation method, preferably walk-forward validation, to compare models and choose hyperparameters.
- Do not use the final holdout to tune models, choose features, or select the champion.

Evaluate using:
- MAE
- RMSE

Compare all candidate models against the naive baseline. Record the evaluation period, prediction horizon, and metrics for every experiment.

For a one-day-ahead forecasting task, implement evaluation without using actual values from future dates as input to predictions that would have been made earlier. Clearly distinguish rolling one-step-ahead evaluation from a fixed-origin seven-day forecast.

Report the final seven-day holdout results separately from validation metrics. Acknowledge the limitations of drawing conclusions from only seven observations.

## 6. MLOps and Model Versioning

Use **MLflow** for experiment tracking and model versioning.

Track at minimum:
- Model type and hyperparameters
- Dataset identifier or dataset hash
- Training and validation periods
- Evaluation metrics
- Model artifact
- Code version or Git commit, when available

Use MLflow's model registry if supported by the chosen deployment environment; otherwise, maintain explicit, reproducible model versions through MLflow artifacts and metadata.

Implement a simple model promotion rule:
- A candidate must pass tests and beat the agreed baseline or satisfy a predefined performance threshold.
- A failed candidate must not replace the active model.
- The deployed model version must be identifiable.

Do not introduce a complex model-serving platform.

## 7. Data Drift and Performance Monitoring

Implement an offline monitoring experiment using:
- Reference data: the training period.
- Current/holdout data: the final seven observations.

Monitor:
1. Feature distribution changes.
2. PM2.5 target distribution changes, when meaningful.
3. Prediction error on the holdout after actual values are available.

Use Evidently if compatible with the project; otherwise, implement a small, documented statistical drift check.

Requirements:
- Use an appropriate numerical drift test or distribution metric.
- Document the drift method and threshold.
- Handle small sample sizes carefully.
- Clearly distinguish data drift from model performance degradation.
- Display drift results and holdout metrics in the dashboard.

Do not claim that seven observations provide statistically robust drift detection. Treat this as an offline monitoring demonstration. Design the monitoring code so a longer rolling window can be introduced later.

## 8. Application Architecture

Prefer a minimal Python-first stack:

- Python: data processing and model development.
- Pandas and NumPy: data processing.
- Statsmodels: SARIMA.
- PyTorch: LSTM.
- MLflow: experiment tracking and model artifacts.
- pytest: automated tests.
- Evidently or a lightweight statistical implementation: drift monitoring.
- Streamlit: interactive website.
- GitHub Actions: CI/CD.
- Docker: reproducible deployment, if useful for the selected hosting platform.

Keep the initial deployment as simple as possible. Use Streamlit as the primary application and avoid a separate API unless it provides a clear benefit.

## 9. Website Requirements

Create a clean, responsive, easy-to-understand dashboard with three sections.

### A. Forecast Dashboard
- Display the latest available PM2.5 observation and its timestamp.
- Plot historical PM2.5 values and forecasts.
- Clearly distinguish observed values from predictions.
- Display the forecast horizon and units.
- Show the active model version.

### B. Model Performance
- Compare Naive, SARIMA, and LSTM results.
- Display MAE and RMSE.
- Show validation metrics separately from final holdout metrics.
- Identify the champion model and explain the selection criterion.

### C. MLOps Monitoring
- Show data drift status and the method used.
- Display holdout prediction errors and performance metrics.
- Show the active model version and relevant experiment metadata.
- Include clear explanations of warnings and small-sample limitations.

Do not display fabricated, hardcoded performance metrics or monitoring results. All displayed values must come from actual pipeline outputs.

## 10. Automated Testing

Implement pytest tests for:

- Data loading and schema validation.
- Timestamp sorting, duplicates, and missing-value handling.
- Chronological split boundaries.
- Lag/window construction and prevention of target leakage.
- Model training and prediction output shapes.
- Metric calculations.
- Model artifact loading and version metadata.
- Drift monitoring behavior.
- Application/API smoke tests, as applicable.

Use small fixtures for unit tests. Do not retrain every model on the full dataset during every lightweight CI run unless required.

Add a test confirming that the final seven holdout observations are excluded from model fitting and model selection.

## 11. CI/CD and Deployment

Use GitHub Actions for a straightforward workflow.

### Continuous Integration
Trigger on pull requests and pushes:
- Set up the Python environment.
- Install dependencies.
- Run formatting/lint checks if configured.
- Run pytest.
- Run lightweight pipeline validation.

### Deployment
- Deploy only after required checks pass.
- Keep credentials and tokens in repository secrets or the hosting platform's secret manager.
- Never commit API keys, credentials, or private configuration.
- Document environment variables and deployment steps.
- Include a post-deployment smoke test when supported by the hosting platform.

Separate model training from application startup. Do not retrain the model every time a user opens the website.

## 12. Suggested Project Structure

Adapt this to the existing repository instead of enforcing it blindly.

```text
jakarta-pm25-forecast/
├── app/
│   └── streamlit_app.py
├── src/
│   ├── data.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── monitor.py
├── tests/
├── data/
│   └── README.md
├── artifacts/             # Only if appropriate for local development
├── mlruns/                # Local MLflow tracking, not necessarily committed
├── .github/
│   └── workflows/
│       └── ci.yml
├── requirements.txt
├── Dockerfile
├── system-design.md
├── TODO.md
└── README.md
```

Do not commit large generated model artifacts, datasets, MLflow run data, or secrets unless there is a deliberate reason. Document how to reproduce them.

## 13. Implementation Workflow

Work incrementally through these checkpoints:

**Checkpoint 1 — Data audit and design**
- Inspect the repository and dataset.
- Document the schema, assumptions, split strategy, architecture, and TODOs.

**Checkpoint 2 — Forecasting pipeline**
- Implement preprocessing, naive baseline, SARIMA, and evaluation.
- Verify chronological splitting and leakage prevention.

**Checkpoint 3 — LSTM**
- Implement a modest PyTorch LSTM.
- Compare it using the same validation protocol.

**Checkpoint 4 — MLflow**
- Log experiments, metrics, parameters, and model artifacts.
- Implement model version identification and promotion criteria.

**Checkpoint 5 — Monitoring**
- Implement the final seven-day holdout evaluation and offline drift analysis.
- Add tests for monitoring outputs.

**Checkpoint 6 — Website**
- Build the dashboard using actual model and monitoring artifacts.

**Checkpoint 7 — CI/CD and deployment**
- Configure GitHub Actions, deployment, smoke tests, and documentation.

At the end of each checkpoint:
1. Summarize what was implemented.
2. List the files changed.
3. Report tests run and their actual results.
4. Explain unresolved issues.
5. Wait for review and approval before proceeding to the next checkpoint.

Do not skip ahead or implement the entire project in one unreviewed change.

## 14. Definition of Done

The project is complete when:
- The forecasting pipeline is reproducible.
- SARIMA and LSTM are compared against a naive baseline.
- Validation and final holdout results are separated.
- The last seven observations are excluded from training and model selection.
- MLflow records model experiments and versions.
- Drift and prediction-performance monitoring are implemented and tested.
- The website displays real forecasting and monitoring outputs.
- CI tests run automatically through GitHub Actions.
- The application is deployed and accessible.
- README documents setup, architecture, evaluation, monitoring limitations, and deployment.

**Final instruction:** Be pragmatic. Prefer a working, tested, understandable MLOps system over a sophisticated but fragile architecture. Do not add technologies or features beyond what is needed to satisfy these requirements.
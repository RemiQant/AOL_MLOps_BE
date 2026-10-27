# System Design: Jakarta PM2.5 Forecasting

Spec: `DESIGN.md`. This file records what the data audit found and the decisions that follow from it.

## 1. Data audit (`dataset/air_quality_historical.csv`)

| Item | Finding |
|---|---|
| Rows / range | 1298 rows, 2022-08-01 → 2026-02-18, one row per day |
| Timestamp | `date` (YYYY-MM-DD), sorted, 0 duplicates, 0 missing days |
| Target | `pm2_5` (µg/m³), daily value |
| Missing values | The first 3 rows (2022-08-01..03) are empty in every column. There are no gaps after 2022-08-04, so 1295 usable days. `us_aqi` and `european_aqi` each have one extra NaN. |
| Source | Open-Meteo CAMS global model output, not ground sensors. Treat it as a modelled series. |
| Same-day corr with `pm2_5` | pm10 0.89, aod 0.77, ozone 0.70, no2 0.64, so2 0.56, co 0.51, dust 0.08, uv -0.20 |
| Autocorrelation | lag 1 = 0.82, lag 7 = 0.45 |
| Day-of-week means | 64.0 to 66.0 µg/m³, essentially flat. There is no evidence of weekly seasonality. |

Consequences:
- No resampling is needed, because the series is already daily and complete.
- Drop the 3 leading all-NaN rows. Any later NaN is forward-filled, which uses past values only.
- `us_aqi` and `european_aqi` are computed from PM2.5 and other pollutants, so they leak the target. **Do not use them.**
- The other pollutants are usable only as lagged features (≥ 1 day), because the forecast is made at the end of day *t* for day *t+1*.
- The seasonal period for SARIMA is chosen from validation data in checkpoint 2. The flat day-of-week means argue against s=7. An annual cycle (s=365) is too heavy for SARIMA, so use Fourier terms or skip it if it is needed.

## 2. Forecasting protocol

- Horizon: **1 day ahead**. Input at origin *t* is the history up to and including day *t*.
- Split (chronological, by position on the sorted usable frame):
  - `trainval` = everything except the last 7 days: 2022-08-04 → 2026-02-11 (1288 days).
  - `holdout` = last 7 days, `df.tail(7)`: 2026-02-12 → 2026-02-18.
- Validation: walk-forward, expanding window, one-step-ahead, over the last `V=90` days of `trainval`. Same days for all three models. Hyperparameters and the champion are chosen on this alone.
- Holdout reported separately, in two modes that are labelled apart:
  1. Rolling one-step-ahead: each day is predicted from actuals up to the day before. This is the primary holdout metric and matches the deployed use.
  2. Fixed-origin 7-day recursive forecast: all 7 days from the last `trainval` day with no new actuals. This is a secondary, harder number.
- Final refit: the champion is refit on `trainval` only (never on holdout) for the reported holdout metrics. The deployed forecast model for "tomorrow" may be refit on all data after selection. This is documented in the UI as a separate artifact version.
- Metrics: MAE, RMSE. Every model is compared with the naive baseline (ŷ(t+1) = y(t)).
- n=7 holdout: the numbers are anecdotal, not significant. The UI says so.

## 3. Models

| Model | Library | Notes |
|---|---|---|
| Naive | none | Last observed value |
| SARIMA | statsmodels | Small grid by AIC on the training part of each validation fold, or a single fit with filtered updates, to keep runtime low |
| LSTM | PyTorch | 1 layer, small hidden size, window of 14 days, scaler fit on the training window only, fixed seed |

The champion is the model with the lowest validation RMSE.

## 4. MLflow and promotion

- Local SQLite tracking (`mlruns/mlflow.db`, git-ignored). MLflow 3.17 refuses the file store. Each run logs: model type, params, dataset sha256, train/val/holdout periods, metrics, git commit, and a `model.json` artifact. That artifact is a recipe, not a weights file: params, periods and the next-day forecast. Training is deterministic (fixed seed) and takes seconds, so the recipe reproduces the model.
- The MLflow registry on a file store is deprecated in recent versions, so versions are identified by **run ID + a `champion.json`** (run ID, model type, dataset hash, git commit, metrics). Revisit if the deploy target supports a DB-backed registry.
- Promotion rule: a candidate replaces the active model only if tests pass **and** validation RMSE < naive validation RMSE. Otherwise `artifacts/champion/` is not touched.
- `artifacts/champion/` (model + `champion.json` + `metrics.json` + `drift.json`) is small and **is committed**, so the deployed app has real pipeline outputs without retraining at startup.

## 5. Monitoring (offline)

- Reference = `trainval`. Current = holdout (7 rows).
- Monitored column: `pm2_5` only. The models are univariate, so their one input (lagged PM2.5) is also the target. Other pollutants are not model inputs, so they are not monitored.
- Drift test: two-sample KS test (`scipy.stats.ks_2samp`), plus mean shift in reference standard deviations. The flag is raised if KS p < 0.05 **and** |shift| > 0.5σ. The two-part rule is there because with n=7 the KS test alone has little power and is noisy.
- Performance: champion holdout RMSE (rolling one-step) against its validation RMSE, flagged as degraded above a ratio of 1.5, shown separately from drift. Drift means the inputs moved. Degradation means errors rose. They can occur independently.
- Evidently is not used. It adds a heavy dependency for one KS test. `monitor.py` takes any current window, so a longer rolling window works without changes.

## 6. App and deploy

- Streamlit only, with three sections (Forecast, Model Performance, MLOps Monitoring), all read from `artifacts/champion/` and the dataset.
- Deploy: Streamlit Community Cloud from `main`, with the CI workflow required to pass. No Docker unless the host needs it.
- Smoke test: `streamlit.testing.v1.AppTest` in pytest, plus an HTTP check of the deployed URL.

## 7. Layout (extends the existing repo)

```
dataset/            existing, committed (small)
eda.ipynb           existing, untouched
src/{data,features,evaluate,train,predict,monitor}.py
app/streamlit_app.py
tests/
artifacts/champion/
.github/workflows/ci.yml
requirements.txt  TODO.md  system-design.md  README.md (extended later)
```

`src/` modules are added only when the checkpoint needs them. Models are added to `train.py` rather than split across files.

## 8. Environment

- Python 3.13.9 (Anaconda) already has pandas 2.3.3, numpy, scipy, statsmodels 0.14.5, torch 2.12.1, streamlit 1.51.0, pytest 8.4.2.
- **Missing: `mlflow`.** It is needed from checkpoint 4. The default `python` on PATH is 3.7 and cannot run the project, so use the Anaconda interpreter or a new venv.

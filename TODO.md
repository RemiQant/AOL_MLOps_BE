# TODO: checkpoints and acceptance criteria

Each checkpoint ends with a summary, the files changed, the real test results, and a wait for approval.

- [x] **1. Data audit and design.** `system-design.md` and `TODO.md` exist, and the findings come from the actual data.
- [x] **2. Forecasting pipeline** (`data`, `evaluate`, `train`: naive + SARIMA). `features.py` moved to checkpoint 3, where the LSTM windows need it.
  - [x] Loader validates the schema, sorts, rejects duplicate dates and gaps, drops leading NaN rows and forward-fills the rest.
  - [x] `tail(7)` holdout is excluded from fitting and selection (a test asserts it).
  - [x] Predictions for day *t* use only data ≤ *t*-1 (a test changes the future and asserts earlier predictions are unchanged).
  - [x] Walk-forward validation gives MAE/RMSE for naive and SARIMA over the same 90 days.
  - [x] Metric unit tests pass.
- [x] **3. LSTM.** It trains with a fixed seed, the scaler is fit on training data only, it is scored on the same validation days, and output shapes are tested.
- [ ] **4. MLflow.** Runs log all fields listed in `system-design.md` §4. The promotion rule is tested: a worse candidate leaves `artifacts/champion/` unchanged. `champion.json` identifies the version.
- [ ] **5. Monitoring.** The holdout is evaluated in both modes and reported apart from validation. `monitor.py` outputs the drift flag, KS statistic, p-value and mean shift. Tests cover a no-drift case, a shifted case, and a short window.
- [ ] **6. Website.** Three sections, with every number read from `artifacts/` or the dataset. The small-sample warning is shown. An `AppTest` smoke test passes.
- [ ] **7. CI/CD.** The workflow runs pytest and a lightweight pipeline check on push and PR, deploys only on a green `main`, and has a post-deploy smoke check. README documents setup, evaluation, limits and deployment. Nothing secret is committed.

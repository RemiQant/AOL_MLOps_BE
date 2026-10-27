import hashlib
import json
import subprocess
import sys
import warnings
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from statsmodels.tsa.statespace.sarimax import SARIMAX
from torch import nn

from src.data import DATA_PATH, TARGET, load, split
from src.evaluate import VAL_DAYS, validate
from src.features import make_windows

EXPERIMENT = "jakarta-pm25"

# (order, seasonal_order); s=7 is kept only so AIC can reject it (weekly means are flat)
SARIMA_GRID = [
    (o, s)
    for o in [(1, 0, 0), (2, 0, 0), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    for s in [(0, 0, 0, 0), (1, 0, 0, 7)]
]


def naive(series, start):
    return series.shift(1).iloc[start:]


def sarima(series, start):
    """Pick the order by AIC on series[:start], then filter the rest with fixed params."""
    train = series.iloc[:start]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # convergence noise from the small grid
        fits = {}
        for o, s in SARIMA_GRID:
            try:
                fits[(o, s)] = SARIMAX(train, order=o, seasonal_order=s).fit(disp=False)
            except np.linalg.LinAlgError:  # a candidate can fail to initialise; skip it
                continue
        order = min(fits, key=lambda k: fits[k].aic)  # fits is never empty in practice
        # apply() re-runs the filter on all data without re-estimating, so the
        # prediction at day t only sees actuals up to t-1.
        pred = fits[order].apply(series).get_prediction(start=start).predicted_mean
    pred.attrs["order"] = str(order)
    return pred


class _Net(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.out(self.lstm(x.unsqueeze(-1))[0][:, -1]).squeeze(-1)


def lstm(series, start, window=14, hidden=16, epochs=30, lr=1e-2, seed=0):
    """Small LSTM on a sliding window. Scaler and training windows use series[:start] only."""
    torch.manual_seed(seed)
    train = series.iloc[:start]
    scaled = (series - train.mean()) / train.std()
    X, y = make_windows(scaled.values, window)
    X, y = torch.from_numpy(X.copy()), torch.from_numpy(y)
    n_train = start - window  # windows whose target day is < start
    net = _Net(hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(epochs):
        for idx in torch.randperm(n_train).split(64):
            opt.zero_grad()
            nn.functional.mse_loss(net(X[idx]), y[idx]).backward()
            opt.step()
    with torch.no_grad():
        pred = net(X[n_train:]).numpy()
    out = pd.Series(pred * train.std() + train.mean(), index=series.index[start:])
    out.attrs = dict(window=window, hidden=hidden, epochs=epochs, lr=lr, seed=seed)
    return out


MODELS = {"naive": naive, "sarima": sarima, "lstm": lstm}


def run(df, days=VAL_DAYS):
    """Validation metrics per model, using the training portion only (holdout excluded)."""
    trainval, _ = split(df)
    return validate(MODELS, trainval[TARGET], days)


def forecast_next(model, series):
    """One-day-ahead forecast past the last observation, fit on all of `series`."""
    ext = series.reindex(pd.date_range(series.index[0], series.index[-1] + pd.Timedelta(days=1)))
    return model(ext, len(series))  # the NaN target day is never used as an input


def select(scores, tests_ok):
    """Best validation RMSE, promoted only if the tests passed and it beats the naive baseline."""
    best = min(scores, key=lambda m: scores[m]["RMSE"])
    beats_naive = scores[best]["RMSE"] < scores["naive"]["RMSE"]
    return best if tests_ok and beats_naive else None


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(path=DATA_PATH, out=Path("artifacts/champion"), mlruns=Path("mlruns"), days=VAL_DAYS, tests_ok=True):
    """Log every model to MLflow, then promote the champion to `out` if it passes the rule."""
    df = load(path)
    trainval, holdout = split(df)
    scores = run(df, days)
    meta = {
        "dataset_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "git_commit": _git_commit(),
        "horizon_days": 1,
        "train_period": [str(trainval.index[0].date()), str(trainval.index[-days - 1].date())],
        "validation_period": [str(trainval.index[-days].date()), str(trainval.index[-1].date())],
        "holdout_period": [str(holdout.index[0].date()), str(holdout.index[-1].date())],
        "forecast_trained_through": str(df.index[-1].date()),
    }
    mlruns = Path(mlruns).resolve()
    mlruns.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(mlruns / 'mlflow.db').as_posix()}")  # file store is deprecated
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(mlruns / "artifacts").as_uri())
    mlflow.set_experiment(EXPERIMENT)
    infos = {}
    for name, model in MODELS.items():
        pred = forecast_next(model, df[TARGET])
        with mlflow.start_run(run_name=name) as r:
            info = {**meta, "model": name, "params": dict(pred.attrs), "version": r.info.run_id,
                    "forecast_date": str(pred.index[0].date()), "forecast": float(pred.iloc[0]),
                    "validation_metrics": scores[name]}
            mlflow.log_params({"model": name, **info["params"], **{k: str(v) for k, v in meta.items()}})
            mlflow.log_metrics(scores[name])
            mlflow.log_dict(info, "model.json")
        infos[name] = info
    print(pd.DataFrame(scores).T.round(2))
    best = select(scores, tests_ok)
    if best is None:
        print("No candidate promoted; active model unchanged.")
        return None
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "champion.json").write_text(json.dumps({**infos[best], "all_validation_metrics": scores}, indent=2))
    print(f"Promoted {best}, version {infos[best]['version']}")
    return best


if __name__ == "__main__":
    ok = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"]).returncode == 0
    sys.exit(0 if main(tests_ok=ok) else 1)

import json
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

from src.data import DATA_PATH, TARGET, load, split
from src.evaluate import mae, rmse
from src.train import MODELS

# The models are univariate, so their only input (lagged PM2.5) is also the target.
DRIFT_COLS = [TARGET]
KS_ALPHA = 0.05  # drift needs a significant KS test ...
SHIFT_SD = 0.5  # ... and a mean shift of at least this many reference standard deviations
SMALL_SAMPLE = 30  # below this many current rows the result is only indicative
DEGRADED_RATIO = 1.5  # holdout RMSE / validation RMSE above this counts as degradation


def drift(reference, current):
    """Per-column drift of `current` against `reference` (any window length)."""
    if len(current) < 2:
        raise ValueError("need at least 2 current rows")
    cols = {}
    for c in DRIFT_COLS:
        ks = ks_2samp(reference[c], current[c])
        shift = (current[c].mean() - reference[c].mean()) / reference[c].std()
        cols[c] = {"ks_statistic": float(ks.statistic), "p_value": float(ks.pvalue),
                   "mean_shift_sd": float(shift),
                   "drifted": bool(ks.pvalue < KS_ALPHA and abs(shift) >= SHIFT_SD)}
    return {"method": "two-sample KS test + mean shift", "ks_alpha": KS_ALPHA, "shift_sd": SHIFT_SD,
            "n_reference": len(reference), "n_current": len(current),
            "small_sample": len(current) < SMALL_SAMPLE,
            "columns": cols, "drifted": any(v["drifted"] for v in cols.values())}


def holdout_predictions(df):
    """Predictions for the last 7 days from models fit on the rest, in two clearly separate modes.

    rolling_one_step: each day is predicted from actuals up to the day before.
    fixed_origin_7d: all 7 days predicted from the last training day, holdout actuals hidden.
    """
    y = df[TARGET]
    start = len(split(df)[0])
    hidden = y.copy()
    hidden.iloc[start:] = np.nan
    return {"rolling_one_step": {m: f(y, start) for m, f in MODELS.items()},
            "fixed_origin_7d": {m: f(hidden, start) for m, f in MODELS.items()}}


def report(df, champion):
    """Drift, holdout metrics and champion errors. `champion` is the parsed champion.json."""
    trainval, holdout = split(df)
    preds = holdout_predictions(df)
    actual = holdout[TARGET]
    metrics = {mode: {m: {"MAE": mae(actual, p), "RMSE": rmse(actual, p)} for m, p in by_model.items()}
               for mode, by_model in preds.items()}
    name = champion["model"]
    daily = [{"date": str(d.date()), "actual": float(a),
              **{f"{mode}_{k}": float(v) for mode, by_model in preds.items()
                 for k, v in (("prediction", by_model[name][d]), ("error", by_model[name][d] - a))}}
             for d, a in actual.items()]
    ratio = metrics["rolling_one_step"][name]["RMSE"] / champion["validation_metrics"]["RMSE"]
    return {
        "champion": {"model": name, "version": champion["version"]},
        "reference_period": [str(trainval.index[0].date()), str(trainval.index[-1].date())],
        "holdout_period": [str(holdout.index[0].date()), str(holdout.index[-1].date())],
        "drift": drift(trainval, holdout),
        "holdout_metrics": metrics,
        "champion_daily": daily,
        "performance": {"validation_rmse": champion["validation_metrics"]["RMSE"],
                        "holdout_rmse": metrics["rolling_one_step"][name]["RMSE"],
                        "ratio": ratio, "degraded_ratio_threshold": DEGRADED_RATIO,
                        "degraded": bool(ratio > DEGRADED_RATIO)},
    }


def main(path=DATA_PATH, out=Path("artifacts/champion")):
    out = Path(out)
    champion = json.loads((out / "champion.json").read_text())
    result = report(load(path), champion)
    (out / "monitoring.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    r = main()
    print(json.dumps({k: r[k] for k in ("drift", "holdout_metrics", "performance")}, indent=2))

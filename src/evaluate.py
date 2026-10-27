import numpy as np

VAL_DAYS = 90


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def validate(models, series, days=VAL_DAYS):
    """Walk-forward one-step-ahead scores over the last `days` of `series`.

    A model is `f(series, start) -> predictions for series.index[start:]`. It may fit
    on series[:start] only; the prediction for day t may use actuals up to t-1.
    """
    start = len(series) - days
    actual = series.iloc[start:]
    out = {}
    for name, model in models.items():
        pred = model(series, start)
        assert pred.index.equals(actual.index), f"{name}: prediction index mismatch"
        out[name] = {"MAE": mae(actual, pred), "RMSE": rmse(actual, pred)}
    return out

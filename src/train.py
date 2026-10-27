import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.data import TARGET, load, split
from src.evaluate import VAL_DAYS, validate

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
    pred.attrs["order"] = order
    return pred


MODELS = {"naive": naive, "sarima": sarima}


def run(df, days=VAL_DAYS):
    """Validation metrics per model, using the training portion only (holdout excluded)."""
    trainval, _ = split(df)
    return validate(MODELS, trainval[TARGET], days)


if __name__ == "__main__":
    df = load()
    print(pd.DataFrame(run(df)).T.round(2))
    trainval, _ = split(df)
    print("sarima order:", sarima(trainval[TARGET], len(trainval) - VAL_DAYS).attrs["order"])

import warnings

import numpy as np
import pandas as pd
import torch
from statsmodels.tsa.statespace.sarimax import SARIMAX
from torch import nn

from src.data import TARGET, load, split
from src.evaluate import VAL_DAYS, validate
from src.features import make_windows

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
    return pd.Series(pred * train.std() + train.mean(), index=series.index[start:])


MODELS = {"naive": naive, "sarima": sarima, "lstm": lstm}


def run(df, days=VAL_DAYS):
    """Validation metrics per model, using the training portion only (holdout excluded)."""
    trainval, _ = split(df)
    return validate(MODELS, trainval[TARGET], days)


if __name__ == "__main__":
    df = load()
    print(pd.DataFrame(run(df)).T.round(2))
    trainval, _ = split(df)
    print("sarima order:", sarima(trainval[TARGET], len(trainval) - VAL_DAYS).attrs["order"])

import numpy as np
import pandas as pd
import pytest

from src.evaluate import mae, rmse, validate
from src.train import MODELS, lstm, naive, run, sarima


def synthetic(n=150, seed=0):
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = 0.8 * y[i - 1] + rng.normal()
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({"pm2_5": y + 50}, index=idx)


def test_metrics():
    assert mae([1, 2, 3], [2, 2, 5]) == pytest.approx(1.0)
    assert rmse([0, 0], [3, 4]) == pytest.approx(np.sqrt(12.5))


def test_naive_is_previous_day():
    s = synthetic()["pm2_5"]
    pred = naive(s, 100)
    assert len(pred) == 50
    assert pred.iloc[0] == s.iloc[99] and pred.iloc[-1] == s.iloc[148]


def test_validate_shapes_and_keys():
    s = synthetic()["pm2_5"]
    out = validate(MODELS, s, days=30)
    assert set(out) == {"naive", "sarima", "lstm"}
    assert all(set(m) == {"MAE", "RMSE"} for m in out.values())


def test_lstm_output_shape_and_reproducible():
    s = synthetic()["pm2_5"]
    a, b = lstm(s, 100, epochs=3), lstm(s, 100, epochs=3)
    assert a.index.equals(s.index[100:]) and a.notna().all()
    pd.testing.assert_series_equal(a, b)


@pytest.mark.parametrize("model", [naive, sarima, lstm])
def test_no_future_leakage(model):
    """Changing actuals from day t onward must not change predictions for days <= t."""
    s = synthetic()["pm2_5"]
    start, k = 100, 10
    changed = s.copy()
    changed.iloc[start + k:] += 500
    a, b = model(s, start), model(changed, start)
    assert len(a) == 50
    pd.testing.assert_series_equal(a.iloc[: k + 1], b.iloc[: k + 1], check_exact=False, rtol=1e-6)


def test_holdout_excluded_from_fitting_and_selection():
    df = synthetic()
    poisoned = df.copy()
    poisoned.iloc[-7:] = 1e6  # absurd holdout values
    assert run(df, days=30) == run(poisoned, days=30)

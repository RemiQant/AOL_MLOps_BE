import json

import numpy as np
import pandas as pd
import pytest

from src.monitor import drift, holdout_predictions, main, report
from src.train import MODELS
from tests.test_evaluate import synthetic


def frames(ref_mean, cur_mean, n_ref=500, n_cur=7, seed=0):
    rng = np.random.default_rng(seed)
    return (pd.DataFrame({"pm2_5": rng.normal(ref_mean, 1, n_ref)}),
            pd.DataFrame({"pm2_5": rng.normal(cur_mean, 1, n_cur)}))


def test_no_drift_when_same_distribution():
    assert not drift(*frames(0, 0))["drifted"]


def test_drift_when_shifted():
    d = drift(*frames(0, 4))
    assert d["drifted"] and d["columns"]["pm2_5"]["mean_shift_sd"] > 3


def test_significant_but_tiny_shift_is_not_drift():
    d = drift(*frames(0, 0.2, n_ref=5000, n_cur=5000))
    assert d["columns"]["pm2_5"]["p_value"] < 0.05 and not d["drifted"]


def test_small_sample_flagged_and_too_short_rejected():
    ref, cur = frames(0, 0)
    assert drift(ref, cur)["small_sample"]
    assert not drift(ref, frames(0, 0, n_cur=60)[1])["small_sample"]
    with pytest.raises(ValueError):
        drift(ref, cur.iloc[:1])


def test_holdout_predictions_cover_7_days_in_both_modes():
    preds = holdout_predictions(synthetic())
    for by_model in preds.values():
        assert set(by_model) == set(MODELS) and all(len(p) == 7 for p in by_model.values())


def test_fixed_origin_ignores_holdout_actuals_but_rolling_does_not():
    df = synthetic()
    poisoned = df.copy()
    poisoned.iloc[-7:] += 100
    a, b = holdout_predictions(df), holdout_predictions(poisoned)
    for m in MODELS:
        pd.testing.assert_series_equal(a["fixed_origin_7d"][m], b["fixed_origin_7d"][m])
    assert not a["rolling_one_step"]["naive"].equals(b["rolling_one_step"]["naive"])


def test_naive_modes():
    df = synthetic()
    y, preds = df["pm2_5"], holdout_predictions(df)
    assert (preds["fixed_origin_7d"]["naive"] == y.iloc[-8]).all()  # last training day
    assert preds["rolling_one_step"]["naive"].tolist() == y.iloc[-8:-1].tolist()  # previous day


def test_report_and_main_write_real_outputs(tmp_path):
    df = synthetic()
    df.index.name = "date"
    csv = tmp_path / "d.csv"
    df.reset_index().to_csv(csv, index=False)
    champ = {"model": "naive", "version": "v1", "validation_metrics": {"RMSE": 1.0}}
    (tmp_path / "champion.json").write_text(json.dumps(champ))
    r = main(csv, tmp_path)
    assert json.loads((tmp_path / "monitoring.json").read_text())["champion"] == {"model": "naive", "version": "v1"}
    assert r["reference_period"][1] < r["holdout_period"][0]  # reference excludes the holdout
    assert r["drift"]["n_reference"] == len(df) - 7 and r["drift"]["n_current"] == 7
    assert len(r["champion_daily"]) == 7
    assert r["performance"]["degraded"] == (r["performance"]["ratio"] > 1.5)


def test_degradation_flag_is_separate_from_drift():
    df = synthetic()
    low = report(df, {"model": "naive", "version": "v", "validation_metrics": {"RMSE": 1e-6}})
    assert low["performance"]["degraded"]  # errors far above validation
    assert isinstance(low["drift"]["drifted"], bool)  # independent decision

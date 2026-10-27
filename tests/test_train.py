import json

import mlflow
import pandas as pd
import pytest

from src.train import MODELS, forecast_next, main, select
from tests.test_evaluate import synthetic


def scores(naive, sarima, lstm):
    return {m: {"RMSE": r} for m, r in zip(["naive", "sarima", "lstm"], [naive, sarima, lstm])}


def test_select_picks_lowest_rmse_that_beats_naive():
    assert select(scores(10, 9, 8), tests_ok=True) == "lstm"
    assert select(scores(10, 9, 11), tests_ok=True) == "sarima"


def test_select_rejects_failed_tests_or_no_gain_over_naive():
    assert select(scores(10, 9, 8), tests_ok=False) is None
    assert select(scores(8, 9, 10), tests_ok=True) is None  # naive is best
    assert select(scores(8, 8, 9), tests_ok=True) is None  # a tie is not an improvement


@pytest.mark.parametrize("name", list(MODELS))
def test_forecast_next_is_one_value_on_the_next_day(name):
    s = synthetic()["pm2_5"]
    pred = forecast_next(MODELS[name], s)
    assert len(pred) == 1 and pred.notna().all()
    assert pred.index[0] == s.index[-1] + pd.Timedelta(days=1)


def test_naive_forecast_is_last_observation():
    s = synthetic()["pm2_5"]
    assert forecast_next(MODELS["naive"], s).iloc[0] == s.iloc[-1]


@pytest.fixture
def csv(tmp_path):
    df = synthetic()
    df.index.name = "date"
    p = tmp_path / "d.csv"
    df.reset_index().to_csv(p, index=False)
    return p


def test_promotion_writes_identifiable_version_and_logs_to_mlflow(csv, tmp_path):
    out, mlruns = tmp_path / "champ", tmp_path / "mlruns"
    best = main(csv, out, mlruns, days=30)
    info = json.loads((out / "champion.json").read_text())
    assert info["model"] == best
    assert {"version", "dataset_sha256", "git_commit", "validation_period", "holdout_period",
            "params", "forecast", "validation_metrics", "all_validation_metrics"} <= set(info)
    run = mlflow.get_run(info["version"])  # the version is a real MLflow run
    assert run.data.metrics["RMSE"] == pytest.approx(info["validation_metrics"]["RMSE"])
    assert run.data.params["model"] == best


def test_failed_candidate_does_not_replace_active_model(csv, tmp_path):
    out = tmp_path / "champ"
    out.mkdir()
    (out / "champion.json").write_text('{"version": "active"}')
    assert main(csv, out, tmp_path / "mlruns", days=30, tests_ok=False) is None
    assert json.loads((out / "champion.json").read_text()) == {"version": "active"}


def test_holdout_dates_are_reported_and_after_validation(csv, tmp_path):
    out = tmp_path / "champ"
    main(csv, out, tmp_path / "mlruns", days=30)
    info = json.loads((out / "champion.json").read_text())
    assert info["validation_period"][1] < info["holdout_period"][0]

import json
from pathlib import Path

from src.data import dataset_hash, load, split

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "champion"
DATA = ROOT / "dataset" / "air_quality_historical.csv"


def test_committed_artifacts_are_consistent_with_code_and_data():
    champ = json.loads((ART / "champion.json").read_text())
    mon = json.loads((ART / "monitoring.json").read_text())
    scores = champ["all_validation_metrics"]
    df = load(DATA)
    _, holdout = split(df)

    assert champ["version"] and champ["git_commit"]
    assert scores[champ["model"]]["RMSE"] < scores["naive"]["RMSE"]  # promotion rule held
    assert champ["dataset_sha256"] == dataset_hash(DATA)  # artifacts are not stale
    assert champ["forecast_trained_through"] == str(df.index[-1].date())
    assert champ["holdout_period"] == [str(holdout.index[0].date()), str(holdout.index[-1].date())]
    assert mon["champion"] == {"model": champ["model"], "version": champ["version"]}
    assert mon["holdout_period"] == champ["holdout_period"]

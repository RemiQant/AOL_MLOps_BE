import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.data import load

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "artifacts" / "champion"
client = TestClient(app)


def real(name):
    return json.loads((REAL / name).read_text())


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_model_identifies_the_active_version():
    r = client.get("/model").json()
    c = real("champion.json")
    assert r["version"] == c["version"] and r["model"] == c["model"] and r["dataset_sha256"] == c["dataset_sha256"]


def test_forecast_matches_artifact_and_dataset():
    r = client.get("/forecast").json()
    c, df = real("champion.json"), load(ROOT / "dataset" / "air_quality_historical.csv")
    assert r["forecast"] == {"date": c["forecast_date"], "value": c["forecast"]}
    assert r["latest_observation"]["date"] == str(df.index[-1].date())
    assert r["unit"] == "µg/m³" and r["horizon_days"] == 1


def test_history_length_order_and_bounds():
    obs = client.get("/history", params={"days": 5}).json()["observations"]
    assert len(obs) == 5 and [o["date"] for o in obs] == sorted(o["date"] for o in obs)
    assert len(client.get("/history").json()["observations"]) == 120
    assert client.get("/history", params={"days": 0}).status_code == 422
    assert client.get("/history", params={"days": 99999}).status_code == 422


def test_performance_keeps_validation_and_holdout_separate():
    r = client.get("/performance").json()
    assert set(r["validation"]["metrics"]) == {"naive", "sarima", "lstm"}
    assert set(r["holdout"]["metrics"]) == {"rolling_one_step", "fixed_origin_7d"}
    assert r["holdout"]["n_observations"] == 7 and "7 observations" in r["holdout"]["caveat"]
    assert r["validation"]["period"][1] < r["holdout"]["period"][0]


def test_monitoring_separates_drift_from_performance():
    r = client.get("/monitoring").json()
    assert {"drifted", "small_sample", "method"} <= set(r["drift"])
    assert {"degraded", "ratio"} <= set(r["performance"])
    assert len(r["champion_daily"]) == 7


def test_responses_come_from_artifacts_not_hardcoded(tmp_path, monkeypatch):
    art = tmp_path / "a"
    shutil.copytree(REAL, art)
    c, m = real("champion.json"), real("monitoring.json")
    c.update(version="TESTVERSION", forecast=123.4)
    c["all_validation_metrics"]["naive"]["RMSE"] = 98.76
    m["drift"]["drifted"] = True
    (art / "champion.json").write_text(json.dumps(c))
    (art / "monitoring.json").write_text(json.dumps(m))
    monkeypatch.setenv("ARTIFACT_DIR", str(art))
    assert client.get("/model").json()["version"] == "TESTVERSION"
    assert client.get("/forecast").json()["forecast"]["value"] == 123.4
    assert client.get("/performance").json()["validation"]["metrics"]["naive"]["RMSE"] == 98.76
    assert client.get("/monitoring").json()["drift"]["drifted"] is True


@pytest.mark.parametrize("path", ["/model", "/forecast", "/performance", "/monitoring"])
def test_missing_artifacts_return_503_with_instructions(path, tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path))
    r = client.get(path)
    assert r.status_code == 503 and "python -m src.train" in r.json()["detail"]


def test_cors_allows_a_frontend_origin():
    r = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert r.headers["access-control-allow-origin"] == "*"

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.data import DATA_PATH, TARGET, load

ROOT = Path(__file__).resolve().parents[1]
UNIT = "µg/m³"
SELECTION_RULE = ("lowest validation RMSE, and it must beat the naive baseline with the tests passing; "
                  "the final holdout is never used for selection")
HOLDOUT_CAVEAT = "The holdout has only 7 observations: anecdotal, not statistically meaningful."

app = FastAPI(title="Jakarta PM2.5 forecast API")
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
                   allow_methods=["GET"])


def artifact(name):
    """Read a pipeline output on each request, so a retrained champion is served without a restart."""
    path = Path(os.environ.get("ARTIFACT_DIR", ROOT / "artifacts" / "champion")) / name
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        raise HTTPException(503, f"{name} not found; run `python -m src.train` and `python -m src.monitor`")


def dataset():
    return load(ROOT / DATA_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model")
def model():
    c = artifact("champion.json")
    keys = ["model", "version", "params", "horizon_days", "train_period", "validation_period",
            "holdout_period", "forecast_trained_through", "dataset_sha256", "git_commit"]
    return {k: c[k] for k in keys}


@app.get("/forecast")
def forecast():
    c, df = artifact("champion.json"), dataset()
    return {"unit": UNIT, "horizon_days": c["horizon_days"], "model": c["model"], "version": c["version"],
            "forecast": {"date": c["forecast_date"], "value": c["forecast"]},
            "latest_observation": {"date": str(df.index[-1].date()), "value": float(df[TARGET].iloc[-1])}}


@app.get("/history")
def history(days: int = Query(120, ge=1, le=2000)):
    s = dataset()[TARGET].tail(days)
    return {"unit": UNIT, "observations": [{"date": str(d.date()), "value": float(v)} for d, v in s.items()]}


@app.get("/performance")
def performance():
    c, m = artifact("champion.json"), artifact("monitoring.json")
    return {"unit": UNIT, "champion": c["model"], "selection_rule": SELECTION_RULE,
            "validation": {"period": c["validation_period"], "horizon_days": c["horizon_days"],
                           "mode": "walk-forward one-step-ahead", "metrics": c["all_validation_metrics"]},
            "holdout": {"period": m["holdout_period"], "n_observations": len(m["champion_daily"]),
                        "caveat": HOLDOUT_CAVEAT, "metrics": m["holdout_metrics"]}}


@app.get("/monitoring")
def monitoring():
    m = artifact("monitoring.json")
    keys = ["champion", "reference_period", "holdout_period", "drift", "performance", "champion_daily"]
    return {k: m[k] for k in keys}

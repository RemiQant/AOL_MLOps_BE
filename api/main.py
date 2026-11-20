import json
import os
from datetime import date as Date

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.core import (COMPARISON_PATH, DATA_PATH, MODEL_PATH, TEST_DAYS,
                      check_model, clean, predict_one, validate)

SERVED = "LightGBM + Optuna"
app = FastAPI(title="JakartaAir Forecast API", version="1.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("FRONTEND_ORIGINS", "http://localhost:5173").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
state = {}


def load():
    data = clean(pd.read_csv(DATA_PATH, parse_dates=["date"]))
    validate(data)
    comparison = json.loads(COMPARISON_PATH.read_text())
    # update paling akhir: kalau ada yang gagal di atas, state lama tetap utuh
    state.update(data=data, model=joblib.load(MODEL_PATH), comparison=comparison,
                 test_mae=comparison[SERVED]["mae"])


load()


@app.get("/health")
def health():
    d = state["data"]["date"]
    return {"status": "ok", "data_from": f"{d.min():%Y-%m-%d}", "data_until": f"{d.max():%Y-%m-%d}",
            "model": SERVED, "test_mae": state["test_mae"]}

@app.get("/predict")
def predict(date: Date | None = None):
    try:
        return predict_one(state["data"], state["model"], date)
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.get("/monitoring")
def monitoring(simulate_drift: bool = False):
    return check_model(state["data"], state["model"], state["test_mae"], simulate_drift)

@app.get("/models")
def models():
    rows = [{"name": name, **m} for name, m in state["comparison"].items()]
    return {"test_days": TEST_DAYS, "served": SERVED, "models": rows}


@app.post("/reload")
def reload():
    load()
    return {"status": "reloaded"}
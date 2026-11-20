from pathlib import Path
import os
import datetime

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/dataset/air_quality_historical.csv"
MODEL_PATH = ROOT / "notebooks/models/model.joblib"
COMPARISON_PATH = ROOT / "notebooks/models/comparison.json"

POLUTAN = ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide",
           "ozone", "aerosol_optical_depth", "dust", "uv_index"]
AQI = ["us_aqi", "european_aqi"]

BAD_DATES = ["2022-12-14", "2022-12-15"]
EXOG = ["pm10", "carbon_monoxide", "nitrogen_dioxide", "ozone", "aerosol_optical_depth"]
LAGS = [1, 2, 3, 4, 5, 7, 9, 14, 30]

TEST_DAYS, SEED, N_TRIALS, N_EPOCHS = 180, 42, 100, 30
MIN_IMPROVEMENT = 0.05
os.makedirs("models", exist_ok=True)

EXP_ID = f"{datetime.now():%m%d-%H%M}_lags{'-'.join(map(str, LAGS))}"
print("Eksperimen:", EXP_ID)
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]  # folder aol-mlops, dari mana pun dijalankan
DATA_PATH = ROOT / "data/dataset/air_quality_historical.csv"
MODEL_PATH = ROOT / "notebooks/models/model.joblib"
COMPARISON_PATH = ROOT / "notebooks/models/comparison.json"

POLUTAN = ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide",
           "ozone", "aerosol_optical_depth", "dust", "uv_index"]
AQI = ["us_aqi", "european_aqi"]  # dihitung dari PM2.5 -> hanya untuk validasi, bukan fitur
BAD_DATES = ["2022-12-14", "2022-12-15"]
EXOG = ["pm10", "carbon_monoxide", "nitrogen_dioxide", "ozone", "aerosol_optical_depth"]
LAGS = [1, 2, 3, 4, 5, 7, 9, 14, 30]
TEST_DAYS = 180


def clean(raw_df: pd.DataFrame) -> pd.DataFrame:
    data = raw_df.sort_values("date").drop_duplicates("date").dropna(subset=POLUTAN)
    data.loc[data["date"].isin(pd.to_datetime(BAD_DATES)), POLUTAN + AQI] = np.nan
    data = data.set_index("date").asfreq("D")
    data[POLUTAN + AQI] = data[POLUTAN + AQI].interpolate(limit=3, limit_area="inside")
    return data.reset_index()


def validate(data: pd.DataFrame) -> None:
    assert data["date"].is_unique, "ada tanggal dobel"
    assert (data["date"].diff().dropna() == pd.Timedelta(days=1)).all(), "ada tanggal bolong"
    assert data[POLUTAN].notna().all().all(), "polutan masih ada yang kosong"
    assert data["pm2_5"].between(5, 500).all(), "PM2.5 di luar rentang wajar (5-500)"
    rho = data["pm2_5"].corr(data["us_aqi"], method="spearman")
    assert rho > 0.7, f"PM2.5 dan US AQI tidak searah (Spearman {rho:.2f})"
    print(f"Validasi lulus: {len(data)} baris, {data['date'].min().date()} s.d. "
          f"{data['date'].max().date()} | Spearman PM2.5 vs US AQI = {rho:.2f}")


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    d = data.set_index("date")
    X = pd.DataFrame({"pm25_today": d["pm2_5"]})
    for lag in LAGS:
        X[f"pm25_lag{lag}"] = d["pm2_5"].shift(lag)
    for w in [3, 7]:
        X[f"pm25_mean{w}"] = d["pm2_5"].rolling(w).mean()
    X[EXOG] = d[EXOG]
    X["month_sin"] = np.sin(2 * np.pi * X.index.month / 12)
    X["month_cos"] = np.cos(2 * np.pi * X.index.month / 12)
    X["dayofweek"] = X.index.dayofweek
    return X
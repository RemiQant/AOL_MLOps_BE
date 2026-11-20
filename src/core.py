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
WINDOW_DAYS = 30
DRIFT_COLS = ["pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide", "ozone"]
MAE_RATIO_MAX, DRIFT_SHARE_MAX = 1.3, 0.5


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

# ---------- ISPU ----------
ISPU = [  # (batas atas, kategori, warna, saran) — PermenLHK P.14/2020
    (15.5, "Baik", "#16A34A", "Kualitas udara baik, aman untuk aktivitas luar ruangan."),
    (55.4, "Sedang", "#2563EB", "Kelompok sensitif sebaiknya mengurangi aktivitas berat di luar ruangan."),
    (150.4, "Tidak Sehat", "#EAB308", "Kurangi aktivitas luar ruangan dan gunakan masker."),
    (250.4, "Sangat Tidak Sehat", "#DC2626", "Hindari aktivitas luar ruangan, gunakan masker N95."),
    (float("inf"), "Berbahaya", "#111827", "Tetap di dalam ruangan dan tutup jendela."),
]


def ispu(pm25: float) -> dict:
    for upper, category, color, advice in ISPU:
        if pm25 <= upper:
            return {"category": category, "color": color, "advice": advice}


# ---------- Serving ----------
def predict_one(data: pd.DataFrame, model, date=None) -> dict:
    s = data.set_index("date")["pm2_5"]
    X_all = build_features(data).dropna()  # buang 30 hari awal yang lag-nya belum lengkap
    date = X_all.index.max() if date is None else pd.Timestamp(date)
    if date not in X_all.index:
        raise ValueError(f"Tanggal harus antara {X_all.index.min():%Y-%m-%d} dan {X_all.index.max():%Y-%m-%d}")

    X = X_all.loc[[date]]
    pm25 = float(model.predict(X)[0] + X["pm25_today"].iloc[0])  # model memprediksi selisih
    target = date + pd.Timedelta(days=1)
    actual = float(s[target]) if target in s.index else None
    actual_cat = ispu(actual)["category"] if actual is not None else None
    cat = ispu(pm25)
    history = s.loc[date - pd.Timedelta(days=WINDOW_DAYS - 1): date]

    return {
        "date": f"{date:%Y-%m-%d}", "target_date": f"{target:%Y-%m-%d}",
        "today_pm25": round(float(s[date]), 1), "pm25": round(pm25, 1), **cat,
        "actual": None if actual is None else round(actual, 1), "actual_category": actual_cat,
        "error": None if actual is None else round(abs(pm25 - actual), 1),
        "category_match": None if actual is None else cat["category"] == actual_cat,
        "history": [{"date": f"{d:%Y-%m-%d}", "pm25": round(float(v), 1)} for d, v in history.items()],
    }

def check_model(data: pd.DataFrame, model, test_mae: float, simulate_drift=False) -> dict:
    end = data["date"].max()
    start = end - pd.Timedelta(days=WINDOW_DAYS - 1)
    full = build_features(data).join(data.set_index("date")["pm2_5"].shift(-1).rename("y")).dropna()
    recent = full[full.index >= start]
    X = recent.drop(columns="y")
    recent_mae = mean_absolute_error(recent["y"], model.predict(X) + X["pm25_today"].values)

    current = data[data["date"] >= start].copy()
    if simulate_drift:
        current[DRIFT_COLS] *= 1.8
    one_year = pd.Timedelta(days=365)
    year_ago = data[data["date"].between(start - one_year, end - one_year)]
    drift = []
    for c in DRIFT_COLS:
        p = float(ks_2samp(year_ago[c], current[c]).pvalue)
        drift.append({"column": c, "p_value": round(p, 4), "drifted": p < 0.05})
    drift_share = sum(d["drifted"] for d in drift) / len(drift)
    mae_ratio = recent_mae / test_mae

    return {
        "checked_at": f"{end:%Y-%m-%d}", "window_days": WINDOW_DAYS,
        "recent_mae": round(float(recent_mae), 2), "test_mae": round(float(test_mae), 2),
        "mae_ratio": round(float(mae_ratio), 2), "drift": drift, "drift_share": drift_share,
        "retrain_needed": bool(mae_ratio > MAE_RATIO_MAX or drift_share > DRIFT_SHARE_MAX),
        "simulated": simulate_drift,
        "thresholds": {"mae_ratio": MAE_RATIO_MAX, "drift_share": DRIFT_SHARE_MAX},
    }
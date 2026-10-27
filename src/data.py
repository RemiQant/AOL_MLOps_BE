import pandas as pd

TARGET = "pm2_5"
HOLDOUT_DAYS = 7


def load(path="dataset/air_quality_historical.csv"):
    df = pd.read_csv(path, parse_dates=["date"])
    if TARGET not in df.columns:
        raise ValueError(f"missing column {TARGET}")
    if df["date"].duplicated().any():
        raise ValueError("duplicate dates")
    df = df.sort_values("date").set_index("date")
    df = df[df[TARGET].notna().cummax()]  # drop the leading all-NaN rows
    if (df.index.to_series().diff().dropna() != pd.Timedelta(days=1)).any():
        raise ValueError("dates are not contiguous daily")
    return df.asfreq("D").ffill()  # ffill uses past values only, so no leakage


def split(df, holdout=HOLDOUT_DAYS):
    return df.iloc[:-holdout], df.iloc[-holdout:]

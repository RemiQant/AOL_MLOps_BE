import pandas as pd
import pytest

from src.data import TARGET, load, split


def write(tmp_path, rows):
    p = tmp_path / "d.csv"
    pd.DataFrame(rows, columns=["date", TARGET, "pm10"]).to_csv(p, index=False)
    return p


def test_load_drops_leading_nan_sorts_and_ffills(tmp_path):
    p = write(tmp_path, [
        ("2024-01-03", 3.0, None),
        ("2024-01-01", None, None),
        ("2024-01-02", 2.0, 20.0),
        ("2024-01-04", 4.0, 40.0),
    ])
    df = load(p)
    assert list(df.index.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-03", "2024-01-04"]
    assert df.loc["2024-01-03", "pm10"] == 20.0  # filled from the past
    assert not df[TARGET].isna().any()


def test_load_rejects_duplicates(tmp_path):
    p = write(tmp_path, [("2024-01-01", 1.0, 1.0), ("2024-01-01", 2.0, 2.0)])
    with pytest.raises(ValueError, match="duplicate"):
        load(p)


def test_load_rejects_gaps(tmp_path):
    p = write(tmp_path, [("2024-01-01", 1.0, 1.0), ("2024-01-03", 2.0, 2.0)])
    with pytest.raises(ValueError, match="contiguous"):
        load(p)


def test_load_rejects_missing_target(tmp_path):
    p = tmp_path / "d.csv"
    pd.DataFrame({"date": ["2024-01-01"], "pm10": [1.0]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match=TARGET):
        load(p)


def test_split_is_chronological_and_disjoint():
    df = pd.DataFrame({TARGET: range(30)}, index=pd.date_range("2024-01-01", periods=30))
    trainval, holdout = split(df)
    assert len(holdout) == 7
    assert holdout.equals(df.tail(7))
    assert trainval.index.max() < holdout.index.min()
    assert len(trainval) + len(holdout) == len(df)


def test_real_dataset_loads():
    df = load()
    assert len(df) == 1295
    assert df.index[0] == pd.Timestamp("2022-08-04")
    assert df.index[-1] == pd.Timestamp("2026-02-18")

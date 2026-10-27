import json
import os
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import TARGET, load  # noqa: E402

ART = Path(os.environ.get("ARTIFACT_DIR", ROOT / "artifacts" / "champion"))
UNIT = "µg/m³"
SHOWN_DAYS = 120

st.set_page_config(page_title="Jakarta PM2.5 Forecast", layout="wide")
st.title("Jakarta PM2.5 forecast")

try:
    champ = json.loads((ART / "champion.json").read_text())
    mon = json.loads((ART / "monitoring.json").read_text())
except FileNotFoundError:
    st.error("Pipeline outputs not found. Run `python -m src.train` and `python -m src.monitor` first.")
    st.stop()

df = load(ROOT / "dataset" / "air_quality_historical.csv")
name, version = champ["model"], champ["version"]
last_date, last_value = df.index[-1], df[TARGET].iloc[-1]
try:
    dark = st.context.theme.type == "dark"
except Exception:  # theme info is unavailable outside a browser session
    dark = False
BLUE, ORANGE = ("#3987e5", "#d95926") if dark else ("#2a78d6", "#eb6834")  # validated pairs

st.caption("Daily mean PM2.5 from the Open-Meteo CAMS global model (modelled values, not ground sensors).")
tab_fc, tab_perf, tab_mon = st.tabs(["Forecast", "Model performance", "MLOps monitoring"])


def metrics_table(by_model):
    t = pd.DataFrame(by_model).T[["MAE", "RMSE"]].round(2)
    t.index = [f"{m} (champion)" if m == name else m for m in t.index]
    return t


with tab_fc:
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Latest observation ({last_date.date()})", f"{last_value:.1f} {UNIT}")
    c2.metric(f"Forecast for {champ['forecast_date']}", f"{champ['forecast']:.1f} {UNIT}")
    c3.metric("Active model", f"{name} · {version[:8]}")
    st.caption(f"Horizon: {champ['horizon_days']} day ahead, using data through {champ['forecast_trained_through']}. "
               f"Units: {UNIT}.")
    obs = df[TARGET].tail(SHOWN_DAYS).rename_axis("date").reset_index()
    obs["series"] = "Observed"
    fc = pd.DataFrame({"date": [last_date, pd.Timestamp(champ["forecast_date"])],
                       TARGET: [last_value, champ["forecast"]], "series": "Forecast"})
    fc["last"] = [False, True]
    long = pd.concat([obs, fc])
    domain = ["Observed", "Forecast"]
    base = alt.Chart(long).encode(
        x=alt.X("date:T", title=None),
        y=alt.Y(f"{TARGET}:Q", title=f"PM2.5 ({UNIT})"),
        color=alt.Color("series:N", scale=alt.Scale(domain=domain, range=[BLUE, ORANGE]),
                        legend=alt.Legend(title=None, orient="top")),
        strokeDash=alt.StrokeDash("series:N", scale=alt.Scale(domain=domain, range=[[1, 0], [4, 3]]), legend=None),
        tooltip=[alt.Tooltip("date:T"), alt.Tooltip(f"{TARGET}:Q", format=".1f"), "series:N"],
    )
    marker = base.mark_point(filled=True, size=90, shape="diamond").transform_filter(alt.datum["last"] == True)  # noqa: E712
    st.altair_chart(base.mark_line(strokeWidth=2) + marker, width="stretch")
    with st.expander("Table view"):
        st.dataframe(long.drop(columns="last", errors="ignore").sort_values("date", ascending=False), hide_index=True)

with tab_perf:
    vs, ve = champ["validation_period"]
    st.subheader("Validation")
    st.caption(f"Walk-forward, 1 day ahead, {vs} to {ve}. Models are chosen on this alone.")
    st.dataframe(metrics_table(champ["all_validation_metrics"]))
    st.info(f"**Champion: {name}.** Selection rule: lowest validation RMSE, and it must beat the naive "
            f"baseline (naive RMSE {champ['all_validation_metrics']['naive']['RMSE']:.2f}) with the tests passing.")
    hs, he = mon["holdout_period"]
    st.subheader("Final holdout")
    st.caption(f"{hs} to {he}. Never used for fitting or selection.")
    for mode, label in [("rolling_one_step", "Rolling one-step-ahead (each day predicted from the actuals before it)"),
                        ("fixed_origin_7d", "Fixed-origin 7-day forecast (all 7 days predicted from the last training day)")]:
        st.markdown(f"**{label}**")
        st.dataframe(metrics_table(mon["holdout_metrics"][mode]))
    st.warning("The holdout is only 7 observations. These numbers are anecdotal and cannot show that one model "
               "is better than another. A single spike can reorder the models.")

with tab_mon:
    d, p = mon["drift"], mon["performance"]
    col = d["columns"][TARGET]
    st.subheader("Data drift")
    (st.warning if d["drifted"] else st.success)(
        f"{'⚠️ Drift detected' if d['drifted'] else '✅ No drift detected'} in {TARGET} "
        f"({d['n_current']} current vs {d['n_reference']} reference days)")
    st.caption(f"Method: {d['method']}. Drift is flagged when KS p < {d['ks_alpha']} **and** the mean shift is at "
               f"least {d['shift_sd']} reference standard deviations. Reference: {mon['reference_period'][0]} to "
               f"{mon['reference_period'][1]}; current: the holdout.")
    k1, k2, k3 = st.columns(3)
    k1.metric("KS statistic", f"{col['ks_statistic']:.3f}")
    k2.metric("KS p-value", f"{col['p_value']:.3f}")
    k3.metric("Mean shift (ref. SD)", f"{col['mean_shift_sd']:+.2f}")
    if d["small_sample"]:
        st.warning(f"Only {d['n_current']} current observations: the KS test has little power at this size, so "
                   "this is an offline demonstration, not robust drift detection.")

    st.subheader("Model performance on the holdout")
    (st.warning if p["degraded"] else st.success)(
        f"{'⚠️ Degraded' if p['degraded'] else '✅ Not degraded'}: holdout RMSE {p['holdout_rmse']:.2f} vs "
        f"validation RMSE {p['validation_rmse']:.2f} (ratio {p['ratio']:.2f}, threshold {p['degraded_ratio_threshold']})")
    st.caption("Drift means the inputs changed; degradation means the errors grew. Either can happen without the other.")
    daily = pd.DataFrame(mon["champion_daily"]).set_index("date")
    st.markdown(f"**Daily errors of {name} (prediction − actual, {UNIT})**")
    st.bar_chart(daily[["rolling_one_step_error", "fixed_origin_7d_error"]])
    st.dataframe(daily.round(1))

    st.subheader("Active model and experiment")
    st.table(pd.Series({
        "Model": name, "Version (MLflow run ID)": version, "Params": json.dumps(champ["params"]),
        "Training period": " to ".join(champ["train_period"]), "Validation period": " to ".join(champ["validation_period"]),
        "Holdout period": " to ".join(champ["holdout_period"]), "Dataset SHA-256": champ["dataset_sha256"],
        "Git commit": champ["git_commit"],
    }, name="value").to_frame())

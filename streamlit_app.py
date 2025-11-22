import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import IsolationForest
from lightgbm import LGBMRegressor
import pmdarima as pm
import io

st.set_page_config(page_title="AI Budget Analyst", layout="wide")
st.title("📊 AI Budget Analyst – Avancerad företags-AI")

st.write("Ladda upp en Excel- eller CSV-fil med kolumner för datum/månad, intäkter och kostnader.")

uploaded = st.file_uploader("Ladda upp Excel/CSV", type=["xlsx", "csv"])

if uploaded:
    # --- Läs data ---
    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded, sheet_name=0)

    st.subheader("Uppladdad data")
    st.dataframe(df)

    # --- Gissa kolumner ---
    cols = [c.lower() for c in df.columns]
    date_col = None
    revenue_col = None
    costs_col = None

    for c in df.columns:
        lc = c.lower()
        if any(x in lc for x in ["månad","month","date","datum"]):
            date_col = c
        if any(x in lc for x in ["intäkt","revenue","sales"]):
            revenue_col = c
        if any(x in lc for x in ["kostnad","cost","expense"]):
            costs_col = c

    if date_col is None or revenue_col is None or costs_col is None:
        st.error("Kunde inte hitta rätt kolumner. Lägg till kolumner som Datum/Månad, Intäkter och Kostnader.")
        st.stop()

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    df["revenue"] = pd.to_numeric(df[revenue_col], errors="coerce")
    df["costs"] = pd.to_numeric(df[costs_col], errors="coerce")
    df["profit"] = df["revenue"] - df["costs"]

    # === ANOMALIDETEKTION ===
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso_preds = iso.fit_predict(df["profit"].values.reshape(-1, 1))
    df["anomaly"] = (iso_preds == -1).astype(int)

    # Skapa tidsserie
    df = df.set_index(date_col)
    ts = df["profit"].asfreq("M").fillna(method="ffill")

    st.subheader("📌 Historisk vinst")
    st.line_chart(ts)

    # === AUTO ARIMA ===
    st.write("🔄 Tränar AI-modell (auto_arima + SARIMAX + LightGBM)...")
    arima_model = pm.auto_arima(ts, seasonal=True, m=12, suppress_warnings=True)
    sarimax = SARIMAX(ts, order=arima_model.order, seasonal_order=arima_model.seasonal_order)
    sarimax_res = sarimax.fit(disp=False)

    # === LightGBM ===
    feat = pd.DataFrame({"ds": ts.index, "y": ts.values})
    for lag in [1, 2, 3, 6, 12]:
        feat[f"lag_{lag}"] = feat["y"].shift(lag)
    feat["month"] = feat["ds"].dt.month
    feat = feat.dropna()

    X = feat.drop(columns=["ds", "y"])
    y = feat["y"]

    ml = LGBMRegressor()
    ml.fit(X, y)

    # === PROGNOS ===
    steps = st.slider("Hur många månader ska AI:n förutspå?", 6, 36, 12)

    last_date = ts.index[-1]
    future_index = pd.date_range(last_date + pd.offsets.MonthEnd(1), periods=steps, freq="M")

    sarimax_pred = sarimax_res.get_forecast(steps=steps)
    sarimax_mean = pd.Series(sarimax_pred.predicted_mean, index=future_index)
    sarimax_ci = sarimax_pred.conf_int()

    # ML recursive
    ml_preds = []
    last_vals = ts[-12:].values.tolist()

    for i in range(steps):
        row = {
            "lag_1": last_vals[-1],
            "lag_2": last_vals[-2],
            "lag_3": last_vals[-3],
            "lag_6": last_vals[-6],
            "lag_12": last_vals[-12],
            "month": future_index[i].month
        }
        p = ml.predict(pd.DataFrame([row]))[0]
        ml_preds.append(p)
        last_vals.append(p)

    ml_pred = pd.Series(ml_preds, index=future_index)

    # Ensemble = medel av två modeller
    ensemble = (sarimax_mean + ml_pred) / 2

    st.subheader("📈 Prognos (ensemble)")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ts.index, ts.values, label="Historisk vinst")
    ax.plot(ensemble.index, ensemble.values, label="Prognos", linestyle="--")
    ax.fill_between(
        future_index,
        sarimax_ci.iloc[:, 0],
        sarimax_ci.iloc[:, 1],
        alpha=0.2
    )
    ax.legend()
    st.pyplot(fig)

    # === SCENARIO ===
    st.subheader("🔮 Scenario-analys")
    optimistic = ensemble * 1.15
    pessimistic = ensemble * 0.85

    st.line_chart(pd.DataFrame({
        "Ensemble": ensemble,
        "Optimistisk": optimistic,
        "Pessimistisk": pessimistic
    }))

    # === Export Excel ===
    st.subheader("📤 Ladda ner resultat")

    export_df = pd.DataFrame({
        "Datum": ensemble.index,
        "Prognos": ensemble.values,
        "Optimistisk": optimistic.values,
        "Pessimistisk": pessimistic.values
    })

    buffer = io.BytesIO()
    export_df.to_excel(buffer, index=False)
    st.download_button("Ladda ner prognos som Excel", buffer.getvalue(), "forecast.xlsx")

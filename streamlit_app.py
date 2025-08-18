# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import datetime
import numpy as np
import xgboost as xgb
import os

st.set_page_config(page_title="Lahore AQI Dashboard", layout="centered")

# --- CONFIG ---
PROCESSED_CSV = "processed_data.csv"
MODEL_PATH = os.path.join("models", "aqi_best_model.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")  # optional

FEATURES = [
    "temperature", "humidity", "wind_speed",
    "pm2_5", "pm10", "co", "no2",
    "aqi_lag1", "pm2_5_lag1", "pm10_lag1", "co_lag1", "no2_lag1"
]

# --- UTILITIES ---
@st.cache_data
def load_files():
    if not os.path.exists(PROCESSED_CSV):
        raise FileNotFoundError(f"{PROCESSED_CSV} not found.")
    df = pd.read_csv(PROCESSED_CSV)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"{MODEL_PATH} not found.")
    model = joblib.load(MODEL_PATH)
    scaler = None
    if os.path.exists(SCALER_PATH):
        try:
            scaler = joblib.load(SCALER_PATH)
        except Exception:
            scaler = None
    return df, model, scaler

def model_predict_safe(model, X_array):
    try:
        preds = model.predict(X_array)
        if hasattr(preds, "__len__"):
            return float(np.asarray(preds).ravel()[0])
        return float(preds)
    except Exception as e1:
        try:
            if hasattr(model, "get_booster"):
                booster = model.get_booster()
                dmatrix = xgb.DMatrix(X_array)
                preds = booster.predict(dmatrix)
                return float(np.asarray(preds).ravel()[0])
        except Exception:
            pass
        raise e1

def forecast_aqi_from_df(df, model, days=3):
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise KeyError(f"Missing required features: {missing}")

    temp_df = df.copy().reset_index(drop=True)
    results = []
    for i in range(days):
        latest = temp_df.iloc[-1].copy()
        X_row = latest[FEATURES].values.reshape(1, -1).astype(float)
        pred_val = model_predict_safe(model, X_row)
        pred_val = round(float(pred_val), 2)

        if "timestamp" in latest and pd.notnull(latest["timestamp"]):
            next_date = pd.to_datetime(latest["timestamp"]) + datetime.timedelta(days=1)
        else:
            next_date = pd.to_datetime(datetime.datetime.utcnow().date()) + datetime.timedelta(days=1)

        new_row = latest.copy()
        new_row["timestamp"] = next_date
        new_row["aqi"] = pred_val
        new_row["aqi_lag1"] = latest.get("aqi", np.nan)
        new_row["pm2_5_lag1"] = latest.get("pm2_5", np.nan)
        new_row["pm10_lag1"] = latest.get("pm10", np.nan)
        new_row["co_lag1"] = latest.get("co", np.nan)
        new_row["no2_lag1"] = latest.get("no2", np.nan)

        results.append({"Date": next_date.strftime("%Y-%m-%d"), "Predicted AQI": pred_val})
        temp_df = pd.concat([temp_df, pd.DataFrame([new_row])], ignore_index=True)

    return pd.DataFrame(results)

# --- APP UI ---
st.title("🌍 Lahore AQI Dashboard")
st.write("Showing only **today's AQI** and **next 3 days forecast**.")

try:
    df, model, scaler = load_files()
except Exception as e:
    st.error(f"Failed to load files: {e}")
    st.stop()

if "aqi" not in df.columns:
    st.error("processed_data.csv does not contain 'aqi'.")
    st.stop()

try:
    latest_row = df.dropna(subset=["aqi"]).iloc[-1]
except Exception:
    st.error("No AQI values found.")
    st.stop()

try:
    latest_aqi = float(latest_row["aqi"])
    st.subheader("📌 Today's AQI")
    st.metric(label="Current AQI", value=round(latest_aqi, 2))
except Exception as e:
    st.error(f"Could not read latest AQI: {e}")
    st.stop()

st.subheader("📅 Next 3 Days AQI Forecast")
try:
    forecast_df = forecast_aqi_from_df(df, model, days=3)
    st.table(forecast_df)
except KeyError as ke:
    st.error(f"Error in forecasting: {ke}")
    st.write("Make sure processed_data.csv contains these columns:")
    st.write(FEATURES + ["timestamp", "aqi"])
except Exception as e:
    st.error(f"Error in forecasting: {e}")
    with st.expander("Show error details"):
        st.exception(e)


import streamlit as st
import pandas as pd
import joblib
import datetime
import numpy as np
import xgboost as xgb
import os

st.set_page_config(page_title="Lahore AQI Dashboard", layout="centered")

# --- CONFIG ---
# Make sure these filenames match what's in your repo
PROCESSED_CSV = "processed_data.csv"
MODEL_PATH = os.path.join("models", "aqi_best_model.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")  # we load but won't re-scale if data already scaled

# Features order used during training (must match model_train.py)
FEATURES = [
    "temperature", "humidity", "wind_speed",
    "pm2_5", "pm10", "co", "no2",
    "aqi_lag1", "pm2_5_lag1", "pm10_lag1", "co_lag1", "no2_lag1"
]

# --- UTILITIES ---
@st.cache_data
def load_files():
    # load processed data
    if not os.path.exists(PROCESSED_CSV):
        raise FileNotFoundError(f"{PROCESSED_CSV} not found in repository root.")
    df = pd.read_csv(PROCESSED_CSV)
    # ensure timestamp parsed
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    # load model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"{MODEL_PATH} not found.")
    model = joblib.load(MODEL_PATH)
    # load scaler if present (we may not need it if processed data is already scaled)
    scaler = None
    if os.path.exists(SCALER_PATH):
        try:
            scaler = joblib.load(SCALER_PATH)
        except Exception:
            scaler = None
    return df, model, scaler

def model_predict_safe(model, X_array):
    """
    Try to predict with model robustly:
      - Prefer model.predict(X)
      - If that errors (XGBoost sklearn-wrapper issues), try Booster.predict(DMatrix)
    X_array: numpy 2D array shape (1, n_features)
    Returns: float prediction
    """
    # Try sklearn-style predict
    try:
        preds = model.predict(X_array)
        # if returns array-like
        if hasattr(preds, "__len__"):
            return float(np.asarray(preds).ravel()[0])
        return float(preds)
    except Exception as e1:
        # Try XGBoost Booster predict
        try:
            if hasattr(model, "get_booster"):
                booster = model.get_booster()
                dmatrix = xgb.DMatrix(X_array)
                preds = booster.predict(dmatrix)
                return float(np.asarray(preds).ravel()[0])
        except Exception:
            pass
        # fallback: raise original exception (we'll catch at call site)
        raise e1

def forecast_aqi_from_df(df, model, days=3):
    """
    df: processed_data dataframe (with scaled numeric features if that was the training step)
    model: trained model object
    returns: forecast DataFrame with columns Date, Predicted AQI
    """
    # Check features existence
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise KeyError(f"Missing required features in processed_data.csv: {missing}")

    temp_df = df.copy().reset_index(drop=True)
    results = []
    for i in range(days):
        latest = temp_df.iloc[-1].copy()
        # Build feature vector in the exact order
        X_row = latest[FEATURES].values.reshape(1, -1).astype(float)

        # Predict safely
        pred_val = model_predict_safe(model, X_row)
        pred_val = round(float(pred_val), 2)

        # Next date
        if "timestamp" in latest and pd.notnull(latest["timestamp"]):
            next_date = pd.to_datetime(latest["timestamp"]) + datetime.timedelta(days=1)
        else:
            next_date = pd.to_datetime(datetime.datetime.utcnow().date()) + datetime.timedelta(days=1)

        # Build new row for iterative forecasting:
        new_row = latest.copy()
        new_row["timestamp"] = next_date
        new_row["aqi"] = pred_val

        # update lag features for next iteration: note we set lag1 from latest (previous) values
        new_row["aqi_lag1"] = latest.get("aqi", np.nan)
        new_row["pm2_5_lag1"] = latest.get("pm2_5", np.nan)
        new_row["pm10_lag1"] = latest.get("pm10", np.nan)
        new_row["co_lag1"] = latest.get("co", np.nan)
        new_row["no2_lag1"] = latest.get("no2", np.nan)

        results.append({"Date": next_date.strftime("%Y-%m-%d"), "Predicted AQI": pred_val})

        # append to temp_df so next iteration uses new_row
        temp_df = pd.concat([temp_df, pd.DataFrame([new_row])], ignore_index=True)

    return pd.DataFrame(results)

# --- APP UI ---
st.title("🌍 Lahore AQI Dashboard")
st.write("Showing only **today's AQI** (latest available) and **next 3 days forecast**.")

# Load data & model
try:
    df, model, scaler = load_files()
except Exception as e:
    st.error(f"Failed to load files: {e}")
    st.stop()

# show today's AQI (take latest non-null 'aqi' from processed data)
if "aqi" not in df.columns:
    st.error("processed_data.csv does not contain 'aqi' column.")
    st.stop()

try:
    latest_row = df.dropna(subset=["aqi"]).iloc[-1]
except Exception:
    st.error("No AQI values found in processed_data.csv.")
    st.stop()

# Current AQI metric
try:
    latest_aqi = float(latest_row["aqi"])
    st.subheader("📌 Today's AQI")
    st.metric(label="Current AQI", value=round(latest_aqi, 2))
except Exception as e:
    st.error(f"Could not read latest AQI: {e}")
    st.stop()

# Forecast block: next 3 days
st.subheader("📅 Next 3 Days AQI Forecast")
try:
    forecast_df = forecast_aqi_from_df(df, model, days=3)
    st.table(forecast_df)
except KeyError as ke:
    st.error(f"Error in forecasting: {ke}")
    st.write("Make sure `processed_data.csv` contains these columns (exact names):")
    st.write(FEATURES + ["timestamp", "aqi"])
except Exception as e:
    # show short message + log details
    st.error(f"Error in forecasting: {e}")
    # For debug (optional) show exception text in expander (safe since model/data may contain secrets — don't print those)
    with st.expander("Show error details"):
        st.exception(e) 

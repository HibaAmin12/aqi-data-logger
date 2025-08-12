import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import timedelta

# --- Load model, scaler, and latest data ---
@st.cache_data
def load_assets():
    model = joblib.load("models/aqi_best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    latest = pd.read_csv("latest_pollutants.csv")
    return model, scaler, latest

model, scaler, latest = load_assets()

st.title("🌍 Lahore AQI Dashboard")
st.subheader("📌 Today's AQI Prediction")

# --- Prepare features for prediction ---
features = ["temperature", "humidity", "wind_speed", "pm2_5", "pm10", "co", "no2",
            "aqi_lag1", "pm2_5_lag1", "pm10_lag1", "co_lag1", "no2_lag1"]

# Scale numeric features
numeric_features = ["temperature", "humidity", "wind_speed", "pm2_5", "pm10", "co", "no2"]

# Extract latest feature row
X_raw = latest[features].copy()

# Scale numeric columns only
X_raw[numeric_features] = scaler.transform(X_raw[numeric_features])

# Convert to numpy array for prediction
X = X_raw.values

# Predict AQI
try:
    predicted_aqi = model.predict(X)[0]
    predicted_aqi = round(float(predicted_aqi), 2)
except Exception as e:
    st.error(f"Prediction failed: {e}")
    predicted_aqi = None

if predicted_aqi is not None:
    st.metric("Current AQI (Predicted)", value=predicted_aqi)

    # --- Next 3 days forecast (simple lag-based) ---
    st.subheader("📅 Next 3 Days AQI Forecast")
    forecast_results = []
    temp_row = latest.iloc[0].copy()

    for i in range(1, 4):
        # Update lags with previous prediction
        temp_row["aqi_lag1"] = predicted_aqi if i == 1 else forecast_results[-1]["Predicted AQI"]
        temp_row["pm2_5_lag1"] = temp_row["pm2_5"]
        temp_row["pm10_lag1"] = temp_row["pm10"]
        temp_row["co_lag1"] = temp_row["co"]
        temp_row["no2_lag1"] = temp_row["no2"]

        # Prepare features and scale
        features_row = temp_row[features].copy()
        features_row[numeric_features] = scaler.transform([features_row[numeric_features]])[0]

        # Predict next day AQI
        next_aqi = model.predict([features_row.values])[0]
        next_aqi = round(float(next_aqi), 2)

        forecast_date = pd.to_datetime(temp_row["timestamp"]) + timedelta(days=i)
        forecast_results.append({"Date": forecast_date.strftime("%Y-%m-%d"), "Predicted AQI": next_aqi})

    forecast_df = pd.DataFrame(forecast_results)
    st.table(forecast_df)
else:
    st.warning("Could not generate AQI prediction.")


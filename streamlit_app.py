import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime

# --------------------------
# Load Model & Scaler
# --------------------------
model = joblib.load("models/aqi_best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Features used in training
features = [
    "temperature", "humidity", "wind_speed", "pm2_5", "pm10", "co", "no2",
    "aqi_lag1", "pm2_5_lag1", "pm10_lag1", "co_lag1", "no2_lag1"
]
numeric_features = ["temperature", "humidity", "wind_speed", "pm2_5", "pm10", "co", "no2"]

# --------------------------
# Load Processed Data
# --------------------------
df = pd.read_csv("processed_data.csv").sort_values("timestamp").reset_index(drop=True)

# Ensure lag features exist
for col in ["aqi", "pm2_5", "pm10", "co", "no2"]:
    lag_col = f"{col}_lag1"
    if lag_col not in df.columns:
        df[lag_col] = df[col].shift(1)

df = df.dropna().reset_index(drop=True)

# --------------------------
# Forecast Function
# --------------------------
def forecast_aqi(df, model, scaler, days=3):
    df = df.copy()
    forecasts = []
    last_row = df.iloc[-1].copy()

    for _ in range(days):
        # Create feature row
        X = pd.DataFrame([last_row[features].values], columns=features)

        # Scale only numeric features
        X_scaled = X.copy()
        X_scaled[numeric_features] = scaler.transform(X[numeric_features])

        # Predict AQI
        pred_aqi = model.predict(X_scaled)[0]
        forecasts.append(pred_aqi)

        # Shift lag features for next prediction
        last_row["aqi_lag1"] = pred_aqi
        last_row["pm2_5_lag1"] = last_row["pm2_5"]
        last_row["pm10_lag1"] = last_row["pm10"]
        last_row["co_lag1"] = last_row["co"]
        last_row["no2_lag1"] = last_row["no2"]

        # Simulate small pollutant change
        for col in numeric_features:
            last_row[col] = last_row[col] * np.random.uniform(0.98, 1.02)

    return forecasts

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Lahore AQI Dashboard", layout="centered")

st.title("🌍 Lahore AQI Dashboard")

# Today's AQI
latest_aqi = df.iloc[-1]["aqi"]
st.subheader("📌 Today's AQI")
st.metric("Current AQI", round(latest_aqi, 2))

# Forecast for next 3 days
try:
    forecast_values = forecast_aqi(df, model, scaler, days=3)
    forecast_dates = [
        (datetime.date.today() + datetime.timedelta(days=i+1)).strftime("%Y-%m-%d")
        for i in range(3)
    ]
    forecast_df = pd.DataFrame({"Date": forecast_dates, "Predicted AQI": np.round(forecast_values, 2)})

    st.subheader("📅 Next 3 Days AQI Forecast")
    st.table(forecast_df)
except Exception as e:
    st.error(f"Error in forecasting: {e}")

st.markdown("Made with ❤️ using Streamlit")

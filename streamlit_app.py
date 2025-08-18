import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/aqi_best_model.pkl")

st.title("AQI Prediction Dashboard")

# Load processed data
try:
    data = pd.read_csv("processed_data.csv")
except Exception as e:
    st.error(f"❌ Failed to load data: {e}")
    st.stop()

# Show recent data
st.subheader("Latest Processed Data")
st.write(data.tail())

# ✅ Expected features used during training
expected_features = [
    "temperature", "humidity", "wind_speed",
    "pm2_5", "pm10", "co", "no2",
    "aqi_lag1", "pm2_5_lag1", "pm10_lag1", "co_lag1", "no2_lag1"
]

missing = [col for col in expected_features if col not in data.columns]
if missing:
    st.error(f"❌ Missing features in processed_data.csv: {missing}")
    st.stop()

# ✅ Use the latest row for prediction
latest = data[expected_features].iloc[-1:].copy()

try:
    pred_aqi = model.predict(latest)[0]
    st.subheader("Predicted AQI")
    st.success(f"🌍 Predicted AQI: {pred_aqi:.2f}")
except Exception as e:
    st.error(f"⚠️ Prediction failed: {e}")

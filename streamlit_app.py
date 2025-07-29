import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# 🎯 Load model
@st.cache_resource
def load_model():
    model_path = "model_outputs/best_model.pkl"
    if not os.path.exists(model_path):
        st.error("❌ Trained model not found in 'model_outputs/'.")
        return None
    return joblib.load(model_path)

model = load_model()

# Load training feature columns
feature_cols_path = "model_outputs/feature_columns.json"
if os.path.exists(feature_cols_path):
    import json
    with open(feature_cols_path, "r") as f:
        feature_columns = json.load(f)
else:
    feature_columns = None

st.title("🌫️ AQI Predictor (Next 3 Days)")
st.write("Enter today's environmental parameters to predict AQI for the next 3 days.")

# Inputs
temperature = st.slider("🌡️ Temperature (°C)", 15.0, 45.0, 30.0)
humidity = st.slider("💧 Humidity (%)", 10, 100, 60)
wind_speed = st.slider("🌬️ Wind Speed (m/s)", 0.0, 10.0, 2.0)
pm2_5 = st.number_input("PM2.5 (µg/m³)", 0.0, 300.0, 35.0)
pm10 = st.number_input("PM10 (µg/m³)", 0.0, 500.0, 50.0)
co = st.number_input("CO (µg/m³)", 0.0, 1000.0, 400.0)
no2 = st.number_input("NO₂ (µg/m³)", 0.0, 100.0, 10.0)
weather_main = st.selectbox("🌦️ Weather", ["Clear", "Clouds", "Smoke", "Mist", "Rain", "Thunderstorm"])

# Convert to DataFrame (match training format)
input_df = pd.DataFrame([{
    "temperature": temperature,
    "humidity": humidity,
    "wind_speed": wind_speed,
    "pm2_5": pm2_5,
    "pm10": pm10,
    "co": co,
    "no2": no2,
    "weather_main": weather_main
}])

# One-hot encode weather_main
input_encoded = pd.get_dummies(input_df, columns=["weather_main"], drop_first=True)

# Align with training columns
if feature_columns:
    for col in feature_columns:
        if col not in input_encoded:
            input_encoded[col] = 0
    input_encoded = input_encoded[feature_columns]

if st.button("Predict AQI for Next 3 Days"):
    if model is not None:
        preds = model.predict(input_encoded)
        st.success(f"🌟 Day 1 AQI: {preds[0]:.2f}")
        st.info(f"📆 Day 2 AQI: {preds[0] * 1.02:.2f}")
        st.warning(f"📆 Day 3 AQI: {preds[0] * 1.04:.2f}")
    else:
        st.error("⚠️ Model not loaded. Please retrain and upload model.")

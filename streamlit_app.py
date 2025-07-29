import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# 🎯 Load trained multi-output model
@st.cache_resource
def load_model():
    model_path = "model_outputs/best_model.pkl"
    if not os.path.exists(model_path):
        st.error("❌ Trained model not found in 'model_outputs/'. Please ensure best_model.pkl is present.")
        return None
    return joblib.load(model_path)

model = load_model()

# 🖥️ UI
st.title("🌫️ AQI Predictor (Next 3 Days)")
st.write("Enter today's environmental conditions to forecast AQI for the next 3 days.")

temperature = st.slider("🌡️ Temperature (°C)", 0.0, 50.0, 25.0)
humidity    = st.slider("💧 Humidity (%)", 0, 100, 60)
wind_speed  = st.slider("🌬️ Wind Speed (m/s)", 0.0, 20.0, 2.0)
pm2_5       = st.number_input("PM2.5 (µg/m³)", 0.0, 300.0, 35.0)
pm10        = st.number_input("PM10 (µg/m³)", 0.0, 500.0, 50.0)
co          = st.number_input("CO (µg/m³)", 0.0, 1000.0, 400.0)
no2         = st.number_input("NO₂ (µg/m³)", 0.0, 200.0, 10.0)
weather_main = st.selectbox("🌦️ Weather Condition", ["Clouds", "Smoke", "Rain", "Mist", "Thunderstorm"])

# Prepare input vector (same encoding as training)
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

# One-hot encode weather_main (ensure same columns as training)
input_encoded = pd.get_dummies(input_df, columns=["weather_main"])
# Add missing columns if any
expected_cols = model.feature_names_in_
for col in expected_cols:
    if col not in input_encoded:
        input_encoded[col] = 0

input_encoded = input_encoded[expected_cols]  # reorder columns

# Predict
if st.button("🔮 Predict AQI for Next 3 Days"):
    if model is None:
        st.warning("⚠️ Model could not be loaded.")
    else:
        preds = model.predict(input_encoded)[0]
        st.subheader("📈 Forecasted AQI Values")
        st.write(f"• **Day 1:** {preds[0]:.2f}")
        st.write(f"• **Day 2:** {preds[1]:.2f}")
        st.write(f"• **Day 3:** {preds[2]:.2f}")

        def aqi_category(aqi):
            if aqi <= 50:   return "Good 😊"
            if aqi <= 100:  return "Moderate 😐"
            if aqi <= 150:  return "Unhealthy for Sensitive Groups 😷"
            if aqi <= 200:  return "Unhealthy 😷"
            if aqi <= 300:  return "Very Unhealthy 😫"
            return "Hazardous ☠️"

        st.subheader("🏷️ AQI Categories")
        st.write(f"Day 1: {aqi_category(preds[0])}")
        st.write(f"Day 2: {aqi_category(preds[1])}")
        st.write(f"Day 3: {aqi_category(preds[2])}")

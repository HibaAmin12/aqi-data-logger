import streamlit as st
import numpy as np
import joblib
import os

# 🎯 Load trained model from file
@st.cache_resource
def load_model():
    model_path = "model_outputs/best_model.pkl"
    if not os.path.exists(model_path):
        st.error("❌ Trained model not found in 'model_outputs/'. Please upload 'best_model.pkl'.")
        return None
    return joblib.load(model_path)

model = load_model()

# 🖥️ Streamlit UI
st.title("🌫️ AQI Predictor")

st.write("Enter environmental parameters to predict Air Quality Index (AQI):")

temperature = st.slider("🌡️ Temperature (°C)", 15.0, 45.0, 30.0)
humidity = st.slider("💧 Humidity (%)", 10, 100, 60)
wind_speed = st.slider("🌬️ Wind Speed (m/s)", 0.0, 10.0, 2.0)
pm2_5 = st.number_input("PM2.5 (µg/m³)", 0.0, 300.0, 35.0)
pm10 = st.number_input("PM10 (µg/m³)", 0.0, 500.0, 50.0)
co = st.number_input("CO (µg/m³)", 0.0, 1000.0, 400.0)
no2 = st.number_input("NO₂ (µg/m³)", 0.0, 100.0, 10.0)

input_data = np.array([[temperature, humidity, wind_speed, pm2_5, pm10, co, no2]])

if st.button("Predict AQI"):
    if model:
        aqi = model.predict(input_data)[0]
        st.success(f"✅ Predicted AQI: **{aqi:.2f}**")
    else:
        st.warning("⚠️ Model could not be loaded. Please check 'model_outputs/best_model.pkl'.")

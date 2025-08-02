import streamlit as st
import joblib
import numpy as np
from datetime import datetime, timedelta

# Load trained model
model = joblib.load("models/aqi_best_model.pkl")

# Title
st.title("🌍 3-Day AQI Prediction App")
st.markdown("Enter today's weather and pollutant data to predict AQI for the next 3 days.")

# Date Selection
start_date = st.date_input("📅 Select Base Date:", datetime.today())

# Input fields
temperature = st.number_input("🌡 Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0)
humidity = st.number_input("💧 Humidity (%)", min_value=0, max_value=100, value=60)
wind_speed = st.number_input("💨 Wind Speed (m/s)", min_value=0.0, max_value=20.0, value=2.0)
pm2_5 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, max_value=500.0, value=35.0)
pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, max_value=600.0, value=50.0)
co = st.number_input("CO (µg/m³)", min_value=0.0, max_value=2000.0, value=400.0)
no2 = st.number_input("NO₂ (µg/m³)", min_value=0.0, max_value=200.0, value=10.0)

# Predict Button
if st.button("Predict 3-Day AQI"):
    features = np.array([[temperature, humidity, wind_speed, pm2_5, pm10, co, no2]])
    
    predictions = {}
    for i in range(3):
        day = start_date + timedelta(days=i)
        prediction = model.predict(features)[0]
        predictions[day.strftime("%Y-%m-%d")] = prediction

    # Display Results
    for date, pred in predictions.items():
        st.write(f"📅 **{date}** → Predicted AQI: **{pred:.2f}**")
        
        # AQI Levels
        if pred <= 50:
            st.success("🟢 Good")
        elif pred <= 100:
            st.info("🟡 Moderate")
        elif pred <= 150:
            st.warning("🟠 Unhealthy for Sensitive Groups")
        elif pred <= 200:
            st.warning("🔴 Unhealthy")
        elif pred <= 300:
            st.error("🟣 Very Unhealthy")
        else:
            st.error("⚫ Hazardous")

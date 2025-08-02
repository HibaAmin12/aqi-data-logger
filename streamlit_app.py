import streamlit as st
import joblib
import requests
import numpy as np
from datetime import datetime, timedelta

API_KEY = "16e3fa6809dc606fa5e160ea82e475d1"
LAT, LON = 31.5497, 74.3436  # Lahore example (replace with user city)

# Load model
model = joblib.load("models/aqi_best_model.pkl")

st.title("🌍 3-Day AQI Prediction (Auto)")

if st.button("Predict Next 3 Days AQI"):
    # Fetch 3-day weather forecast
    forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
    weather_data = requests.get(forecast_url).json()

    # Fetch current pollution data
    pollution_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}"
    pollution_data = requests.get(pollution_url).json()
    components = pollution_data["list"][0]["components"]

    pm2_5, pm10, co, no2 = components["pm2_5"], components["pm10"], components["co"], components["no2"]

    predictions = {}
    for i in range(3):
        day = datetime.today() + timedelta(days=i)
        temp = weather_data["list"][i*8]["main"]["temp"]
        hum = weather_data["list"][i*8]["humidity"]
        wind = weather_data["list"][i*8]["wind"]["speed"]

        features = np.array([[temp, hum, wind, pm2_5, pm10, co, no2]])
        pred = model.predict(features)[0]
        predictions[day.strftime("%Y-%m-%d")] = pred

    # Show results
    for date, aqi in predictions.items():
        st.write(f"📅 **{date}** → Predicted AQI: **{aqi:.2f}**")
        if aqi <= 50: st.success("🟢 Good")
        elif aqi <= 100: st.info("🟡 Moderate")
        elif aqi <= 150: st.warning("🟠 Unhealthy for Sensitive Groups")
        elif aqi <= 200: st.warning("🔴 Unhealthy")
        elif aqi <= 300: st.error("🟣 Very Unhealthy")
        else: st.error("⚫ Hazardous")

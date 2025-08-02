import streamlit as st
import joblib
import requests
import numpy as np
import hopsworks
import pandas as pd
from datetime import datetime, timedelta

# Load trained model
model = joblib.load("models/aqi_best_model.pkl")

# OpenWeather API details
API_KEY = "16e3fa6809dc606fa5e160ea82e475d1"  # Replace with your key
LAT, LON = 31.5497, 74.3436  # Lahore

st.title("3-Day AQI Prediction")

# ------------------------
# Fetch latest pollutants from Hopsworks
# ------------------------
@st.cache_data
def fetch_latest_pollutants():
    project = hopsworks.login(api_key_value=st.secrets["HOPSWORKS_API_KEY"], 
                              project="Api_feature_store", 
                              host="c.app.hopsworks.ai")
    fs = project.get_feature_store()
    fg = fs.get_feature_group("aqi_features", version=1)
    df = fg.read()
    df = df.dropna(subset=["aqi"])
    latest = df.sort_values("timestamp", ascending=False).iloc[0]
    return {
        "pm2_5": latest["pm2_5"],
        "pm10": latest["pm10"],
        "co": latest["co"],
        "no2": latest["no2"]
    }

pollutants = fetch_latest_pollutants()

# ------------------------
# Fetch 3-day weather forecast
# ------------------------
def fetch_weather_forecast():
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
    resp = requests.get(url).json()
    forecast = []
    for i in range(0, 24, 8):  # 3 days, every 24h step from forecast
        if "list" in resp and len(resp["list"]) > i:
            entry = resp["list"][i]
            forecast.append({
                "temp": entry["main"]["temp"],
                "humidity": entry["main"]["humidity"],
                "wind": entry["wind"]["speed"]
            })
    return forecast

if st.button("Predict Next 3 Days AQI"):
    weather = fetch_weather_forecast()
    predictions = {}

    pm2_5 = pollutants["pm2_5"]
    pm10 = pollutants["pm10"]
    co = pollutants["co"]
    no2 = pollutants["no2"]

    for i in range(3):
        day = datetime.today() + timedelta(days=i)
        temp = weather[i]["temp"]
        hum = weather[i]["humidity"]
        wind = weather[i]["wind"]

        features = np.array([[temp, hum, wind, pm2_5, pm10, co, no2]])
        pred_aqi = model.predict(features)[0]
        predictions[day.strftime("%Y-%m-%d")] = pred_aqi

        # Recursive pollutant adjustment (model-driven)
        pm2_5 *= 0.98  # small decay
        pm10 *= 0.98
        co *= 0.99
        no2 *= 0.99

    # Display predictions
    for date, aqi in predictions.items():
        st.write(f"{date} → Predicted AQI: {aqi:.2f}")
        if aqi <= 50:
            st.write("Good")
        elif aqi <= 100:
            st.write("Moderate")
        elif aqi <= 150:
            st.write("Unhealthy for Sensitive Groups")
        elif aqi <= 200:
            st.write("Unhealthy")
        elif aqi <= 300:
            st.write("Very Unhealthy")
        else:
            st.write("Hazardous")

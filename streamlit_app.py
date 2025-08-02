import streamlit as st
import joblib
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

# --------------------
# Load Model & Data
# --------------------
model = joblib.load("models/aqi_best_model.pkl")
latest_df = pd.read_csv("latest_pollutants.csv").iloc[0]  # Latest AQI + pollutants with lags

# OpenWeather API (Weather Forecast)
API_KEY = "16e3fa6809dc606fa5e160ea82e475d1"
LAT, LON = 31.5497, 74.3436  # Lahore Coordinates

# --------------------
# Streamlit UI
# --------------------
st.title("3-Day AQI Prediction (Lag-based Model)")
st.write("This app predicts AQI for the next 3 days using weather forecasts and lagged AQI trends.")

# --------------------
# Fetch 3-Day Weather Forecast
# --------------------
def fetch_weather():
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
    resp = requests.get(url).json()
    forecast = []
    for i in range(0, 24, 8):  # Every 24h → 3 points for 3 days
        entry = resp["list"][i]
        forecast.append({
            "temp": entry["main"]["temp"],
            "humidity": entry["main"]["humidity"],
            "wind": entry["wind"]["speed"]
        })
    return forecast

# --------------------
# Prediction Logic (Recursive with Lag)
# --------------------
if st.button("Predict AQI for Next 3 Days"):
    forecast = fetch_weather()

    # Initialize lags from latest CSV
    aqi_lag = latest_df["aqi"]
    pm2_5_lag, pm10_lag, co_lag, no2_lag = latest_df["pm2_5"], latest_df["pm10"], latest_df["co"], latest_df["no2"]

    predictions = {}

    for i in range(3):
        day = datetime.today() + timedelta(days=i)
        temp = forecast[i]["temp"]
        hum = forecast[i]["humidity"]
        wind = forecast[i]["wind"]

        # Feature vector (weather + pollutants + lagged AQI/pollutants)
        features = np.array([[
            temp, hum, wind,
            pm2_5_lag, pm10_lag, co_lag, no2_lag,
            aqi_lag, pm2_5_lag, pm10_lag, co_lag, no2_lag
        ]])

        # Predict AQI
        pred_aqi = model.predict(features)[0]
        predictions[day.strftime("%Y-%m-%d")] = pred_aqi

        # Update lags for next day
        aqi_lag = pred_aqi  # Predicted AQI becomes next day's lag
        # Pollutants decay slightly (realistic approximation)
        pm2_5_lag *= 0.98
        pm10_lag *= 0.98
        co_lag *= 0.99
        no2_lag *= 0.99

    # --------------------
    # Display Results
    # --------------------
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

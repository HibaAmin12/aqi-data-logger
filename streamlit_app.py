import streamlit as st
import joblib
import requests
import numpy as np
from datetime import datetime, timedelta

API_KEY = "16e3fa6809dc606fa5e160ea82e475d1"  
LAT, LON = 31.5497, 74.3436  

model = joblib.load("models/aqi_best_model.pkl")
st.title("3-Day AQI Prediction (Auto)")

if st.button("Predict Next 3 Days AQI"):
    forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
    weather_response = requests.get(forecast_url)
    
    if weather_response.status_code != 200:
        st.error(f"Weather API Error: {weather_response.json().get('message', 'Unknown error')}")
    else:
        weather_data = weather_response.json()

        if "list" not in weather_data:
            st.error("Weather forecast data not available.")
        else:
            pollution_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}"
            pollution_response = requests.get(pollution_url)
            
            if pollution_response.status_code != 200:
                st.error(f"Pollution API Error: {pollution_response.json().get('message', 'Unknown error')}")
            else:
                components = pollution_response.json()["list"][0]["components"]
                pm2_5, pm10, co, no2 = components["pm2_5"], components["pm10"], components["co"], components["no2"]

                predictions = {}
                for i in range(3):
                    if i*8 < len(weather_data["list"]):
                        day = datetime.today() + timedelta(days=i)
                        temp = weather_data["list"][i*8]["main"]["temp"]
                        hum = weather_data["list"][i*8]["main"]["humidity"]
                        wind = weather_data["list"][i*8]["wind"]["speed"]

                        features = np.array([[temp, hum, wind, pm2_5, pm10, co, no2]])
                        pred = model.predict(features)[0]
                        predictions[day.strftime("%Y-%m-%d")] = pred

                for date, aqi in predictions.items():
                    st.write(f"{date} → Predicted AQI: {aqi:.2f}")
                    if aqi <= 50: st.write("Good")
                    elif aqi <= 100: st.write("Moderate")
                    elif aqi <= 150: st.write("Unhealthy for Sensitive Groups")
                    elif aqi <= 200: st.write("Unhealthy")
                    elif aqi <= 300: st.write("Very Unhealthy")
                    else: st.write("Hazardous")

import streamlit as st
import pandas as pd
import pickle
import requests
from datetime import datetime, timedelta

# ---------------------
# CONFIGURATION
# ---------------------
API_KEY = "YOUR_OPENWEATHER_API_KEY"  # apna API key yahan lagao
CITY = "Lahore"  # apni city ka naam
MODEL_PATH = "aqi_best_model.pkl"  # trained model ka path

# ---------------------
# HELPER FUNCTIONS
# ---------------------
def load_model():
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
    return model

def get_forecast_data(city, api_key):
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json()

    forecast_list = []
    for entry in data["list"]:
        forecast_list.append({
            "datetime": datetime.fromtimestamp(entry["dt"]),
            "temp": entry["main"]["temp"],
            "humidity": entry["main"]["humidity"],
            "wind_speed": entry["wind"]["speed"],
            "pressure": entry["main"]["pressure"],
            # Pollutant placeholder (agar OpenWeather se na mile to default value lagao)
            "pm2_5": 50,  
            "pm10": 70
        })

    forecast_df = pd.DataFrame(forecast_list)
    return forecast_df

def prepare_features(df):
    # Features ka selection tumhare training ke hisab se
    return df[["temp", "humidity", "wind_speed", "pressure", "pm2_5", "pm10"]]

# ---------------------
# STREAMLIT UI
# ---------------------
st.set_page_config(page_title="AQI Forecast Dashboard", layout="wide")

st.title("🌫 AQI Prediction Dashboard")
st.markdown("### Next 3 Days Air Quality Forecast")

# Model load
model = load_model()

# Data fetch
st.sidebar.header("Settings")
city_input = st.sidebar.text_input("Enter City Name", value=CITY)

if st.sidebar.button("Get Forecast"):
    with st.spinner("Fetching weather data..."):
        forecast_df = get_forecast_data(city_input, API_KEY)

        # Next 3 days ka filter
        today = datetime.now()
        end_date = today + timedelta(days=3)
        forecast_df = forecast_df[(forecast_df["datetime"] >= today) & (forecast_df["datetime"] <= end_date)]

        # Prediction
        features = prepare_features(forecast_df)
        forecast_df["Predicted_AQI"] = model.predict(features)

        # Display table
        st.dataframe(forecast_df[["datetime", "temp", "humidity", "wind_speed", "pm2_5", "pm10", "Predicted_AQI"]])

        # Chart
        st.line_chart(forecast_df.set_index("datetime")["Predicted_AQI"])

        st.success("Forecast generated successfully!")

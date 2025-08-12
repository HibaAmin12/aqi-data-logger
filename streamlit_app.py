import streamlit as st
import pandas as pd
import joblib
import datetime

# Load data and model
@st.cache_data
def load_data_and_model():
    df = pd.read_csv("processed_data.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    latest = pd.read_csv("latest_pollutants.csv")
    model = joblib.load("models/aqi_best_model.pkl")
    return df, latest, model

df, latest, model = load_data_and_model()

st.title("🌍 Lahore AQI Dashboard")

# Show today's AQI
today_aqi = round(float(latest.iloc[0]['aqi']), 2)
st.subheader("📌 Today's AQI")
st.metric("Current AQI", value=today_aqi)

# Prepare features for next 3 days forecasting
features = [
    "temperature", "humidity", "wind_speed",
    "pm2_5", "pm10", "co", "no2",
    "aqi_lag1", "pm2_5_lag1", "pm10_lag1", "co_lag1", "no2_lag1"
]

# Get latest data row (last row in historical df)
latest_data = df.iloc[-1]

forecast_results = []

for i in range(1, 4):  # next 3 days
    X = latest_data[features].values.reshape(1, -1)
    predicted_aqi = model.predict(X)[0]
    predicted_aqi = round(float(predicted_aqi), 2)

    next_date = latest_data["timestamp"] + datetime.timedelta(days=i)
    forecast_results.append({"Date": next_date.strftime("%Y-%m-%d"), "Predicted AQI": predicted_aqi})

    # Update lag features for next day prediction
    latest_data["aqi_lag1"] = predicted_aqi
    latest_data["pm2_5_lag1"] = latest_data["pm2_5"]
    latest_data["pm10_lag1"] = latest_data["pm10"]
    latest_data["co_lag1"] = latest_data["co"]
    latest_data["no2_lag1"] = latest_data["no2"]

forecast_df = pd.DataFrame(forecast_results)

st.subheader("📅 Next 3 Days AQI Forecast")
st.table(forecast_df)

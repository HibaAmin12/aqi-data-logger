import streamlit as st
import pandas as pd
import joblib
import datetime
import numpy as np

st.set_page_config(page_title="Lahore AQI Dashboard", layout="wide")

@st.cache_data
def load_data_and_model():
    df = pd.read_csv("processed_data.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    model = joblib.load("models/aqi_best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return df, model, scaler

df, model, scaler = load_data_and_model()

def forecast_aqi(df, model, scaler, days=3):
    features = [
        "temperature", "humidity", "wind_speed",
        "pm2_5", "pm10", "co", "no2",
        "aqi_lag1", "pm2_5_lag1", "pm10_lag1", "co_lag1", "no2_lag1"
    ]
    
    forecast_results = []
    temp_df = df.copy()

    for _ in range(days):
        latest_data = temp_df.iloc[-1].copy()
        X = latest_data[features].values.reshape(1, -1)
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]
        pred = float(round(pred, 2))
        
        next_date = latest_data["timestamp"] + datetime.timedelta(days=1)
        new_row = latest_data.copy()
        new_row["timestamp"] = next_date
        new_row["aqi"] = pred
        
        # Update lag features for next prediction
        new_row["aqi_lag1"] = latest_data["aqi"]
        new_row["pm2_5_lag1"] = latest_data["pm2_5"]
        new_row["pm10_lag1"] = latest_data["pm10"]
        new_row["co_lag1"] = latest_data["co"]
        new_row["no2_lag1"] = latest_data["no2"]

        forecast_results.append({"Date": next_date.strftime("%Y-%m-%d"), "Predicted AQI": pred})
        temp_df = pd.concat([temp_df, pd.DataFrame([new_row])], ignore_index=True)
    
    return pd.DataFrame(forecast_results)

st.title("🌍 Lahore AQI Dashboard")

st.subheader("📌 Today's AQI")
latest_aqi = df.iloc[-1]["aqi"]
st.metric("Current AQI", round(float(latest_aqi), 2))

st.subheader("📅 Next 3 Days AQI Forecast")
forecast_df = forecast_aqi(df, model, scaler, days=3)
st.table(forecast_df)

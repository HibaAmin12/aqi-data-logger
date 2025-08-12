import streamlit as st
import pandas as pd
import joblib
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import numpy as np

st.set_page_config(page_title="Lahore AQI Dashboard", layout="wide")

@st.cache_data
def load_data_and_model():
    df = pd.read_csv("processed_data.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    model = joblib.load("models/aqi_best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")  # Agar scaler use karte ho
    return df, model, scaler

df, model, scaler = load_data_and_model()

def forecast_aqi(df, model, scaler=None, days=3):
    features = [
        "temperature", "humidity", "wind_speed",
        "pm2_5", "pm10", "co", "no2",
        "aqi_lag1", "pm2_5_lag1", "pm10_lag1", "co_lag1", "no2_lag1"
    ]
    
    forecast_results = []
    temp_df = df.copy()
    
    for i in range(days):
        latest_data = temp_df.iloc[-1].copy()
        X_latest = latest_data[features].values.reshape(1, -1)
        
        if scaler:
            X_latest = scaler.transform(X_latest)
        
        predicted_aqi = model.predict(X_latest)[0]
        predicted_aqi = round(float(predicted_aqi), 2)
        
        next_date = latest_data["timestamp"] + datetime.timedelta(days=1)
        
        new_row = latest_data.copy()
        new_row["timestamp"] = next_date
        new_row["aqi"] = predicted_aqi
        
        # Update lag features for next iteration
        new_row["aqi_lag1"] = latest_data["aqi"]
        new_row["pm2_5_lag1"] = latest_data["pm2_5"]
        new_row["pm10_lag1"] = latest_data["pm10"]
        new_row["co_lag1"] = latest_data["co"]
        new_row["no2_lag1"] = latest_data["no2"]
        
        forecast_results.append({
            "Date": next_date.strftime("%Y-%m-%d"),
            "Predicted AQI": predicted_aqi
        })
        
        temp_df = pd.concat([temp_df, pd.DataFrame([new_row])], ignore_index=True)
    
    return pd.DataFrame(forecast_results)

# Show title
st.title("🌍 Lahore AQI Dashboard")

# Show today's AQI (last available from df)
latest_aqi = df.iloc[-1]["aqi"]
st.subheader("📌 Today's AQI")
st.metric(label="Current AQI", value=round(float(latest_aqi), 2))

# Forecast next 3 days AQI
forecast_df = forecast_aqi(df, model, scaler, days=3)
st.subheader("📅 Next 3 Days AQI Forecast")
st.table(forecast_df)

# Plot Historical and Forecast AQI
st.subheader("📈 Historical AQI & Forecast Trend")

plt.figure(figsize=(10, 5))
sns.lineplot(x=df["timestamp"], y=df["aqi"], label="Historical AQI")
sns.lineplot(x=pd.to_datetime(forecast_df["Date"]), y=forecast_df["Predicted AQI"], label="Forecast AQI")
plt.xticks(rotation=45)
plt.ylabel("AQI")
plt.xlabel("Date")
plt.legend()
plt.tight_layout()
st.pyplot(plt)

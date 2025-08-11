import streamlit as st
import pandas as pd
import joblib
import numpy as np
import datetime

# -------------------
# Load Model & Scaler
# -------------------
model = joblib.load("models/aqi_best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# -------------------
# Load Latest Data
# -------------------
latest_data = pd.read_csv("latest_pollutants.csv")

# Features for prediction
numeric_features = ["temperature", "humidity", "wind_speed", "pm2_5", "pm10", "co", "no2"]
lag_features = ["aqi_lag1", "pm2_5_lag1", "pm10_lag1", "co_lag1", "no2_lag1"]

# Scale numeric features
latest_data[numeric_features] = scaler.transform(latest_data[numeric_features])

# -------------------
# Predict Next 3 Days
# -------------------
predictions = []
current_features = latest_data.copy()

for i in range(1, 4):  # 3 days
    pred_aqi = model.predict(current_features[numeric_features + lag_features])[0]
    predictions.append({
        "Date": (datetime.datetime.now() + datetime.timedelta(days=i)).strftime("%Y-%m-%d"),
        "Predicted AQI": round(pred_aqi, 2)
    })

    # Update lag features for next day prediction
    current_features["aqi_lag1"] = pred_aqi
    # For simplicity, keep pollutant lag features same
    current_features["pm2_5_lag1"] = current_features["pm2_5"].iloc[0]
    current_features["pm10_lag1"] = current_features["pm10"].iloc[0]
    current_features["co_lag1"] = current_features["co"].iloc[0]
    current_features["no2_lag1"] = current_features["no2"].iloc[0]

# -------------------
# Streamlit UI
# -------------------
st.title("🌍 Lahore AQI Forecast Dashboard")
st.subheader("Latest Recorded AQI")
st.metric(label="Current AQI", value=round(latest_data["aqi"].iloc[0], 2))

st.subheader("📅 Next 3 Days Forecast")
df_pred = pd.DataFrame(predictions)
st.table(df_pred)

st.line_chart(df_pred.set_index("Date")["Predicted AQI"])

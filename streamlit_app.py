import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
import datetime

# -------------------
# Load Model & Scaler
# -------------------
model = joblib.load("models/aqi_best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# -------------------
# Load Full Historical Data
# -------------------
df = pd.read_csv("api.csv")
df = df.dropna(subset=["aqi"])  # ensure AQI is present
df = df.sort_values("timestamp").reset_index(drop=True)

# -------------------
# Outlier Capping
# -------------------
def cap_outliers(data, cols):
    for col in cols:
        Q1, Q3 = data[col].quantile(0.25), data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        data[col] = np.clip(data[col], lower, upper)
    return data

df = cap_outliers(df, ["pm2_5", "pm10", "wind_speed"])

# -------------------
# Scaling
# -------------------
numeric_features = ["temperature", "humidity", "wind_speed", "pm2_5", "pm10", "co", "no2"]
df[numeric_features] = scaler.transform(df[numeric_features])

# -------------------
# Lag Features
# -------------------
for col in ["aqi", "pm2_5", "pm10", "co", "no2"]:
    df[f"{col}_lag1"] = df[col].shift(1)

df = df.dropna().reset_index(drop=True)

# -------------------
# Prepare last row for forecasting
# -------------------
latest_row = df.iloc[-1].copy()
features = numeric_features + ["aqi_lag1", "pm2_5_lag1", "pm10_lag1", "co_lag1", "no2_lag1"]

# -------------------
# Forecast Next 3 Days
# -------------------
forecast_dates = []
forecast_aqi = []

current_date = datetime.datetime.strptime(latest_row["timestamp"], "%Y-%m-%d %H:%M:%S")

for i in range(1, 4):
    # Predict
    pred_aqi = model.predict([latest_row[features]])[0]
    forecast_aqi.append(round(pred_aqi, 2))
    forecast_dates.append((current_date + datetime.timedelta(days=i)).strftime("%Y-%m-%d"))

    # Update lag features for next prediction
    latest_row["aqi_lag1"] = pred_aqi
    latest_row["pm2_5_lag1"] = latest_row["pm2_5"]
    latest_row["pm10_lag1"] = latest_row["pm10"]
    latest_row["co_lag1"] = latest_row["co"]
    latest_row["no2_lag1"] = latest_row["no2"]

# -------------------
# Streamlit UI
# -------------------
st.title("🌍 Lahore AQI Dashboard")

st.subheader("Latest Recorded AQI")
st.metric(label="Current AQI", value=round(df.iloc[-1]["aqi"], 2))

st.subheader("📅 Next 3 Days Forecast")
forecast_df = pd.DataFrame({"Date": forecast_dates, "Predicted AQI": forecast_aqi})
st.dataframe(forecast_df)

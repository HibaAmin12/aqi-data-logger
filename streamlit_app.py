import streamlit as st
import pandas as pd
import joblib
import numpy as np
import datetime
import os

# --------------------
# Preprocessing Function (copied from training)
# --------------------
def preprocess_data(df):
    # Lag features
    for col in ["aqi", "pm2_5", "pm10", "co", "no2"]:
        df[f"{col}_lag1"] = df[col].shift(1)

    df = df.dropna().reset_index(drop=True)

    # Outlier Capping
    def cap_outliers(data, cols):
        for col in cols:
            Q1, Q3 = data[col].quantile(0.25), data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            data[col] = np.clip(data[col], lower, upper)
        return data

    df = cap_outliers(df, ["pm2_5", "pm10", "wind_speed"])

    return df

# --------------------
# Load Model & Scaler
# --------------------
model = joblib.load("models/aqi_best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# --------------------
# Load Data
# --------------------
df = pd.read_csv("api.csv")  # Full historical data
df = df.dropna(subset=["aqi"])
df = preprocess_data(df)

numeric_features = ["temperature", "humidity", "wind_speed", "pm2_5", "pm10", "co", "no2"]

# Scaling
df_scaled = df.copy()
df_scaled[numeric_features] = scaler.transform(df_scaled[numeric_features])

# --------------------
# Forecasting Next 3 Days
# --------------------
latest_row = df_scaled.iloc[-1].copy()
predictions = []
dates = []

for i in range(1, 4):
    X_input = latest_row[numeric_features + ["aqi_lag1", "pm2_5_lag1", "pm10_lag1", "co_lag1", "no2_lag1"]].values.reshape(1, -1)
    pred_aqi = model.predict(X_input)[0]
    predictions.append(round(pred_aqi, 2))
    dates.append((datetime.date.today() + datetime.timedelta(days=i)).strftime("%Y-%m-%d"))

    # Shift lag values for next prediction
    latest_row["aqi_lag1"] = pred_aqi
    latest_row["pm2_5_lag1"] = latest_row["pm2_5"]
    latest_row["pm10_lag1"] = latest_row["pm10"]
    latest_row["co_lag1"] = latest_row["co"]
    latest_row["no2_lag1"] = latest_row["no2"]

# --------------------
# Streamlit Dashboard
# --------------------
st.title("🌍 Lahore AQI Dashboard")

# Latest AQI
st.subheader("Latest Recorded AQI")
st.metric(label="Current AQI", value=round(df.iloc[-1]["aqi"], 2))

# Forecast
st.subheader("📅 Next 3 Days Forecast")
forecast_df = pd.DataFrame({"Date": dates, "Predicted AQI": predictions})
st.dataframe(forecast_df, use_container_width=True)

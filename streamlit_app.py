import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta

# --------------------
# Load model and scaler
# --------------------
model = joblib.load("models/aqi_best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# --------------------
# Load latest pollutant data (latest_pollutants.csv)
# --------------------
latest_data = pd.read_csv("latest_pollutants.csv")

# Columns to scale (same as training)
numeric_features = ["temperature", "humidity", "wind_speed", "pm2_5", "pm10", "co", "no2"]

# --------------------
# Prepare input features for prediction
# --------------------
def prepare_features(input_df, prev_aqi, prev_pm2_5, prev_pm10, prev_co, prev_no2):
    df = input_df.copy()

    # Scale numeric features using loaded scaler
    df[numeric_features] = scaler.transform(df[numeric_features])

    # Add lag features
    df["aqi_lag1"] = prev_aqi
    df["pm2_5_lag1"] = prev_pm2_5
    df["pm10_lag1"] = prev_pm10
    df["co_lag1"] = prev_co
    df["no2_lag1"] = prev_no2

    features = numeric_features + ["aqi_lag1", "pm2_5_lag1", "pm10_lag1", "co_lag1", "no2_lag1"]
    return df[features]

# --------------------
# Predict AQI for next N days
# --------------------
def predict_next_days(latest_row, days=3):
    predictions = []

    prev_aqi = latest_row["aqi"]
    prev_pm2_5 = latest_row["pm2_5"]
    prev_pm10 = latest_row["pm10"]
    prev_co = latest_row["co"]
    prev_no2 = latest_row["no2"]

    current_date = pd.to_datetime(latest_row["timestamp"])

    for i in range(1, days + 1):
        next_date = current_date + timedelta(days=i)

        input_data = {
            "temperature": [latest_row["temperature"]],
            "humidity": [latest_row["humidity"]],
            "wind_speed": [latest_row["wind_speed"]],
            "pm2_5": [latest_row["pm2_5"]],
            "pm10": [latest_row["pm10"]],
            "co": [latest_row["co"]],
            "no2": [latest_row["no2"]],
        }
        input_df = pd.DataFrame(input_data)

        X_pred = prepare_features(input_df, prev_aqi, prev_pm2_5, prev_pm10, prev_co, prev_no2)

        pred_aqi = model.predict(X_pred)[0]

        predictions.append({
            "date": next_date.strftime("%Y-%m-%d"),
            "predicted_aqi": pred_aqi
        })

        # Update lag values for next prediction
        prev_aqi = pred_aqi
        prev_pm2_5 = latest_row["pm2_5"]
        prev_pm10 = latest_row["pm10"]
        prev_co = latest_row["co"]
        prev_no2 = latest_row["no2"]

    return pd.DataFrame(predictions)

# --------------------
# Streamlit UI
# --------------------
st.title("3-Day AQI Forecast Dashboard")

st.markdown("""
This dashboard predicts Air Quality Index (AQI) for the next 3 days using the trained model and latest pollutant/weather data.
""")

st.subheader("Latest Pollutant and Weather Data")
st.dataframe(latest_data)

pred_df = predict_next_days(latest_data.iloc[0])

st.subheader("AQI Predictions for Next 3 Days")
st.table(pred_df)

st.line_chart(pred_df.set_index("date")["predicted_aqi"])

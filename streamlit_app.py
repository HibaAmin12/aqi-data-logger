import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import timedelta
from model_train import preprocess_data  # same preprocessing as training

# --------------------
# Load Model & Scaler
# --------------------
model = joblib.load("models/aqi_best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# --------------------
# Load Latest Data
# --------------------
df_raw = pd.read_csv("latest_pollutants.csv")
df_all = pd.read_csv("eda_outputs/standardized_data.csv") if "eda_outputs/standardized_data.csv" else None

if df_all is not None:
    # Reverse scale numeric features back to original for consistency
    numeric_features = ["temperature", "humidity", "wind_speed", "pm2_5", "pm10", "co", "no2"]
    df_all[numeric_features] = scaler.inverse_transform(df_all[numeric_features])
    df, _ = preprocess_data(df_all)
else:
    df, _ = preprocess_data(df_raw)

# --------------------
# Streamlit Title
# --------------------
st.set_page_config(page_title="AQI Forecast Dashboard", layout="centered")
st.title("🌫 Air Quality Index Dashboard")

# --------------------
# Latest AQI
# --------------------
latest_aqi = df.iloc[-1]["aqi"]
st.metric("Latest Recorded AQI", f"{latest_aqi:.2f}")

# --------------------
# Forecast Next 3 Days
# --------------------
future_dates = pd.date_range(start=pd.to_datetime(df.iloc[-1]["timestamp"]) + timedelta(days=1), periods=3)

predictions = []
last_row = df.iloc[-1].copy()

for i in range(3):
    # Prepare features for prediction
    features = [
        last_row["temperature"], last_row["humidity"], last_row["wind_speed"],
        last_row["pm2_5"], last_row["pm10"], last_row["co"], last_row["no2"],
        last_row["aqi"], last_row["pm2_5"], last_row["pm10"], last_row["co"], last_row["no2"]
    ]
    features = np.array(features).reshape(1, -1)

    pred_aqi = model.predict(features)[0]
    predictions.append(pred_aqi)

    # Shift lags for next iteration
    last_row["aqi"] = pred_aqi
    last_row["aqi_lag1"] = pred_aqi
    last_row["pm2_5_lag1"] = last_row["pm2_5"]
    last_row["pm10_lag1"] = last_row["pm10"]
    last_row["co_lag1"] = last_row["co"]
    last_row["no2_lag1"] = last_row["no2"]

# --------------------
# Show Forecast Table
# --------------------
forecast_df = pd.DataFrame({
    "Date": future_dates.strftime("%Y-%m-%d"),
    "Predicted AQI": [round(v, 2) for v in predictions]
})

st.subheader("📅 Next 3 Days Forecast")
st.table(forecast_df)

# --------------------
# Plot
# --------------------
st.line_chart(forecast_df.set_index("Date"))

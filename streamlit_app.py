import streamlit as st
import pandas as pd
import joblib
import datetime as dt

# Load trained model
model = joblib.load("models/aqi_best_model.pkl")

# Title
st.title("🌍 Real-Time Air Quality Index (AQI) Prediction")

# Load processed data
try:
    df = pd.read_csv("processed_data.csv")
except FileNotFoundError:
    st.error("❌ processed_data.csv file not found! Upload it first.")
    st.stop()

# Show last few rows of data
st.subheader("📊 Latest Available Data")
st.dataframe(df.tail())

# Select only the features available in processed_data.csv
feature_columns = [
    "temperature", "humidity", "wind_speed", "pm2_5", "pm10", "co", "no2",
    "aqi_lag1", "pm2_5_lag1", "pm10_lag1", "co_lag1", "no2_lag1"
]

missing = [col for col in feature_columns if col not in df.columns]
if missing:
    st.error(f"❌ Missing required columns in processed_data.csv: {missing}")
    st.stop()

# Get the latest row
latest_data = df.iloc[-1][feature_columns].values.reshape(1, -1)

# Predict AQI
pred_aqi = model.predict(latest_data)[0]

# Show result
st.subheader("📈 Predicted AQI for Next Hour")
st.success(f"Predicted AQI: {pred_aqi:.2f}")

# Show forecast for next 3 hours (using last row repeatedly)
st.subheader("🔮 AQI Forecast for Next 3 Hours")
forecast = []
current_time = pd.to_datetime(df.iloc[-1]["timestamp"])
for i in range(1, 4):
    next_time = current_time + dt.timedelta(hours=i)
    pred = model.predict(latest_data)[0]
    forecast.append((next_time.strftime("%Y-%m-%d %H:%M"), pred))

forecast_df = pd.DataFrame(forecast, columns=["Time", "Predicted AQI"])
st.table(forecast_df)

# Plot forecast
st.line_chart(forecast_df.set_index("Time"))

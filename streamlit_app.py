import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime

# Load trained model
model = joblib.load("models/aqi_best_model.pkl")

st.title("🌍 Air Quality Index (AQI) Prediction")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("processed_data.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

data = load_data()

# Show latest data
st.subheader("📊 Latest Available Data")
st.write(data.tail(5))

# Today's AQI (last row of dataset)
latest_row = data.iloc[-1]
today_aqi = latest_row["AQI"]

st.subheader("🌟 Today's AQI")
st.metric("Current AQI", f"{today_aqi:.2f}")

# Prepare features for prediction
feature_cols = [col for col in data.columns if col not in ["date", "AQI"]]

X_latest = data[feature_cols].iloc[-1:].values  # last row for today

# Predict next 3 days
forecast_days = 3
predictions = []
dates = []

start_date = datetime.date.today()  # correct today's date

for i in range(forecast_days):
    # Predict next day's AQI
    pred_aqi = model.predict(X_latest)[0]

    # Store prediction
    next_date = start_date + datetime.timedelta(days=i + 1)
    dates.append(next_date)
    predictions.append(pred_aqi)

    # Feedback loop: use predicted AQI as feature for next day
    X_latest = X_latest.copy()
    if "AQI" in feature_cols:
        X_latest[0][feature_cols.index("AQI")] = pred_aqi

# Show forecast
forecast_df = pd.DataFrame({
    "Date": dates,
    "Predicted AQI": np.round(predictions, 2)
})

st.subheader("📅 Next 3 Days AQI Forecast")
st.write(forecast_df)

import streamlit as st
import pandas as pd
import joblib
import datetime
import numpy as np

st.set_page_config(page_title="Lahore AQI Dashboard", layout="wide")

@st.cache_data
def load_model_and_data():
    df = pd.read_csv("processed_data.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    model = joblib.load("models/aqi_best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return df, model, scaler

df, model, scaler = load_model_and_data()

def forecast_aqi(df, model, scaler, days=3):
    features_base = ["temperature", "humidity", "wind_speed", "pm2_5", "pm10", "co", "no2"]
    lag_features = ["aqi_lag1", "pm2_5_lag1", "pm10_lag1", "co_lag1", "no2_lag1"]
    
    forecast_results = []
    temp_df = df.copy()
    
    for _ in range(days):
        last_row = temp_df.iloc[-1].copy()
        base_vals = last_row[features_base].values.reshape(1, -1)
        base_vals_scaled = scaler.transform(base_vals)
        lag_vals = last_row[lag_features].values.reshape(1, -1)
        X_input = np.hstack([base_vals_scaled, lag_vals])
        
        pred = model.predict(X_input)[0]
        pred = round(float(pred), 2)
        
        next_date = last_row["timestamp"] + datetime.timedelta(days=1)
        new_row = last_row.copy()
        new_row["timestamp"] = next_date
        new_row["aqi"] = pred
        
        # Update lag features for next iteration
        new_row["aqi_lag1"] = last_row["aqi"]
        new_row["pm2_5_lag1"] = last_row["pm2_5"]
        new_row["pm10_lag1"] = last_row["pm10"]
        new_row["co_lag1"] = last_row["co"]
        new_row["no2_lag1"] = last_row["no2"]
        
        forecast_results.append({"Date": next_date.strftime("%Y-%m-%d"), "Predicted AQI": pred})
        temp_df = pd.concat([temp_df, pd.DataFrame([new_row])], ignore_index=True)
    
    return pd.DataFrame(forecast_results)

# Show today's AQI (last row's AQI)
st.title("🌍 Lahore AQI Dashboard")
st.subheader("📌 Today's AQI")
current_aqi = round(float(df.iloc[-1]["aqi"]), 2)
st.metric(label="Current AQI", value=current_aqi)

# Next 3 days forecast
try:
    forecast_df = forecast_aqi(df, model, scaler, days=3)
    st.subheader("📅 Next 3 Days AQI Forecast")
    st.table(forecast_df)
except Exception as e:
    st.error(f"Prediction failed: {e}")

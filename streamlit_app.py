import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

# --------------------
# Load model and scaler
# --------------------
model = joblib.load("models/aqi_best_model.pkl")

# Note: We assume scaler is fitted on training features (numeric_features)
# So save and load scaler in model_train code or recreate it here (fit on training data)
# For simplicity, you can save scaler during training and load here
# Here I will just create scaler instance and load params (if saved)
# If scaler is not saved, you must save it in your training script.

# For demo, create scaler with mean/std from training (replace with actual saved scaler)
scaler = StandardScaler()

# --------------------
# Load latest pollutant data (latest_pollutants.csv)
# --------------------
latest_data = pd.read_csv("latest_pollutants.csv")

# Columns to scale (same as training)
numeric_features = ["temperature", "humidity", "wind_speed", "pm2_5", "pm10", "co", "no2"]

# --------------------
# Function to create lag features for prediction
# --------------------
def create_lag_features(df):
    # Lag1 features are previous timestep's values
    lag_features = {}
    for col in ["aqi", "pm2_5", "pm10", "co", "no2"]:
        lag_features[f"{col}_lag1"] = df[col].values[-1]
    return lag_features

# --------------------
# Prepare input features for prediction
# --------------------
def prepare_features(input_df, prev_aqi, prev_pm2_5, prev_pm10, prev_co, prev_no2):
    df = input_df.copy()

    # Scale numeric features
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
# Main prediction loop for next 3 days
# --------------------
def predict_next_days(latest_row, days=3):
    predictions = []

    # Initialize with latest known values
    prev_aqi = latest_row["aqi"]
    prev_pm2_5 = latest_row["pm2_5"]
    prev_pm10 = latest_row["pm10"]
    prev_co = latest_row["co"]
    prev_no2 = latest_row["no2"]

    current_date = pd.to_datetime(latest_row["timestamp"])

    for i in range(1, days + 1):
        next_date = current_date + timedelta(days=i)

        # For demo, keep latest weather & pollution values same (ideally get forecasted weather)
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

        # Prepare features with lag values from previous prediction
        X_pred = prepare_features(input_df, prev_aqi, prev_pm2_5, prev_pm10, prev_co, prev_no2)

        # Predict AQI
        pred_aqi = model.predict(X_pred)[0]

        # Save prediction and update lag variables for next iteration
        predictions.append({
            "date": next_date.strftime("%Y-%m-%d"),
            "predicted_aqi": pred_aqi
        })

        # Update lag values - Here we simulate as if pollution params stay same (or you can update realistically)
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

# Show latest data
st.subheader("Latest Pollutant and Weather Data")
st.dataframe(latest_data)

# Important: fit scaler on training data mean/std before prediction
# So, you need to save scaler after training, for now simulate scaler params from training mean/std:

@st.cache_data
def load_scaler():
    # Load scaler params from file or define mean/std here
    # Replace these arrays with your actual scaler mean and scale from training
    scaler = StandardScaler()
    # example mean/std (replace with your scaler.mean_ and scaler.scale_)
    scaler.mean_ = np.array([25, 50, 3, 40, 50, 0.5, 15])
    scaler.scale_ = np.array([5, 10, 1.5, 15, 20, 0.1, 5])
    return scaler

scaler = load_scaler()

# Run prediction
pred_df = predict_next_days(latest_data.iloc[0])

st.subheader("AQI Predictions for Next 3 Days")
st.table(pred_df)

# Optionally, plot
st.line_chart(pred_df.set_index("date")["predicted_aqi"])

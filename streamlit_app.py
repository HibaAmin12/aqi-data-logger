import streamlit as st
import pandas as pd
import joblib
import os

# Load model
model_path = "models/aqi_best_model.pkl"
if not os.path.exists(model_path):
    st.error("❌ Trained model file not found. Please upload models/aqi_model.pkl")
    st.stop()

model = joblib.load(model_path)

# Load processed.csv
data_path = "processed_data.csv"
if not os.path.exists(data_path):
    st.error("❌ processed.csv file not found. Please upload data/processed.csv")
    st.stop()

df = pd.read_csv(data_path)

st.title("🌍 Air Quality Index (AQI) Prediction")

st.write("### Last 5 records from processed data:")
st.dataframe(df.tail())

# Ensure correct feature columns
# Replace this list with the SAME order as you used during training
FEATURES = ["pm2_5", "pm10", "no2", "co", "temp", "humidity", "pressure"]

missing_features = [f for f in FEATURES if f not in df.columns]
if missing_features:
    st.error(f"❌ Missing features in processed.csv: {missing_features}")
    st.stop()

X_latest = df[FEATURES].iloc[[-1]]  # last row with only required features

# Predict AQI
try:
    pred_aqi = model.predict(X_latest)[0]
    st.success(f"✅ Predicted AQI: {pred_aqi:.2f}")
except Exception as e:
    st.error(f"⚠️ Prediction failed: {str(e)}")

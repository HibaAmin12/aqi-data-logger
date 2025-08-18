import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/aqi_best_model.pkl")

st.title("🌍 Real-Time AQI Prediction App")

# Load latest processed data
try:
    df = pd.read_csv("processed_data.csv")

    if df.empty:
        st.error("❌ processed_data.csv is empty. Run data pipeline first.")
    else:
        # Latest row
        latest = df.iloc[-1]

        # ✅ Select same features as used in training
        feature_cols = [
            "temperature", "humidity", "wind_speed",
            "pm2_5", "pm10", "co", "no2",
            "aqi_lag1", "pm2_5_lag1", "pm10_lag1", "co_lag1", "no2_lag1"
        ]

        # Ensure all required features exist
        missing = [f for f in feature_cols if f not in df.columns]
        if missing:
            st.error(f"❌ Missing features in processed_data.csv: {missing}")
        else:
            X_latest = latest[feature_cols].values.reshape(1, -1)

            try:
                pred_aqi = model.predict(X_latest)[0]

                st.success(f"✅ Predicted AQI: {pred_aqi:.2f}")

                # Show latest data row
                st.subheader("📊 Latest Input Data")
                st.write(latest[feature_cols])

            except Exception as e:
                st.error(f"⚠️ Prediction failed: {e}")

except FileNotFoundError:
    st.error("❌ processed_data.csv not found. Run data pipeline first.")

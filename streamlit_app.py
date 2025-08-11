
import streamlit as st
import hopsworks
import pandas as pd
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# -------------------
# Hopsworks Connection
# -------------------
st.set_page_config(page_title="AQI Prediction Dashboard", layout="wide")

st.title("🌍 Pearls AQI Predictor")
st.markdown("Predicting next 3 days Air Quality Index using latest pollutant data.")

# Hopsworks credentials
PROJECT_NAME = "HOPSWORKS_PROJECT"
API_KEY = "HOPSWORKS_API_KEY"  # keep in secrets in real use

st.write("Connecting to Hopsworks...")
try:
    conn = hopsworks.login(project=PROJECT_NAME, api_key_value=API_KEY)
    fs = conn.get_feature_store()
    feature_group = fs.get_feature_group(name="aqi_features", version=1)
    hopsworks_df = feature_group.read()
    st.success("Connected to Hopsworks ✅")
except Exception as e:
    st.error(f"Failed to connect to Hopsworks: {e}")
    st.stop()

# -------------------
# Load latest_pollutants.csv
# -------------------
try:
    csv_df = pd.read_csv("latest_pollutants.csv")
    st.success("Loaded latest_pollutants.csv ✅")
except FileNotFoundError:
    st.error("❌ latest_pollutants.csv not found in directory.")
    st.stop()

# -------------------
# Combine datasets
# -------------------
# Make sure columns match
common_cols = [col for col in csv_df.columns if col in hopsworks_df.columns]
combined_df = pd.concat([
    hopsworks_df[common_cols],
    csv_df[common_cols]
], ignore_index=True)

# Keep only latest row for prediction
latest_data = combined_df.tail(1)

# -------------------
# Load trained model
# -------------------
try:
    model = joblib.load("aqi_best_model.pkl")
    st.success("Model loaded ✅")
except FileNotFoundError:
    st.error("❌ aqi_best_model.pkl not found.")
    st.stop()

# -------------------
# Predict next 3 days AQI
# -------------------
predictions = []
today = datetime.today()
for i in range(1, 4):
    future_date = today + timedelta(days=i)
    predicted_aqi = model.predict(latest_data)[0]
    predictions.append({
        "Date": future_date.strftime("%Y-%m-%d"),
        "Predicted AQI": round(predicted_aqi, 2)
    })

pred_df = pd.DataFrame(predictions)

# -------------------
# Show in Streamlit
# -------------------
st.subheader("📊 Next 3 Days AQI Prediction")
st.dataframe(pred_df, use_container_width=True)

# Plot
fig, ax = plt.subplots()
ax.plot(pred_df["Date"], pred_df["Predicted AQI"], marker="o", linestyle="-")
ax.set_title("Next 3 Days AQI Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Predicted AQI")
ax.grid(True)
st.pyplot(fig)

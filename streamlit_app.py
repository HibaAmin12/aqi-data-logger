import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# -------------------
# Streamlit Page Setup
# -------------------
st.set_page_config(page_title="AQI Prediction Dashboard", layout="wide")
st.title("🌍 Pearls AQI Predictor")
st.markdown("Predicting next 3 days Air Quality Index using latest pollutant data (Offline Mode).")

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
# Keep only latest row for prediction
# -------------------
latest_data = csv_df.tail(1)

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

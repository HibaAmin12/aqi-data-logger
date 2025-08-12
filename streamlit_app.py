import streamlit as st
import pandas as pd
import joblib
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def load_data_and_model():
    df = pd.read_csv("processed_data.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    latest = pd.read_csv("latest_pollutants.csv")
    model = joblib.load("models/aqi_best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return df, latest, model, scaler

df, latest, model, scaler = load_data_and_model()

def forecast_aqi(df, model, days=3):
    forecast_results = []
    temp_df = df.copy()
    for i in range(days):
        latest_data = temp_df.iloc[-1].copy()
        features = [
            "temperature", "humidity", "wind_speed",
            "pm2_5", "pm10", "co", "no2",
            "aqi_lag1", "pm2_5_lag1", "pm10_lag1", "co_lag1", "no2_lag1"
        ]
        # Check if all features are in dataframe
        if not all(feat in latest_data.index for feat in features):
            st.error(f"Missing one or more required features for prediction: {features}")
            return pd.DataFrame()
        X_latest = latest_data[features].values.reshape(1, -1)
        predicted_aqi = model.predict(X_latest)[0]
        predicted_aqi = round(float(predicted_aqi), 2)
        next_date = latest_data["timestamp"] + datetime.timedelta(days=1)
        new_row = latest_data.copy()
        new_row["timestamp"] = next_date
        new_row["aqi"] = predicted_aqi
        # Update lag features for next iteration
        new_row["aqi_lag1"] = latest_data["aqi"]
        new_row["pm2_5_lag1"] = latest_data["pm2_5"]
        new_row["pm10_lag1"] = latest_data["pm10"]
        new_row["co_lag1"] = latest_data["co"]
        new_row["no2_lag1"] = latest_data["no2"]
        forecast_results.append({"Date": new_row["timestamp"].strftime("%Y-%m-%d"), "Predicted AQI": predicted_aqi})
        temp_df = pd.concat([temp_df, pd.DataFrame([new_row])], ignore_index=True)
    return pd.DataFrame(forecast_results)

forecast_df = forecast_aqi(df, model, days=3)

st.title("🌍 Lahore AQI Dashboard")
st.subheader("📌 Latest Recorded AQI")
st.metric("Current AQI", value=round(float(latest.iloc[0]['aqi']), 2))

st.subheader("📅 Next 3 Days Forecast")
if forecast_df.empty:
    st.warning("Forecast data is not available.")
else:
    st.table(forecast_df)

st.subheader("📈 Historical AQI & Forecast Trend")
plt.figure(figsize=(8, 4))
sns.lineplot(x=df["timestamp"], y=df["aqi"], label="Historical AQI")

if not forecast_df.empty and "Date" in forecast_df.columns and "Predicted AQI" in forecast_df.columns:
    sns.lineplot(x=pd.to_datetime(forecast_df["Date"]), y=forecast_df["Predicted AQI"], label="Forecast AQI")
else:
    st.warning("Cannot plot forecast trend due to missing forecast data.")

plt.xticks(rotation=45)
plt.ylabel("AQI")
plt.legend()
plt.tight_layout()
st.pyplot(plt)

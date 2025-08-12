import streamlit as st
import pandas as pd
import joblib
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost

@st.cache_data
def load_data_and_model():
    df = pd.read_csv("processed_data.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    latest = pd.read_csv("latest_pollutants.csv")
    model = joblib.load("models/aqi_best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return df, latest, model, scaler

df, latest, model, scaler = load_data_and_model()

st.write(f"XGBoost version: {xgboost.__version__}")
st.write(f"Model type: {type(model)}")

def forecast_aqi(df, model, scaler, days=3):
    forecast_results = []
    temp_df = df.copy()
    features = [
        "temperature", "humidity", "wind_speed",
        "pm2_5", "pm10", "co", "no2",
        "aqi_lag1", "pm2_5_lag1", "pm10_lag1", "co_lag1", "no2_lag1"
    ]
    for i in range(days):
        latest_data = temp_df.iloc[-1].copy()

        # Check if all features exist and print them
        missing_features = [f for f in features if f not in latest_data.index]
        if missing_features:
            st.error(f"Missing features in data: {missing_features}")
            return pd.DataFrame()

        st.write(f"Features used for prediction on day {i+1}:")
        st.write(latest_data[features])

        X_latest_raw = latest_data[features].values.reshape(1, -1)
        try:
            X_latest = scaler.transform(X_latest_raw)  # apply scaler before prediction
        except Exception as e:
            st.error(f"Scaler transform failed: {e}")
            return pd.DataFrame()

        st.write(f"Scaled features shape: {X_latest.shape}")

        try:
            predicted_aqi = model.predict(X_latest)[0]
            predicted_aqi = round(float(predicted_aqi), 2)
        except Exception as e:
            st.error(f"Model prediction failed: {e}")
            return pd.DataFrame()

        next_date = latest_data["timestamp"] + datetime.timedelta(days=1)
        new_row = latest_data.copy()
        new_row["timestamp"] = next_date
        new_row["aqi"] = predicted_aqi
        new_row["aqi_lag1"] = latest_data["aqi"]
        new_row["pm2_5_lag1"] = latest_data["pm2_5"]
        new_row["pm10_lag1"] = latest_data["pm10"]
        new_row["co_lag1"] = latest_data["co"]
        new_row["no2_lag1"] = latest_data["no2"]
        forecast_results.append({"Date": new_row["timestamp"].strftime("%Y-%m-%d"), "Predicted AQI": predicted_aqi})
        temp_df = pd.concat([temp_df, pd.DataFrame([new_row])], ignore_index=True)
    return pd.DataFrame(forecast_results)

forecast_df = forecast_aqi(df, model, scaler, days=3)

st.title("🌍 Lahore AQI Dashboard")
st.subheader("📌 Latest Recorded AQI")
st.metric("Current AQI", value=round(float(latest.iloc[0]['aqi']), 2))

st.subheader("📅 Next 3 Days Forecast")
st.table(forecast_df)

st.subheader("📈 Historical AQI & Forecast Trend")
plt.figure(figsize=(8, 4))
sns.lineplot(x=df["timestamp"], y=df["aqi"], label="Historical AQI")
sns.lineplot(x=pd.to_datetime(forecast_df["Date"]), y=forecast_df["Predicted AQI"], label="Forecast AQI")
plt.xticks(rotation=45)
plt.ylabel("AQI")
plt.legend()
plt.tight_layout()
st.pyplot(plt)

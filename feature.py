import pandas as pd
import os

# === Load raw data ===
df = pd.read_csv("api.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# === Sort by timestamp (important!) ===
df = df.sort_values("timestamp").reset_index(drop=True)

# === Feature Engineering ===
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["temp_diff"] = df["temperature"].diff().fillna(0)
df["humidity_diff"] = df["humidity"].diff().fillna(0)
df["wind_diff"] = df["wind_speed"].diff().fillna(0)

# === One-hot encode weather condition ===
df = pd.get_dummies(df, columns=["weather_main"])

# === Target variable (future AQI – 1 hour ahead) ===
df["target_aqi"] = df["aqi"].shift(-1)  # next hour's AQI is target
df = df.dropna()  # remove rows with NaN in target

# === Final features ===
feature_columns = [
    "timestamp", "temperature", "humidity", "wind_speed",
    "pm2_5", "pm10", "co", "no2", "aqi",  # current aqi (input)
    "hour", "day_of_week", "temp_diff", "humidity_diff", "wind_diff",
    "target_aqi"
] + [col for col in df.columns if col.startswith("weather_main_")]

df_final = df[feature_columns]

# === Save to Feature Store ===
feature_store_path = "feature_store.csv"
df_final.to_csv(feature_store_path, index=False)
print("✅ Features and target saved to feature_store.csv")

import os
import hopsworks
import pandas as pd

# Connect to Hopsworks
project = hopsworks.login(
    api_key_value=os.environ["HOPSWORKS_API_KEY"],
    project=os.environ["HOPSWORKS_PROJECT"],
    host=os.environ["HOPSWORKS_HOST"]
)
fs = project.get_feature_store()

# Load data
df = pd.read_csv("api.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.drop_duplicates(subset=["timestamp"])

float_cols = ["aqi", "temperature", "wind_speed", "pm2_5", "pm10", "co", "no2"]
df[float_cols] = df[float_cols].astype(float)
df["humidity"] = df["humidity"].astype(int)

# ✅ Try to get existing FG or create new
try:
    feature_group = fs.get_feature_group(name="aqi_features", version=1)
    print("✅ Found existing feature group.")
except:
    print("❗ Feature group not found. Creating new one...")
    feature_group = fs.create_feature_group(
        name="aqi_features",
        version=1,
        description="AQI and weather data from OpenWeather API",
        primary_key=["timestamp"],
        event_time="timestamp",
        online_enabled=False,
        schema=df  # ✅ define schema from dataframe
    )
    feature_group.save()

# ✅ Insert data
try:
    feature_group.insert(df)
    print(f"✅ Successfully inserted {len(df)} rows into Hopsworks Feature Store.")
except Exception as e:
    print("❌ Failed to insert data:", e)

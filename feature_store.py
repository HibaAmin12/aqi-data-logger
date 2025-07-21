import os
import hopsworks
import pandas as pd

# 🔐 Load credentials from environment
api_key = os.environ["HOPSWORKS_API_KEY"]
project_name = os.environ["HOPSWORKS_PROJECT"]
host = os.environ["HOPSWORKS_HOST"]

# ✅ Connect to Hopsworks
project = hopsworks.login(api_key_value=api_key, project=project_name, host=host)
fs = project.get_feature_store()

# 📥 Load data
df = pd.read_csv("api.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.drop_duplicates(subset=["timestamp"])

# ✅ Type conversion
float_cols = ["aqi", "temperature", "wind_speed", "pm2_5", "pm10", "co", "no2"]
df[float_cols] = df[float_cols].astype(float)
df["humidity"] = df["humidity"].astype(int)

# ✅ Check if feature group exists
try:
    feature_group = fs.get_feature_group(name="aqi_features", version=1)
    print("✅ Found existing feature group.")
except:
    print("❗ Feature group not found. Creating new one.")
    feature_group = fs.create_feature_group(
        name="aqi_features",
        version=1,
        description="AQI and weather data",
        primary_key=["timestamp"],
        event_time="timestamp",
        online_enabled=False
    )
    feature_group.save()
    print("✅ Feature group created.")

# ✅ Insert data
feature_group.insert(df)
print(f"✅ Successfully inserted {len(df)} rows into Hopsworks Feature Store.")

import os
import pandas as pd
import hopsworks

# ✅ Load API keys from GitHub secrets (set in yml)
api_key = os.environ["HOPSWORKS_API_KEY"]
project_name = os.environ["HOPSWORKS_PROJECT"]
host = os.environ["HOPSWORKS_HOST"]

# ✅ Connect to Hopsworks
project = hopsworks.login(api_key_value=api_key, project=project_name, host=host)
fs = project.get_feature_store()

# ✅ Load CSV
df = pd.read_csv("api.csv")

# ✅ Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])

# ✅ Add unique ID column to make each row unique
df.reset_index(drop=True, inplace=True)
df["id"] = df.index

# ✅ Correct data types
float_cols = ["aqi", "temperature", "wind_speed", "pm2_5", "pm10", "co", "no2"]
df[float_cols] = df[float_cols].astype(float)
df["humidity"] = df["humidity"].astype(int)
df["weather_main"] = df["weather_main"].astype(str)

# ✅ Create feature group if not exists
from hsfs.feature import Feature

try:
    fg = fs.get_feature_group(name="aqi_features", version=1)
    print("✅ Found existing feature group.")
except:
    print("ℹ️ Creating new feature group...")
    fg = fs.create_feature_group(
        name="aqi_features",
        version=1,
        description="AQI data with unique ID and timestamp",
        primary_key=["id"],
        event_time="timestamp",
        online_enabled=False,
        offline_enabled=True
    )
    fg.save()

# ✅ Insert full dataset (even with same timestamps)
try:
    fg.insert(df, write_options={"wait_for_job": True})
    print(f"✅ Successfully inserted {len(df)} rows into Hopsworks Feature Store.")
except Exception as e:
    print("❌ Failed to insert data.")
    print(f"👉 Error: {e}")

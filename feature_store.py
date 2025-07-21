import os
import hopsworks
import pandas as pd
from hsfs.feature_group import FeatureGroup

# Load secrets from environment
api_key = os.environ["HOPSWORKS_API_KEY"]
project_name = os.environ["HOPSWORKS_PROJECT"]
host = os.environ["HOPSWORKS_HOST"]

# Connect to Hopsworks
project = hopsworks.login(api_key_value=api_key, project=project_name, host=host)
fs = project.get_feature_store()

# Load data
df = pd.read_csv("api.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Drop duplicates
df = df.drop_duplicates(subset=["timestamp"])

# Type conversion
float_cols = ["aqi", "temperature", "wind_speed", "pm2_5", "pm10", "co", "no2"]
df[float_cols] = df[float_cols].astype(float)
df["humidity"] = df["humidity"].astype(int)

# Try to get or create feature group
try:
    feature_group = fs.get_feature_group(name="aqi_features", version=1)
except:
    print("🆕 Creating new feature group...")
    feature_group = fs.create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["timestamp"],
        description="AQI features from weather and pollution data",
        event_time="timestamp"
    )
    # 💾 Important: Save metadata first
    feature_group.save()

# ✅ Insert data
feature_group.insert(df)
print(f"✅ Inserted {len(df)} rows into Hopsworks Feature Store.")

import os
import hopsworks
import pandas as pd

# Load secrets from environment
api_key = os.environ["HOPSWORKS_API_KEY"]
project = os.environ["HOPSWORKS_PROJECT"]
host = os.environ["HOPSWORKS_HOST"]

# Connect to Hopsworks
project = hopsworks.login(api_key_value=api_key, project=project, host=host)
fs = project.get_feature_store()

# Read full api.csv
df = pd.read_csv("api.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# ✅ Remove duplicate timestamps (to avoid insert conflict)
df = df.drop_duplicates(subset=["timestamp"])

# ✅ Type conversion
float_cols = ["aqi", "temperature", "wind_speed", "pm2_5", "pm10", "co", "no2"]
df[float_cols] = df[float_cols].astype(float)
df["humidity"] = df["humidity"].astype(int)

# ✅ Get existing feature group
feature_group = fs.get_feature_group(name="aqi_features", version=1)

# ✅ Insert all unique rows
feature_group.insert(df)

print(f"✅ Successfully inserted {len(df)} rows into Hopsworks Feature Store.")

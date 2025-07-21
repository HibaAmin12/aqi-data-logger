import os
import hopsworks
import pandas as pd

# Load credentials
api_key = os.environ["HOPSWORKS_API_KEY"]
project = os.environ["HOPSWORKS_PROJECT"]
host = os.environ["HOPSWORKS_HOST"]

# Connect to Hopsworks
project = hopsworks.login(api_key_value=api_key, project=project, host=host)
fs = project.get_feature_store()

# Load Data
df = pd.read_csv("api.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.drop_duplicates(subset=["timestamp"])

float_cols = ["aqi", "temperature", "wind_speed", "pm2_5", "pm10", "co", "no2"]
df[float_cols] = df[float_cols].astype(float)
df["humidity"] = df["humidity"].astype(int)

# ❗ Delete old feature group manually from Hopsworks before running this script

# ✅ Create a fresh feature group
feature_group = fs.create_feature_group(
    name="aqi_features",
    version=1,
    description="AQI and weather data from OpenWeather API",
    primary_key=["timestamp"],
    event_time="timestamp",
    online_enabled=False,
    schema=df  # auto infer schema from df
)
feature_group.save()

# ✅ Insert data
feature_group.insert(df)
print(f"✅ Successfully inserted {len(df)} rows into Hopsworks Feature Store.")

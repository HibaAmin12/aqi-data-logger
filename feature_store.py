import os
import hopsworks
import pandas as pd
from hsfs.feature import Feature

# 🔐 Load credentials from GitHub Secrets
api_key       = os.environ["HOPSWORKS_API_KEY"]
project_name  = os.environ["HOPSWORKS_PROJECT"]
host          = os.environ["HOPSWORKS_HOST"]

# ✅ Login to Hopsworks
project = hopsworks.login(api_key_value=api_key, project=project_name, host=host)
fs = project.get_feature_store()

# ✅ Read CSV
df = pd.read_csv("api.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# ✅ Assign a unique ID to each row
df.reset_index(drop=True, inplace=True)
df["id"] = df.index + 1  # starts from 1

# ✅ Type conversions
float_cols = ["aqi", "temperature", "wind_speed", "pm2_5", "pm10", "co", "no2"]
df[float_cols] = df[float_cols].astype(float)
df["humidity"] = df["humidity"].astype(int)
df["weather_main"] = df["weather_main"].astype(str)

# ✅ Try to get existing feature group
try:
    fg = fs.get_feature_group("aqi_features", version=1)
    print("✅ Found existing feature group.")

except:
    # ❌ Not found: Create a new one
    print("❌ Feature group not found. Creating new one.")
    features = [
        Feature("id", "int"),
        Feature("timestamp", "timestamp"),
        Feature("aqi", "double"),
        Feature("temperature", "double"),
        Feature("humidity", "int"),
        Feature("wind_speed", "double"),
        Feature("weather_main", "string"),
        Feature("pm2_5", "double"),
        Feature("pm10", "double"),
        Feature("co", "double"),
        Feature("no2", "double")
    ]

    fg = fs.create_feature_group(
        name="aqi_features",
        version=1,
        description="AQI data collected from OpenWeather API",
        primary_key=["id"],  # 👈 Primary key changed from timestamp to id
        event_time="timestamp",
        features=features
    )
    fg.save()

# ✅ Insert ALL data from CSV (no filtering by timestamp now)
fg.insert(df, write_options={"wait_for_job": True})
print(f"✅ Inserted {len(df)} rows successfully into Hopsworks.")

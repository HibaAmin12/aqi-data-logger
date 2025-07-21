import os
import hopsworks
import pandas as pd
from hsfs.feature import Feature
from hsfs.feature_group import FeatureGroup

# 🔐 Load secrets
api_key = os.environ["HOPSWORKS_API_KEY"]
project = os.environ["HOPSWORKS_PROJECT"]
host = os.environ["HOPSWORKS_HOST"]

# 🔗 Connect to Hopsworks
project = hopsworks.login(api_key_value=api_key, project=project, host=host)
fs = project.get_feature_store()

# 📥 Load API data from CSV
df = pd.read_csv("api.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# ✅ Drop duplicate timestamps
df = df.drop_duplicates(subset=["timestamp"])

# 🔢 Type conversion
float_cols = ["aqi", "temperature", "wind_speed", "pm2_5", "pm10", "co", "no2"]
df[float_cols] = df[float_cols].astype(float)
df["humidity"] = df["humidity"].astype(int)

# ✅ Feature group name & version
fg_name = "aqi_features"
fg_version = 1

# ✅ Try to get existing feature group
try:
    feature_group = fs.get_feature_group(name=fg_name, version=fg_version)
    print("✅ Found existing feature group.")

except:
    print("❌ Feature group not found. Creating new one...")

    # Define schema from dataframe
    features = [
        Feature("timestamp", "timestamp"),
        Feature("temperature", "double"),
        Feature("humidity", "int"),
        Feature("wind_speed", "double"),
        Feature("weather_main", "string"),
        Feature("aqi", "double"),
        Feature("pm2_5", "double"),
        Feature("pm10", "double"),
        Feature("co", "double"),
        Feature("no2", "double"),
    ]

    # ✅ Create new feature group
    feature_group = fs.create_feature_group(
        name=fg_name,
        version=fg_version,
        description="AQI features from OpenWeather API",
        primary_key=["timestamp"],
        event_time="timestamp",
        features=features,
        online_enabled=True
    )

# ✅ Insert data
feature_group.insert(df, write_options={"wait_for_job": True})
print(f"✅ Successfully inserted {len(df)} rows into Hopsworks Feature Store.")

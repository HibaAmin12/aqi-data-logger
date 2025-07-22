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

# ✅ Assign unique ID
df.reset_index(drop=True, inplace=True)
df["id"] = df.index + 1

# ✅ Type casting (🛠 FIXED HERE)
float_cols = ["aqi", "temperature", "wind_speed", "pm2_5", "pm10", "co", "no2"]
df[float_cols] = df[float_cols].astype("float32")
df["humidity"] = df["humidity"].astype("int32")
df["id"] = df["id"].astype("int32")
df["weather_main"] = df["weather_main"].astype(str)

# ✅ Define schema
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

# ✅ Recreate feature group
print("🔁 Creating new feature group 'aqi_features'...")
fg = fs.create_feature_group(
    name="aqi_features",
    version=1,
    description="Fresh AQI dataset with id as primary key",
    primary_key=["id"],
    event_time="timestamp",
    features=features
)
fg.save()

# ✅ Insert all data
fg.insert(df, write_options={"wait_for_job": True})
print(f"✅ Successfully inserted {len(df)} rows into Hopsworks.")

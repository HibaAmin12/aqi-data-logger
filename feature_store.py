import os
import hopsworks
import pandas as pd
from hsfs.feature import Feature

# 🔐 Load secrets
api_key = os.environ["HOPSWORKS_API_KEY"]
project = os.environ["HOPSWORKS_PROJECT"]
host = os.environ["HOPSWORKS_HOST"]

# 🔗 Connect to Hopsworks
project = hopsworks.login(api_key_value=api_key, project=project, host=host)
fs = project.get_feature_store()

# 📥 Load data
df = pd.read_csv("api.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.drop_duplicates(subset=["timestamp"])

# 🧹 Type conversion
float_cols = ["aqi", "temperature", "wind_speed", "pm2_5", "pm10", "co", "no2"]
df[float_cols] = df[float_cols].astype(float)
df["humidity"] = df["humidity"].astype("int64")  # ✅ bigint (64-bit integer)

# ✅ Define schema
features = [
    Feature("timestamp", "timestamp"),
    Feature("temperature", "double"),
    Feature("humidity", "bigint"),  # ✅ Fixed here
    Feature("wind_speed", "double"),
    Feature("weather_main", "string"),
    Feature("aqi", "double"),
    Feature("pm2_5", "double"),
    Feature("pm10", "double"),
    Feature("co", "double"),
    Feature("no2", "double"),
]

# ✅ Get or create feature group
fg = fs.get_or_create_feature_group(
    name="aqi_features",
    version=1,
    description="Air Quality Features from API",
    primary_key=["timestamp"],
    event_time="timestamp",
    features=features,
    online_enabled=True
)

# ✅ Insert data
fg.insert(df, write_options={"wait_for_job": True})
print(f"✅ Inserted {len(df)} rows into Feature Store")

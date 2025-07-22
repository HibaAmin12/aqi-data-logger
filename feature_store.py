import os
import hopsworks
import pandas as pd
from hsfs.feature import Feature

# 🔐 Load credentials
api_key = os.environ["HOPSWORKS_API_KEY"]
project_name = os.environ["HOPSWORKS_PROJECT"]
host = os.environ["HOPSWORKS_HOST"]

# ✅ Login to Hopsworks
project = hopsworks.login(api_key_value=api_key, project=project_name, host=host)
fs = project.get_feature_store()

# ✅ Load CSV
df = pd.read_csv("api.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# ✅ Add unique ID
df.reset_index(drop=True, inplace=True)
df["id"] = df.index + 1

# ✅ Fix data types
float_cols = ["aqi", "temperature", "wind_speed", "pm2_5", "pm10", "co", "no2"]
df[float_cols] = df[float_cols].astype("float64")
df["humidity"] = df["humidity"].astype("int32")
df["id"] = df["id"].astype("int32")
df["weather_main"] = df["weather_main"].astype(str)

# ✅ Try to get feature group
try:
    fg = fs.get_feature_group("aqi_features", version=1)
    print("✅ Found existing feature group.")
except:
    # 🔁 Create feature group if not found
    print("🔁 Creating new feature group 'aqi_features'...")
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
        description="AQI dataset with primary key id",
        primary_key=["id"],
        event_time="timestamp",
        features=features
    )
    fg.save()
    print("✅ Feature group created.")

# ✅ Insert full data (you’re not filtering new entries, as discussed)
fg.insert(df, write_options={"wait_for_job": True})
print(f"✅ Successfully inserted {len(df)} rows.")

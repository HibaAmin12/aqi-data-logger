import os
import hopsworks
import pandas as pd

# 🔐 Load credentials
api_key = os.environ["HOPSWORKS_API_KEY"]
project_name = os.environ["HOPSWORKS_PROJECT"]
host = os.environ["HOPSWORKS_HOST"]

# 🔌 Connect to Hopsworks
project = hopsworks.login(api_key_value=api_key, project=project_name, host=host)
fs = project.get_feature_store()

# 📥 Read CSV
df = pd.read_csv("api.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.drop_duplicates(subset=["timestamp"])

# ✅ Type casting
float_cols = ["aqi", "temperature", "wind_speed", "pm2_5", "pm10", "co", "no2"]
df[float_cols] = df[float_cols].astype(float)
df["humidity"] = df["humidity"].astype(int)

# ✅ Check if feature group exists and is valid
feature_group = None
try:
    fg = fs.get_feature_group("aqi_features", version=1)
    if fg.features():  # Check if feature group has schema
        feature_group = fg
        print("✅ Feature group found and loaded.")
    else:
        print("⚠️ Feature group found but has no schema. Consider deleting from Hopsworks UI.")
except:
    print("❌ Feature group not found. Creating new one.")
    feature_group = fs.create_feature_group(
        name="aqi_features",
        version=1,
        description="Air Quality + Weather data",
        primary_key=["timestamp"],
        event_time="timestamp",
        online_enabled=False
    )
    feature_group.save()
    print("✅ New feature group created.")

# ✅ Only insert if feature group is ready
if feature_group:
    feature_group.insert(df)
    print(f"✅ Successfully inserted {len(df)} rows into Hopsworks Feature Store.")
else:
    print("❌ Feature group is None. Data not inserted.")

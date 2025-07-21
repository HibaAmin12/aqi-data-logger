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

# 📥 Load CSV
df = pd.read_csv("api.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.drop_duplicates(subset=["timestamp"])

# 🧹 Type casting
float_cols = ["aqi", "temperature", "wind_speed", "pm2_5", "pm10", "co", "no2"]
df[float_cols] = df[float_cols].astype(float)
df["humidity"] = df["humidity"].astype(int)

# ✅ Try to get feature group
try:
    fg = fs.get_feature_group("aqi_features", version=1)
    if fg is not None:
        print("✅ Feature group found.")
        fg.insert(df)
        print(f"✅ Inserted {len(df)} rows.")
except:
    print("❌ Feature group not found. Creating new one...")
    fg = fs.create_feature_group(
        name="aqi_features",
        version=1,
        description="AQI + Weather Data from OpenWeather API",
        primary_key=["timestamp"],
        event_time="timestamp",
        online_enabled=False,
        dataframe=df  # ✅ Supply DataFrame to define schema
    )
    fg.insert(df)
    print(f"✅ Feature group created and inserted {len(df)} rows.")

import os
import hopsworks
import pandas as pd

# 🔐 Load Hopsworks credentials
api_key = os.environ["HOPSWORKS_API_KEY"]
project_name = os.environ["HOPSWORKS_PROJECT"]
host = os.environ["HOPSWORKS_HOST"]

# ✅ Connect to Hopsworks
project = hopsworks.login(api_key_value=api_key, project=project_name, host=host)
fs = project.get_feature_store()

# ✅ Load data from api.csv
df = pd.read_csv("api.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# ✅ Remove duplicates
df = df.drop_duplicates(subset=["timestamp"])

# ✅ Convert data types
float_cols = ["aqi", "temperature", "wind_speed", "pm2_5", "pm10", "co", "no2"]
df[float_cols] = df[float_cols].astype(float)
df["humidity"] = df["humidity"].astype(int)

# ✅ Get the existing feature group (this must exist already)
try:
    feature_group = fs.get_feature_group(name="aqi_features", version=1)
    feature_group.insert(df)
    print(f"✅ Successfully inserted {len(df)} rows into Hopsworks Feature Store.")
except Exception as e:
    print("❌ Feature group not found or insert failed.")
    print("👉 Error:", e)

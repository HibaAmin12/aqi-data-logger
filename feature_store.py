import os
import hopsworks
import pandas as pd

# Load secrets
api_key = os.environ["HOPSWORKS_API_KEY"]
project = os.environ["HOPSWORKS_PROJECT"]
host = os.environ["HOPSWORKS_HOST"]

project = hopsworks.login(api_key_value=api_key, project=project, host=host)
fs = project.get_feature_store()

# ✅ Load data
df = pd.read_csv("api.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# ✅ Remove duplicate timestamps to avoid re-insert error
df = df.drop_duplicates(subset=["timestamp"])

# ✅ Type conversions
float_cols = ["aqi", "temperature", "wind_speed", "pm2_5", "pm10", "co", "no2"]
df[float_cols] = df[float_cols].astype(float)
df["humidity"] = df["humidity"].astype(int)

# ✅ Get Feature Group
feature_group = fs.get_feature_group(name="aqi_features", version=1)

# ✅ Insert all cleaned data
feature_group.insert(df)

print(f"✅ Inserted {len(df)} unique rows into Feature Store.")

import os
import hopsworks
import pandas as pd

# Load secrets
api_key = os.environ["HOPSWORKS_API_KEY"]
project = os.environ["HOPSWORKS_PROJECT"]
host = os.environ["HOPSWORKS_HOST"]

project = hopsworks.login(api_key_value=api_key, project=project, host=host)
fs = project.get_feature_store()

# ✅ Load data from API CSV (NOT dummy data)
df = pd.read_csv("api.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

float_cols = ["aqi", "temperature", "wind_speed", "pm2_5", "pm10", "co", "no2"]
df[float_cols] = df[float_cols].astype(float)
df["humidity"] = df["humidity"].astype(int)

feature_group = fs.get_feature_group(name="aqi_features", version=1)
feature_group.insert(df)

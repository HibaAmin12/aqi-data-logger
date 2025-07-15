import os
import hopsworks
import pandas as pd
from datetime import datetime
import requests

#  Load credentials from GitHub Secrets
api_key = os.environ["HOPSWORKS_API_KEY"]
project = os.environ["HOPSWORKS_PROJECT"]
host = os.environ["HOPSWORKS_HOST"]

# Login to Hopsworks
project = hopsworks.login(
    api_key_value=api_key,
    project=project,
    host=host
)

fs = project.get_feature_store()



df = pd.DataFrame([{
    "timestamp": datetime.utcnow(),
    "temperature": 32.5,
    "humidity": 78,
    "wind_speed": 2.4,
    "weather_main": "Clear",
    "aqi": 97,
    "pm2_5": 10.5,
    "pm10": 20.3,
    "co": 0.3,
    "no2": 8.9
}])

# Convert timestamp to proper datetime format
df["timestamp"] = pd.to_datetime(df["timestamp"])

#  Get existing feature group
feature_group = fs.get_feature_group(name="aqi_features", version=1)

# 🛠 Insert in batch mode to avoid Kafka issues
feature_group.insert(df, write_options={"wait_for_job": True})

import os
import pandas as pd
import hopsworks
from hsfs.feature import Feature

# 1) Load credentials
api_key      = os.environ["HOPSWORKS_API_KEY"]
project_name = os.environ["HOPSWORKS_PROJECT"]
host         = os.environ["HOPSWORKS_HOST"]

# 2) Login
project = hopsworks.login(api_key_value=api_key, project=project_name, host=host)
fs      = project.get_feature_store()

# 3) Read CSV and prepare DataFrame
df = pd.read_csv("api.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df.reset_index(drop=True, inplace=True)
df["id"] = df.index  # synthetic primary key

# Convert types
df["humidity"]     = df["humidity"].astype("int64")
for col in ["aqi", "temperature", "wind_speed", "pm2_5", "pm10", "co", "no2"]:
    df[col] = df[col].astype(float)
df["weather_main"] = df["weather_main"].astype(str)

# 4) Define schema manually
features = [
    Feature("id", "int"),
    Feature("timestamp", "timestamp"),
    Feature("aqi", "double"),
    Feature("temperature", "double"),
    Feature("humidity", "bigint"),
    Feature("wind_speed", "double"),
    Feature("weather_main", "string"),
    Feature("pm2_5", "double"),
    Feature("pm10", "double"),
    Feature("co", "double"),
    Feature("no2", "double")
]

# 5) Create feature group with schema
fg = fs.create_feature_group(
    name="aqi_features",
    version=1,
    description="AQI + weather data for training",
    primary_key=["id"],
    event_time="timestamp",
    features=features
)

# 6) Save schema and insert data
fg.save()
fg.insert(df, write_options={"wait_for_job": True})

print(f"✅ Successfully inserted {len(df)} rows into Hopsworks Feature Store.")

import os
import pandas as pd
import hopsworks

# 1) Load Hopsworks credentials
api_key      = os.environ["HOPSWORKS_API_KEY"]
project_name = os.environ["HOPSWORKS_PROJECT"]
host         = os.environ["HOPSWORKS_HOST"]

# 2) Connect
project = hopsworks.login(api_key_value=api_key, project=project_name, host=host)
fs      = project.get_feature_store()

# 3) Read and prepare your CSV
df = pd.read_csv("api.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df.reset_index(drop=True, inplace=True)
df["id"] = df.index  # synthetic primary key

# Ensure correct dtypes
df["humidity"]     = df["humidity"].astype("int64")
for c in ["aqi","temperature","wind_speed","pm2_5","pm10","co","no2"]:
    df[c] = df[c].astype(float)
df["weather_main"] = df["weather_main"].astype(str)

# 4) Create a brand-new feature group (this will succeed once)
fg = fs.create_feature_group(
    name="aqi_features",
    version=1,
    description="AQI & weather data with unique IDs",
    primary_key=["id"],
    event_time="timestamp",
    online_enabled=False,
    offline_enabled=True,
    dataframe=df   # <- exactly supply your DataFrame here so Hopsworks infers all columns
)
fg.save()
print("✅ Created new feature group for the first time.")

# 5) Insert all rows
fg.insert(df, write_options={"wait_for_job": True})
print(f"✅ Inserted {len(df)} rows into Hopsworks Feature Store.")

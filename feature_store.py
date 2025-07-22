import os
import pandas as pd
import hopsworks

# ── 1) Load Hopsworks credentials ──────────────────────────────────
api_key      = os.environ["HOPSWORKS_API_KEY"]
project_name = os.environ["HOPSWORKS_PROJECT"]
host         = os.environ["HOPSWORKS_HOST"]

# ── 2) Connect to Hopsworks ──────────────────────────────────────
project = hopsworks.login(
    api_key_value=api_key,
    project=project_name,
    host=host
)
fs = project.get_feature_store()

# ── 3) Read & prepare your CSV ───────────────────────────────────
df = pd.read_csv("api.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df.reset_index(drop=True, inplace=True)
df["id"] = df.index  # unique row id

# type conversion
float_cols = ["aqi", "temperature", "wind_speed", "pm2_5", "pm10", "co", "no2"]
df[float_cols] = df[float_cols].astype(float)
df["humidity"] = df["humidity"].astype("int64")
df["weather_main"] = df["weather_main"].astype(str)

# ── 4) Create (or get) the feature group, supplying dataframe for schema ──────
try:
    fg = fs.get_feature_group(name="aqi_features", version=1)
    print("✅ Found existing feature group.")
except Exception:
    print("⚠️ Feature group not found. Creating a new one with schema from DataFrame…")
    fg = fs.create_feature_group(
        name="aqi_features",
        version=1,
        description="AQI & weather data with unique IDs",
        primary_key=["id"],        # use our synthetic ID
        event_time="timestamp",    # still record event time
        online_enabled=False,
        offline_enabled=True,
        dataframe=df               # ← critical: provide DataFrame so Hopsworks infers schema
    )
    fg.save()
    print("✅ Feature group created.")

# ── 5) Insert all rows ───────────────────────────────────────────────────
fg.insert(df, write_options={"wait_for_job": True})
print(f"✅ Inserted {len(df)} rows into Hopsworks Feature Store.")

import os
import requests
import csv
import datetime

# ✅ Load API Key from Environment Variables
WAQI_API_TOKEN = os.getenv("WAQI_API_TOKEN")
aqicn_url = f"https://api.waqi.info/feed/Lahore/?token={WAQI_API_TOKEN}"

# ✅ Current Timestamp
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ✅ Fetch Data
aqicn_data = requests.get(aqicn_url).json()

# ✅ Extract Data
if aqicn_data['status'] == 'ok':
    aqicn_iaqi = aqicn_data['data'].get('iaqi', {})
    aqicn_data_clean = {
        "timestamp": now,
        "source": "AQICN",
        "temperature": None,
        "humidity": None,
        "wind_speed": None,
        "weather_main": None,
        "aqi": aqicn_data['data']['aqi'],
        "pm2_5": aqicn_iaqi.get('pm25', {}).get('v'),
        "pm10": aqicn_iaqi.get('pm10', {}).get('v'),
        "co": aqicn_iaqi.get('co', {}).get('v'),
        "no2": aqicn_iaqi.get('no2', {}).get('v')
    }
else:
    aqicn_data_clean = {
        "timestamp": now,
        "source": "AQICN",
        "temperature": None,
        "humidity": None,
        "wind_speed": None,
        "weather_main": None,
        "aqi": None,
        "pm2_5": None,
        "pm10": None,
        "co": None,
        "no2": None
    }

# ✅ Save to CSV
file_path = "api.csv"
file_exists = os.path.isfile(file_path)

with open(file_path, mode='a', newline='') as file:
    fieldnames = ["timestamp", "source", "temperature", "humidity", "wind_speed", "weather_main",
                  "aqi", "pm2_5", "pm10", "co", "no2"]
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()
    writer.writerow(aqicn_data_clean)

print("✅ AQICN data saved successfully!")

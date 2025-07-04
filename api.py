import os
import requests
import csv
import datetime

# Load API Keys from Environment Variables
OWM_API_KEY = os.getenv("API_KEY")
WAQI_API_TOKEN = os.getenv("WAQI_API_TOKEN")

LAT = "31.5497"
LON = "74.3436"

# API URLs
weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={OWM_API_KEY}&units=metric"
pollution_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={OWM_API_KEY}"
aqicn_url = f"https://api.waqi.info/feed/Lahore/?token={WAQI_API_TOKEN}"

# Current timestamp
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Fetch Weather Data
weather_data = requests.get(weather_url).json()
# Fetch OWM Air Pollution Data
pollution_data = requests.get(pollution_url).json()
# Fetch AQICN Data
aqicn_data = requests.get(aqicn_url).json()

# Extract Data from OWM (OpenWeatherMap)
owm_pm2_5 = pollution_data["list"][0]["components"]["pm2_5"]
owm_aqi = pollution_data["list"][0]["main"]["aqi"]
owm_data = {
    "temperature": weather_data["main"]["temp"],
    "humidity": weather_data["main"]["humidity"],
    "wind_speed": weather_data["wind"]["speed"],
    "weather_main": weather_data["weather"][0]["main"],
    "owm_aqi": owm_aqi,
    "owm_pm2_5": owm_pm2_5,
    "owm_pm10": pollution_data["list"][0]["components"]["pm10"],
    "owm_co": pollution_data["list"][0]["components"]["co"],
    "owm_no2": pollution_data["list"][0]["components"]["no2"]
}

# Extract Data from AQICN
if aqicn_data['status'] == 'ok':
    aqicn_iaqi = aqicn_data['data'].get('iaqi', {})
    aqicn_data_clean = {
        "aqicn_aqi": aqicn_data['data']['aqi'],
        "aqicn_pm2_5": aqicn_iaqi.get('pm25', {}).get('v'),
        "aqicn_pm10": aqicn_iaqi.get('pm10', {}).get('v'),
        "aqicn_co": aqicn_iaqi.get('co', {}).get('v'),
        "aqicn_no2": aqicn_iaqi.get('no2', {}).get('v')
    }
else:
    aqicn_data_clean = {
        "aqicn_aqi": None,
        "aqicn_pm2_5": None,
        "aqicn_pm10": None,
        "aqicn_co": None,
        "aqicn_no2": None
    }

# Combine All Data
final_data = {
    "timestamp": now,
    **owm_data,
    **aqicn_data_clean
}

# Save to CSV
os.makedirs("data", exist_ok=True)
file_path = "api.csv"
file_exists = os.path.isfile(file_path)

with open(file_path, mode='a', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=final_data.keys())
    if not file_exists:
        writer.writeheader()
    writer.writerow(final_data)

print("Data Saved Successfully!")

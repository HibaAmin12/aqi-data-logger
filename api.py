import os
import requests
import csv
import datetime

# ✅ Load API Keys from Environment Variables (GitHub Secrets)
API_KEY = os.getenv("API_KEY")  # OpenWeatherMap API Key
WAQI_API_TOKEN = os.getenv("WAQI_API_TOKEN")  # AQICN API Token

LAT = "31.5497"
LON = "74.3436"

# ✅ API URLs
weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
pollution_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}"
aqicn_url = f"https://api.waqi.info/feed/Lahore/?token={WAQI_API_TOKEN}"

# ✅ Current Timestamp
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ✅ Fetch Weather Data
weather_data = requests.get(weather_url).json()
# ✅ Fetch OpenWeatherMap Pollution Data
pollution_data = requests.get(pollution_url).json()
# ✅ Fetch AQICN Data
aqicn_data = requests.get(aqicn_url).json()

# ✅ Function to Calculate Continuous AQI from PM2.5
def calculate_pm25_aqi(pm):
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500)
    ]
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= pm <= c_high:
            return round(((i_high - i_low) / (c_high - c_low)) * (pm - c_low) + i_low, 2)
    return None

# ✅ Extract OpenWeatherMap Data
pm2_5 = pollution_data["list"][0]["components"]["pm2_5"]
aqi_value = calculate_pm25_aqi(pm2_5)  # ✅ Continuous AQI

owm_data = {
    "temperature": weather_data["main"]["temp"],
    "humidity": weather_data["main"]["humidity"],
    "wind_speed": weather_data["wind"]["speed"],
    "weather_main": weather_data["weather"][0]["main"],
    "aqi": aqi_value,  # ✅ Continuous AQI here
    "pm2_5": pm2_5,
    "pm10": pollution_data["list"][0]["components"]["pm10"],
    "co": pollution_data["list"][0]["components"]["co"],
    "no2": pollution_data["list"][0]["components"]["no2"]
}

# ✅ Combine All Data
final_data = {
    "timestamp": now,
    **owm_data
}

# ✅ Save to CSV (safe)
file_path = "api.csv"
file_exists = os.path.isfile(file_path)

with open(file_path, mode='a', newline='') as file:
    fieldnames = ["timestamp", "temperature", "humidity", "wind_speed", "weather_main",
                  "aqi", "pm2_5", "pm10", "co", "no2"]
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()
    writer.writerow(final_data)

print("Data Saved Successfully!")

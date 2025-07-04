import os
import requests
import csv
import datetime

# ✅ Load API Key from Environment Variables
API_KEY = os.getenv("API_KEY")  # OpenWeatherMap API Key
LAT = "31.5497"
LON = "74.3436"

# ✅ API URLs
weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
pollution_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}"

# ✅ Current Timestamp
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ✅ Fetch Data
weather_data = requests.get(weather_url).json()
pollution_data = requests.get(pollution_url).json()

# ✅ Continuous AQI from PM2.5
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

# ✅ Extract Data
pm2_5 = pollution_data["list"][0]["components"]["pm2_5"]
aqi_value = calculate_pm25_aqi(pm2_5)

owm_data = {
    "timestamp": now,
    "source": "OpenWeatherMap",
    "temperature": weather_data["main"]["temp"],
    "humidity": weather_data["main"]["humidity"],
    "wind_speed": weather_data["wind"]["speed"],
    "weather_main": weather_data["weather"][0]["main"],
    "aqi": aqi_value,
    "pm2_5": pm2_5,
    "pm10": pollution_data["list"][0]["components"]["pm10"],
    "co": pollution_data["list"][0]["components"]["co"],
    "no2": pollution_data["list"][0]["components"]["no2"]
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
    writer.writerow(owm_data)

print("✅ OpenWeatherMap data saved successfully!")

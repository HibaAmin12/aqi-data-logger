import os
import requests
import csv
import datetime

# Get API Key
API_KEY = os.getenv("API_KEY")
LAT = "31.5497"
LON = "74.3436"

# API URLs
weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
pollution_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}"

# Get current timestamp
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Fetch data
weather_data = requests.get(weather_url).json()
pollution_data = requests.get(pollution_url).json()

# Extract PM2.5
pm2_5 = pollution_data["list"][0]["components"]["pm2_5"]

# AQI breakpoints for PM2.5
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

# Calculate actual AQI from PM2.5
aqi_value = calculate_pm25_aqi(pm2_5)

# Final data dictionary
data = {
    "timestamp": now,
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

# Write to CSV
file_path = "api.csv"
file_exists = os.path.isfile(file_path)

with open(file_path, mode='a', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=data.keys())
    if not file_exists:
        writer.writeheader()
    writer.writerow(data)

print("✅ OpenWeatherMap AQI & Weather data saved successfully!")

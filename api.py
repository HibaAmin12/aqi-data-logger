import os
import requests
import csv
import datetime

API_KEY = os.getenv("API_KEY")
LAT = "31.5497"
LON = "74.3436"

# Weather API
weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
pollution_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}"

now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

weather_data = requests.get(weather_url).json()
pollution_data = requests.get(pollution_url).json()

data = {
    "timestamp": now,
    "temperature": weather_data["main"]["temp"],
    "humidity": weather_data["main"]["humidity"],
    "wind_speed": weather_data["wind"]["speed"],
    "weather_main": weather_data["weather"][0]["main"],
    "aqi": pollution_data["list"][0]["main"]["aqi"],
    "pm2_5": pollution_data["list"][0]["components"]["pm2_5"],
    "pm10": pollution_data["list"][0]["components"]["pm10"],
    "co": pollution_data["list"][0]["components"]["co"],
    "no2": pollution_data["list"][0]["components"]["no2"]
}

file_path = "api.csv"
file_exists = os.path.isfile(file_path)

with open(file_path, mode='a', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=data.keys())
    if not file_exists:
        writer.writeheader()
    writer.writerow(data) isko edit kr k do

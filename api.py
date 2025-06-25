import os
import requests
import csv
import datetime

# Load API key from environment variable
API_KEY = os.getenv("API_KEY")

# Lahore coordinates
LAT = "31.5497"
LON = "74.3436"

# API URLs
weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
pollution_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}"

# Current timestamp
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Fetch data
weather_data = requests.get(weather_url).json()
pollution_data = requests.get(pollution_url).json()

# Extract values
temperature = weather_data["main"]["temp"]
humidity = weather_data["main"]["humidity"]
wind_speed = weather_data["wind"]["speed"]
weather_main = weather_data["weather"][0]["main"]

aqi = pollution_data["list"][0]["main"]["aqi"]
pm2_5 = pollution_data["list"][0]["components"]["pm2_5"]
pm10 = pollution_data["list"][0]["components"]["pm10"]
co = pollution_data["list"][0]["components"]["co"]
no2 = pollution_data["list"][0]["components"]["no2"]

# Function to get AQI index & category from PM2.5
def get_aqi_info(pm25):
    if pm25 <= 12.0:
        return 1, 'Good'
    elif pm25 <= 35.4:
        return 2, 'Moderate'
    elif pm25 <= 55.4:
        return 3, 'Unhealthy for Sensitive Groups'
    elif pm25 <= 150.4:
        return 4, 'Unhealthy'
    elif pm25 <= 250.4:
        return 5, 'Very Unhealthy'
    else:
        return 6, 'Hazardous'

# Calculate from PM2.5
aqi_index, aqi_category = get_aqi_info(pm2_5)

# Prepare data row
data = {
    "timestamp": now,
    "temperature": temperature,
    "humidity": humidity,
    "wind_speed": wind_speed,
    "weather_main": weather_main,
    "aqi": aqi,
    "pm2_5": pm2_5,
    "pm10": pm10,
    "co": co,
    "no2": no2,
    "aqi_index_from_pm25": aqi_index,
    "aqi_category_from_pm25": aqi_category
}

# CSV file path
file_path = "api.csv"
file_exists = os.path.isfile(file_path)

# Write to CSV
with open(file_path, mode='a', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=data.keys())
    if not file_exists:
        writer.writeheader()
    writer.writerow(data)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("api.csv")  # Or load from Hopsworks if needed
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Check basic info
print(df.info())
print(df.describe())

# Heatmap - correlation
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Scatter plots
sns.pairplot(df[["temperature", "humidity", "wind_speed", "pm2_5", "pm10", "aqi"]])
plt.suptitle("Pairwise Feature Relationships", y=1.02)
plt.show()

# AQI trend over time
plt.figure(figsize=(12,5))
sns.lineplot(data=df, x="timestamp", y="aqi")
plt.title("AQI Trend Over Time")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 📁 Create directory for EDA output
os.makedirs("eda_outputs", exist_ok=True)

# 📊 Load data
df = pd.read_csv("api.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# 📝 Save basic info
with open("eda_outputs/basic_info.txt", "w") as f:
    df.info(buf=f)
    f.write("\n\n")
    f.write(str(df.describe()))

# 🔥 Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("eda_outputs/correlation_heatmap.png")
plt.close()

# 🔗 Pairplot
sns.pairplot(df[["temperature", "humidity", "wind_speed", "pm2_5", "pm10", "aqi"]])
plt.suptitle("Pairwise Feature Relationships", y=1.02)
plt.savefig("eda_outputs/pairwise_features.png")
plt.close()

# 📈 AQI Trend
plt.figure(figsize=(12, 5))
sns.lineplot(data=df, x="timestamp", y="aqi")
plt.title("AQI Trend Over Time")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("eda_outputs/aqi_trend.png")
plt.close()

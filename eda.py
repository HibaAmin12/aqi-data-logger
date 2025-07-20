import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ✅ Make sure output folder exists
os.makedirs("eda_outputs", exist_ok=True)

# Load data
df = pd.read_csv("api.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Basic info
print(df.info())
print(df.describe())

# ✅ Numeric-only correlation
numeric_df = df.select_dtypes(include='number')

# Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("eda_outputs/correlation_heatmap.png")

# Pairplot
sns.pairplot(numeric_df)
plt.suptitle("Pairwise Feature Relationships", y=1.02)
plt.savefig("eda_outputs/pairplot.png")

# AQI trend
plt.figure(figsize=(12,5))
sns.lineplot(data=df, x="timestamp", y="aqi")
plt.title("AQI Trend Over Time")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("eda_outputs/aqi_trend.png")

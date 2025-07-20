import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("api.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Info + summary
print(df.info())
print(df.describe())

# ✅ Select only numeric columns for heatmap
numeric_df = df.select_dtypes(include='number')

# Heatmap - correlation
plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("eda_outputs/correlation_heatmap.png")  # 👈 Save instead of show

# Scatter plots
sns.pairplot(numeric_df)
plt.suptitle("Pairwise Feature Relationships", y=1.02)
plt.savefig("eda_outputs/pairplot.png")

# AQI trend over time
plt.figure(figsize=(12,5))
sns.lineplot(data=df, x="timestamp", y="aqi")
plt.title("AQI Trend Over Time")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("eda_outputs/aqi_trend.png")

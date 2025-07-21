import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import hopsworks

os.makedirs("eda_outputs", exist_ok=True)

# ✅ Connect to Hopsworks
api_key = os.environ["HOPSWORKS_API_KEY"]
project = os.environ["HOPSWORKS_PROJECT"]
host = os.environ["HOPSWORKS_HOST"]

project = hopsworks.login(api_key_value=api_key, project=project, host=host)
fs = project.get_feature_store()

# ✅ Read from feature store
feature_group = fs.get_feature_group(name="aqi_features", version=1)
df = feature_group.read()

# ✅ EDA as before
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.dropna(subset=["aqi"])
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

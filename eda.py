import os
import hopsworks
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sweetviz as sv

# 🔐 Credentials from GitHub Secrets
api_key = os.environ["HOPSWORKS_API_KEY"]
project = os.environ["HOPSWORKS_PROJECT"]
host = os.environ["HOPSWORKS_HOST"]

# 🔁 Login and get feature group
project = hopsworks.login(api_key_value=api_key, project=project, host=host)
fs = project.get_feature_store()
fg = fs.get_feature_group("aqi_features", version=1)

# 📥 Read full data
df = fg.read()
df["timestamp"] = pd.to_datetime(df["timestamp"])

# 📁 Save plots
os.makedirs("eda_outputs", exist_ok=True)

# 📊 Scatter Plot: AQI vs Temperature
plt.figure(figsize=(8, 6))
sns.scatterplot(x="temperature", y="aqi", data=df)
plt.title("AQI vs Temperature")
plt.savefig("eda_outputs/aqi_vs_temp.png")
plt.close()

# 📊 Heatmap: Correlation
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.savefig("eda_outputs/heatmap.png")
plt.close()

# 📊 Boxplot: AQI
plt.figure(figsize=(8, 6))
sns.boxplot(y="aqi", data=df)
plt.title("Box Plot - AQI")
plt.savefig("eda_outputs/boxplot_aqi.png")
plt.close()

# 📋 Sweetviz EDA
report = sv.analyze(df)
report.show_html("eda_report.html")

print("✅ EDA report generated and saved.")

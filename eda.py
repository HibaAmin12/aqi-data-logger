import os
import hopsworks
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sweetviz as sv

# Create output directory
os.makedirs("eda_outputs", exist_ok=True)

# 🔐 Load secrets
api_key = os.environ["HOPSWORKS_API_KEY"]
project = os.environ["HOPSWORKS_PROJECT"]
host = os.environ["HOPSWORKS_HOST"]

# ✅ Connect to Hopsworks
project = hopsworks.login(api_key_value=api_key, project=project, host=host)
fs = project.get_feature_store()

# ✅ Load full feature group
fg = fs.get_feature_group("aqi_features", version=1)
df = fg.read()

# ✅ Convert timestamp if needed
df["timestamp"] = pd.to_datetime(df["timestamp"])

# ✅ Save CSV snapshot (optional)
df.to_csv("eda_outputs/aqi_snapshot.csv", index=False)

# ✅ Sweetviz Report
print("📊 Generating Sweetviz report...")
report = sv.analyze(df)
report_path = "eda_outputs/eda_report.html"
report.show_html(filepath=report_path)
print("✅ Sweetviz report saved to", report_path)

# ✅ Heatmap
print("📌 Creating heatmap...")
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("eda_outputs/heatmap.png")
plt.close()

# ✅ Box plots (AQI vs all numeric features except AQI & ID)
numeric_cols = ['temperature', 'humidity', 'wind_speed', 'pm2_5', 'pm10', 'co', 'no2']

for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.tight_layout()
    plt.savefig(f"eda_outputs/boxplot_{col}.png")
    plt.close()

# ✅ Scatter plots (AQI vs features)
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=df[col], y=df["aqi"])
    plt.title(f"AQI vs {col}")
    plt.xlabel(col)
    plt.ylabel("AQI")
    plt.tight_layout()
    plt.savefig(f"eda_outputs/scatter_aqi_vs_{col}.png")
    plt.close()

print("✅ All EDA plots saved in eda_outputs/")

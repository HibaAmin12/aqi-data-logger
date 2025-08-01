import os
import hopsworks
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sweetviz as sv
import numpy as np
from sklearn.preprocessing import StandardScaler

# -------------------
# Hopsworks Login
# -------------------
api_key = os.environ["HOPSWORKS_API_KEY"]
project_name = os.environ["HOPSWORKS_PROJECT"]
host = os.environ["HOPSWORKS_HOST"]

project = hopsworks.login(api_key_value=api_key, project=project_name, host=host)
fs = project.get_feature_store()

# -------------------
# Load Data
# -------------------
fg = fs.get_feature_group("aqi_features", version=1)
df = fg.read()
df = df.dropna(subset=["aqi"])

# -------------------
# Create EDA outputs folder
# -------------------
os.makedirs("eda_outputs", exist_ok=True)

# -------------------
# Summary
# -------------------
df.describe().to_csv("eda_outputs/statistical_summary.csv")

# -------------------
# Sweetviz Report
# -------------------
report = sv.analyze(df)
report.show_html("eda_outputs/sweetviz_report.html")

# -------------------
# Correlation Heatmap
# -------------------
numeric_cols = ["aqi", "temperature", "humidity", "wind_speed", "pm2_5", "pm10", "co", "no2"]
plt.figure(figsize=(10, 7))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("eda_outputs/correlation_heatmap.png")
plt.close()

# -------------------
# Boxplots
# -------------------
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(y=df[col])
    plt.title(f"Boxplot - {col}")
    plt.savefig(f"eda_outputs/boxplot_{col}.png")
    plt.close()

# -------------------
# Histograms
# -------------------
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(f"Distribution of {col}")
    plt.savefig(f"eda_outputs/histogram_{col}.png")
    plt.close()

# -------------------
# Scatter Plots (AQI vs Features)
# -------------------
for col in numeric_cols:
    if col != "aqi":
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=df[col], y=df["aqi"])
        plt.title(f"AQI vs {col}")
        plt.savefig(f"eda_outputs/scatter_aqi_vs_{col}.png")
        plt.close()

# -------------------
# Outlier Capping
# -------------------
def cap_outliers(df, cols):
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] > upper, upper,
                           np.where(df[col] < lower, lower, df[col]))
    return df

df_capped = cap_outliers(df.copy(), ["pm2_5", "pm10", "wind_speed"])

# -------------------
# Standardization
# -------------------
features_to_scale = ["temperature", "humidity", "wind_speed", "pm2_5", "pm10", "co", "no2"]
scaler = StandardScaler()
df_scaled = df_capped.copy()
df_scaled[features_to_scale] = scaler.fit_transform(df_capped[features_to_scale])

df_scaled.to_csv("eda_outputs/standardized_data.csv", index=False)

print("EDA Completed! Reports & plots saved in eda_outputs/")

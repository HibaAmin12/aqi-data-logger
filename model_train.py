import os
import hopsworks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# --------------------
# Hopsworks Login
# --------------------
api_key = os.environ["HOPSWORKS_API_KEY"]
project_name = os.environ["HOPSWORKS_PROJECT"]
host = os.environ["HOPSWORKS_HOST"]

project = hopsworks.login(api_key_value=api_key, project=project_name, host=host)
fs = project.get_feature_store()

# --------------------
# Load Data from Feature Store
# --------------------
fg = fs.get_feature_group("aqi_features", version=1)
df = fg.read().sort_values("timestamp")
df = df.dropna(subset=["aqi"])

# --------------------
# Add Lag Features
# --------------------
df["aqi_lag1"] = df["aqi"].shift(1)
df["pm2_5_lag1"] = df["pm2_5"].shift(1)
df["pm10_lag1"] = df["pm10"].shift(1)
df["co_lag1"] = df["co"].shift(1)
df["no2_lag1"] = df["no2"].shift(1)

df = df.dropna().reset_index(drop=True)

# --------------------
# Outlier Capping
# --------------------
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

numeric_features = ["temperature", "humidity", "wind_speed", "pm2_5", "pm10", "co", "no2"]
df = cap_outliers(df, ["pm2_5", "pm10", "wind_speed"])

# --------------------
# Standardization
# --------------------
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# --------------------
# Features and Target
# --------------------
features = numeric_features + ["aqi_lag1", "pm2_5_lag1", "pm10_lag1", "co_lag1", "no2_lag1"]
X = df[features]
y = df["aqi"]

# --------------------
# Train-Test Split
# --------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------
# Train Multiple Models
# --------------------
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.01),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

results = {}
best_model_name = None
best_model = None
best_r2 = -1

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    corr = np.corrcoef(y_test, y_pred)[0, 1]

    results[name] = {"MSE": mse, "R²": r2, "Correlation": corr}

    # Track best model based on R²
    if r2 > best_r2:
        best_r2 = r2
        best_model_name = name
        best_model = model

# --------------------
# Save Results & Plots
# --------------------
results_df = pd.DataFrame(results).T
print("\n📊 Model Performance:\n")
print(results_df)

os.makedirs("model_outputs", exist_ok=True)
results_df.to_csv("model_outputs/model_results.csv")

# R² Plot
plt.figure(figsize=(8, 5))
plt.bar(results_df.index, results_df["R²"], color='skyblue')
plt.title("R² Score Comparison of Models")
plt.ylabel("R² Score")
plt.xticks(rotation=30)
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig("model_outputs/r2_comparison.png")
plt.close()

# MSE Plot
plt.figure(figsize=(8, 5))
plt.bar(results_df.index, results_df["MSE"], color='salmon')
plt.title("MSE Comparison of Models")
plt.ylabel("Mean Squared Error")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("model_outputs/mse_comparison.png")
plt.close()

# --------------------
# Save Best Model Automatically
# --------------------
print(f"\n Best model selected: {best_model_name} (R²: {best_r2:.4f})")
best_model.fit(X, y)

os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/aqi_best_model.pkl")
print(" Best model saved to 'models/aqi_best_model.pkl'")

# --------------------
# Save Latest Row for Streamlit
# --------------------
latest = df.iloc[-1]
latest.to_frame().T.to_csv("latest_pollutants.csv", index=False)
print(" Latest pollutants and AQI saved to 'latest_pollutants.csv'")

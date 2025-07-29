import os
import hopsworks
import pandas as pd
import numpy as np
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

# 🔐 Load credentials
api_key = os.environ["HOPSWORKS_API_KEY"]
project_name = os.environ["HOPSWORKS_PROJECT"]
host = os.environ["HOPSWORKS_HOST"]

# ✅ Connect to Hopsworks
project = hopsworks.login(api_key_value=api_key, project=project_name, host=host)
fs = project.get_feature_store()
fg = fs.get_feature_group("aqi_features", version=1)

# ✅ Load data from feature store
df = fg.read()
df.dropna(inplace=True)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# ✅ Features & Target (Create AQI targets for next 3 days)
features = ["temperature", "humidity", "wind_speed", "weather_main", "pm2_5", "pm10", "co", "no2"]
df = df.sort_values("timestamp").reset_index(drop=True)

# Shift AQI to create Day 1, Day 2, Day 3 targets
df["aqi_day1"] = df["aqi"].shift(-1)
df["aqi_day2"] = df["aqi"].shift(-2)
df["aqi_day3"] = df["aqi"].shift(-3)
df.dropna(inplace=True)

X = df[features]
y = df[["aqi_day1", "aqi_day2", "aqi_day3"]]

# ✅ One-hot encode weather_main
X = pd.get_dummies(X, columns=["weather_main"], drop_first=True)

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Multi-output models
models = {
    "RandomForest": MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42)),
    "GradientBoosting": MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, random_state=42))
}

results = {}
best_model = None
best_score = -float("inf")

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds, multioutput="uniform_average")
    corr_coef = np.corrcoef(y_test.values.flatten(), preds.flatten())[0, 1]
    pearson_corr, _ = pearsonr(y_test.values.flatten(), preds.flatten())

    results[name] = {
        "mse": mse,
        "r2": r2,
        "corr_coef": corr_coef,
        "pearson": pearson_corr
    }

    if r2 > best_score:
        best_score = r2
        best_model = model
        best_name = name
        best_preds = preds

# ✅ Save outputs
os.makedirs("model_outputs", exist_ok=True)
joblib.dump(best_model, "model_outputs/best_model.pkl")

with open("model_outputs/model_metrics.json", "w") as f:
    json.dump(results[best_name], f, indent=4)

pred_df = pd.DataFrame(y_test, columns=["AQI_Day1_Actual", "AQI_Day2_Actual", "AQI_Day3_Actual"])
pred_df[["AQI_Day1_Pred", "AQI_Day2_Pred", "AQI_Day3_Pred"]] = best_preds
pred_df.to_csv("model_outputs/predictions.csv", index=False)

print(f"✅ Best Model: {best_name}")
print(f"📊 Metrics: {results[best_name]}")
print("✅ Files saved to model_outputs/")

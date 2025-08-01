import os
import hopsworks
import pandas as pd
import numpy as np
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

# 🔑 Hopsworks credentials
api_key = os.environ["HOPSWORKS_API_KEY"]
project_name = os.environ["HOPSWORKS_PROJECT"]
host = os.environ["HOPSWORKS_HOST"]

# ✅ Connect to Hopsworks
project = hopsworks.login(api_key_value=api_key, project=project_name, host=host)
fs = project.get_feature_store()
fg = fs.get_feature_group("aqi_features", version=1)

# ✅ Load data from Feature Store
df = fg.read()
df.dropna(inplace=True)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# ✅ Features (removed pm2_5 & pm10) & Targets
features = ["temperature", "humidity", "wind_speed", "co", "no2"]
df["aqi_day1"] = df["aqi"].shift(-1)
df["aqi_day2"] = df["aqi"].shift(-2)
df["aqi_day3"] = df["aqi"].shift(-3)
df.dropna(inplace=True)

X = df[features]
y = df[["aqi_day1", "aqi_day2", "aqi_day3"]]

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Models
models = {
    "LinearRegression": MultiOutputRegressor(LinearRegression()),
    "RandomForest": MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42)),
    "GradientBoosting": MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, random_state=42))
}

results = {}
best_model = None
best_score = -float("inf")

# ✅ Train & Evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds, multioutput="variance_weighted")
    corr_coef = np.corrcoef(y_test.values.flatten(), preds.flatten())[0, 1]
    pearson_corr, _ = pearsonr(y_test.values.flatten(), preds.flatten())

    results[name] = {"mse": mse, "r2": r2, "corr_coef": corr_coef, "pearson": pearson_corr}

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

pred_df = pd.DataFrame(best_preds, columns=["aqi_day1", "aqi_day2", "aqi_day3"])
pred_df["actual_day1"] = y_test.values[:, 0]
pred_df.to_csv("model_outputs/predictions.csv", index=False)

print(f"✅ Best Model: {best_name}")
print(f"📊 Metrics: {results[best_name]}")
print("✅ Files saved in 'model_outputs/'")

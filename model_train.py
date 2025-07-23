import os
import hopsworks
import pandas as pd
import numpy as np
import json
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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

# ✅ Drop nulls and convert timestamp
df.dropna(inplace=True)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# ✅ Features & target
features = ["temperature", "humidity", "wind_speed", "weather_main", "pm2_5", "pm10", "co", "no2"]
X = df[features]
y = df["aqi"]

# ✅ One-hot encode weather_main
X = pd.get_dummies(X, columns=["weather_main"], drop_first=True)

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Models to train
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}
best_model = None
best_score = -float("inf")

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    corr_coef = np.corrcoef(y_test, preds)[0, 1]
    pearson_corr, _ = pearsonr(y_test, preds)

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

# ✅ Create model_outputs/ folder if not exists
os.makedirs("model_outputs", exist_ok=True)

# ✅ Save best model
joblib.dump(best_model, f"model_outputs/best_model.pkl")

# ✅ Save metrics
with open("model_outputs/model_metrics.json", "w") as f:
    json.dump(results[best_name], f, indent=4)

# ✅ Save predictions
pred_df = pd.DataFrame({"actual": y_test.values, "predicted": best_preds})
pred_df.to_csv("model_outputs/predictions.csv", index=False)

print(f"✅ Best Model: {best_name}")
print(f"📊 Metrics: {results[best_name]}")
print("✅ Files saved to model_outputs/")

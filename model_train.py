import os
import hopsworks
import pandas as pd
import numpy as np
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

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

# ✅ Add lag features (previous AQI values)
df["aqi_lag1"] = df["aqi"].shift(1)
df["aqi_lag2"] = df["aqi"].shift(2)
df["aqi_lag3"] = df["aqi"].shift(3)

# ✅ Create target columns for next 3 days
df["aqi_day1"] = df["aqi"].shift(-1)
df["aqi_day2"] = df["aqi"].shift(-2)
df["aqi_day3"] = df["aqi"].shift(-3)
df.dropna(inplace=True)

# ✅ Features & Target
features = ["temperature", "humidity", "wind_speed", "pm2_5", "pm10", "co", "no2",
            "aqi_lag1", "aqi_lag2", "aqi_lag3"]
X = df[features]
y = df[["aqi_day1", "aqi_day2", "aqi_day3"]]

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ XGBoost Multi-output model
model = MultiOutputRegressor(XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42))
model.fit(X_train, y_train)
preds = model.predict(X_test)

# ✅ Metrics
mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds, multioutput="variance_weighted")
corr_coef = np.corrcoef(y_test.values.flatten(), preds.flatten())[0, 1]
pearson_corr, _ = pearsonr(y_test.values.flatten(), preds.flatten())

metrics = {"mse": mse, "r2": r2, "corr_coef": corr_coef, "pearson": pearson_corr}

# ✅ Save outputs
os.makedirs("model_outputs", exist_ok=True)
joblib.dump(model, "model_outputs/best_model.pkl")

with open("model_outputs/model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

pred_df = pd.DataFrame(preds, columns=["aqi_day1", "aqi_day2", "aqi_day3"])
pred_df["actual_day1"] = y_test.values[:, 0]
pred_df.to_csv("model_outputs/predictions.csv", index=False)

print(f"✅ Best Model: XGBoost")
print(f"📊 Metrics: {metrics}")
print("✅ Files saved in 'model_outputs/'")

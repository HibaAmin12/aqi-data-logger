import os
import json
import pandas as pd
import numpy as np
import hopsworks
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
import joblib

# 🔐 Hopsworks credentials
api_key = os.environ["HOPSWORKS_API_KEY"]
project_name = os.environ["HOPSWORKS_PROJECT"]
host = os.environ["HOPSWORKS_HOST"]

# ✅ Login to Hopsworks
project = hopsworks.login(api_key_value=api_key, project=project_name, host=host)
fs = project.get_feature_store()

# ✅ Load all data from feature store
fg = fs.get_feature_group("aqi_features", version=1)
df = fg.read()
print(f"✅ Loaded {len(df)} rows from feature store")

# ✅ Preprocessing
df = df.drop(columns=["id", "timestamp"])
df = df.dropna()

X = df.drop("aqi", axis=1)
y = df["aqi"]

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Models to train
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42)
}

results = {}
best_model = None
best_score = -np.inf

# ✅ Create output directory
os.makedirs("model_outputs", exist_ok=True)

for name, model in models.items():
    print(f"🔁 Training: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    pearson_corr, _ = pearsonr(y_test, y_pred)
    spearman_corr, _ = spearmanr(y_test, y_pred)

    results[name] = {
        "mse": round(mse, 3),
        "r2_score": round(r2, 3),
        "pearson_corr": round(pearson_corr, 3),
        "spearman_corr": round(spearman_corr, 3)
    }

    # Select best model based on r2_score
    if r2 > best_score:
        best_score = r2
        best_model = model
        best_model_name = name
        best_predictions = pd.DataFrame({
            "Actual": y_test,
            "Predicted": y_pred
        })

# ✅ Save best model
joblib.dump(best_model, "model_outputs/best_model.pkl")
print(f"✅ Best model saved: {best_model_name}")

# ✅ Save predictions
best_predictions.to_csv("model_outputs/predictions.csv", index=False)

# ✅ Save metrics
with open("model_outputs/model_metrics.json", "w") as f:
    json.dump({
        "best_model": best_model_name,
        "all_metrics": results
    }, f, indent=4)

print("✅ Model training complete. Metrics and model saved.")

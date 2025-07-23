import os
import hopsworks
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import joblib
import json

# 🔐 Load creds
project = hopsworks.login(
    api_key_value=os.environ["HOPSWORKS_API_KEY"],
    project=os.environ["HOPSWORKS_PROJECT"],
    host=os.environ["HOPSWORKS_HOST"]
)

fs = project.get_feature_store()
fg = fs.get_feature_group("aqi_features", version=1)
df = fg.read()

# 🎯 Features and target
df = df.dropna()
X = df.drop(columns=["id", "timestamp", "aqi"])
y = df["aqi"]

# 📊 Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🧠 Models
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42)
}

results = []

# 🧪 Train + Evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    pearson_corr, _ = pearsonr(y_test, preds)

    results.append({
        "model": name,
        "object": model,
        "mse": mse,
        "r2": r2,
        "pearson_corr": pearson_corr
    })

# ✅ Best model (lowest MSE)
best_model = sorted(results, key=lambda x: x["mse"])[0]

# 📂 Save outputs
os.makedirs("model_outputs", exist_ok=True)
joblib.dump(best_model["object"], "model_outputs/best_model.pkl")

with open("model_outputs/model_metrics.json", "w") as f:
    json.dump({
        "model": best_model["model"],
        "mse": best_model["mse"],
        "r2": best_model["r2"],
        "pearson_corr": best_model["pearson_corr"]
    }, f)

pd.DataFrame({
    "Actual": y_test,
    "Predicted": best_model["object"].predict(X_test)
}).to_csv("model_outputs/predictions.csv", index=False)

print(f"✅ Best model: {best_model['model']} with MSE: {best_model['mse']:.2f}")

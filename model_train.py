import os
import hopsworks
import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

# ✅ Login to Hopsworks
api_key = os.environ["HOPSWORKS_API_KEY"]
project_name = os.environ["HOPSWORKS_PROJECT"]
host = os.environ["HOPSWORKS_HOST"]

project = hopsworks.login(api_key_value=api_key, project=project_name, host=host)
fs = project.get_feature_store()

# ✅ Load dataset
fg = fs.get_feature_group("aqi_features", version=1)
df = fg.read()

# ✅ Preprocessing
df = df.drop(columns=["id", "timestamp"])
df = df.drop_duplicates()

# ✅ Encode 'weather_main' column
df = pd.get_dummies(df, columns=["weather_main"], drop_first=True)

# ✅ Split features and target
X = df.drop(columns=["aqi"])
y = df["aqi"]

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Define models
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Ridge": Ridge(alpha=1.0)
}

results = {}

# ✅ Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    corr = np.corrcoef(y_test, y_pred)[0, 1]
    pearson_corr, _ = pearsonr(y_test, y_pred)

    results[name] = {
        "model": model,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "corr": corr,
        "pearson": pearson_corr
    }

# ✅ Best model based on R² score
best_model_name = max(results, key=lambda x: results[x]["r2"])
best_model = results[best_model_name]["model"]
metrics = results[best_model_name]

print(f"\n✅ Best model: {best_model_name}")
print("📊 Evaluation metrics:")
for key, value in metrics.items():
    if key != "model":
        print(f"  {key}: {value:.4f}")

# ✅ Save best model
os.makedirs("model_outputs", exist_ok=True)
joblib.dump(best_model, "model_outputs/best_model.pkl")
print("\n✅ Saved best model to model_outputs/best_model.pkl")

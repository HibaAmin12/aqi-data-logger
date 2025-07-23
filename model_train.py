import os
import hopsworks
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

# 🔐 Load credentials
api_key       = os.environ["HOPSWORKS_API_KEY"]
project_name  = os.environ["HOPSWORKS_PROJECT"]
host          = os.environ["HOPSWORKS_HOST"]

# ✅ Login to Hopsworks
project = hopsworks.login(api_key_value=api_key, project=project_name, host=host)
fs = project.get_feature_store()
fg = fs.get_feature_group("aqi_features", version=1)

# ✅ Load data from feature store
df = fg.read()

# ✅ Drop unused columns
df = df.drop(columns=["id", "timestamp"])
df = df.drop_duplicates()

# ✅ Encode categorical feature
df = pd.get_dummies(df, columns=["weather_main"], drop_first=True)

# ✅ Drop rows with missing target
df = df[df["aqi"].notna()]

# ✅ Split into features and target
X = df.drop("aqi", axis=1)
y = df["aqi"]

# ✅ Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Define models
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(),
    "DecisionTree": DecisionTreeRegressor(),
    "RandomForest": RandomForestRegressor()
}

# ✅ Store results
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    corr, _ = pearsonr(y_test, y_pred)

    results.append({
        "model": name,
        "mse": mse,
        "r2": r2,
        "pearson_corr": corr,
        "object": model
    })

# ✅ Find best model (lowest MSE)
best_model = min(results, key=lambda x: x["mse"])

# ✅ Save model
os.makedirs("models", exist_ok=True)
joblib.dump(best_model["object"], "models/best_model.pkl")

# ✅ Save metrics
pd.DataFrame(results).to_csv("models/metrics.csv", index=False)

print(f"✅ Best model: {best_model['model']}")
print(f"📉 MSE: {best_model['mse']:.2f}, R²: {best_model['r2']:.2f}, Pearson Corr: {best_model['pearson_corr']:.2f}")
print("✅ Model and metrics saved in models/ folder.")

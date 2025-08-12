import os
import hopsworks
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Hopsworks Login
api_key = os.getenv("HOPSWORKS_API_KEY")
project_name = os.getenv("HOPSWORKS_PROJECT")
host = os.getenv("HOPSWORKS_HOST")

if not api_key or not project_name or not host:
    raise ValueError("Hopsworks credentials not found in environment variables.")

project = hopsworks.login(api_key_value=api_key, project=project_name, host=host)
fs = project.get_feature_store()

# Load Data
fg = fs.get_feature_group("aqi_features", version=1)
df = fg.read().sort_values("timestamp")
df = df.dropna(subset=["aqi"])

# Lag Features
for col in ["aqi", "pm2_5", "pm10", "co", "no2"]:
    df[f"{col}_lag1"] = df[col].shift(1)

df = df.dropna().reset_index(drop=True)

# Outlier Capping
def cap_outliers(data, cols):
    for col in cols:
        Q1, Q3 = data[col].quantile(0.25), data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        data[col] = np.clip(data[col], lower, upper)
    return data

df = cap_outliers(df, ["pm2_5", "pm10", "wind_speed"])

# Scaling
numeric_features = ["temperature", "humidity", "wind_speed", "pm2_5", "pm10", "co", "no2"]
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Features & Target
features = numeric_features + ["aqi_lag1", "pm2_5_lag1", "pm10_lag1", "co_lag1", "no2_lag1"]
X, y = df[features], df["aqi"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.01),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        objective="reg:squarederror", random_state=42,
        use_label_encoder=False, eval_metric='rmse'
    )
}

results = {}
best_model_name, best_model, best_r2 = None, None, -1

for name, model in models.items():
    model.fit(X_train, y_train)
    y_train_pred, y_test_pred = model.predict(X_train), model.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    results[name] = {
        "Train R²": train_r2,
        "Test R²": test_r2,
    }

    if test_r2 > best_r2:
        best_r2, best_model_name, best_model = test_r2, name, model

# Save Results
results_df = pd.DataFrame(results).T
os.makedirs("model_outputs", exist_ok=True)
results_df.to_csv("model_outputs/model_results.csv")

# Refit best model on full data and save
best_model.fit(X, y)
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/aqi_best_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

# Save latest row and full dataset
df.iloc[[-1]].to_csv("latest_pollutants.csv", index=False)
df.to_csv("processed_data.csv", index=False)

print(f"Model training complete using {best_model_name}. Model and scaler saved.")
print("Latest and full processed dataset saved.")

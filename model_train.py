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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
    "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
}

results = {}
best_model_name = None
best_model = None
best_r2 = -1

for name, model in models.items():
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Train metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    # Test metrics
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    results[name] = {
        "Train MSE": train_mse,
        "Train RMSE": train_rmse,
        "Train MAE": train_mae,
        "Train R²": train_r2,
        "Test MSE": test_mse,
        "Test RMSE": test_rmse,
        "Test MAE": test_mae,
        "Test R²": test_r2
    }

    # Track best model
    if test_r2 > best_r2:
        best_r2 = test_r2
        best_model_name = name
        best_model = model

# --------------------
# Check Overfitting for Best Model
# --------------------
train_r2_best = results[best_model_name]["Train R²"]
test_r2_best = results[best_model_name]["Test R²"]

if train_r2_best - test_r2_best > 0.05:  # significant overfitting
    print(f"\n Overfitting detected in {best_model_name}. Retraining with tuned parameters...")
    if best_model_name == "XGBoost":
        best_model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        best_model.fit(X_train, y_train,
                       eval_set=[(X_test, y_test)],
                       verbose=False,
                       early_stopping_rounds=20)
    elif best_model_name == "Random Forest":
        best_model = RandomForestRegressor(
            n_estimators=80,
            max_depth=10,
            min_samples_leaf=4,
            random_state=42
        )
        best_model.fit(X_train, y_train)

# --------------------
# Save Results
# --------------------
results_df = pd.DataFrame(results).T
print("\n📊 Model Performance (Train & Test):\n")
print(results_df)

os.makedirs("model_outputs", exist_ok=True)
results_df.to_csv("model_outputs/model_results.csv")

# --------------------
# Save Best Model
# --------------------
best_model.fit(X, y)
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/aqi_best_model.pkl")
print(f"\n✅ Best model saved: {best_model_name} (R²: {best_r2:.4f})")

# Save latest row for Streamlit
latest = df.iloc[-1]
latest.to_frame().T.to_csv("latest_pollutants.csv", index=False)
print(" Latest pollutants saved to 'latest_pollutants.csv'")

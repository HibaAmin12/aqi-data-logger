import hopsworks
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# 🔐 Load credentials from GitHub Secrets
api_key = os.environ["HOPSWORKS_API_KEY"]
project = os.environ["HOPSWORKS_PROJECT"]
host = os.environ["HOPSWORKS_HOST"]

# ✅ Connect to Hopsworks
project = hopsworks.login(
    api_key_value=api_key,
    project=project,
    host=host
)
fs = project.get_feature_store()

# ✅ Read data from feature store
feature_group = fs.get_feature_group(name="aqi_features", version=1)
df = feature_group.read()

print("✅ Loaded data from feature store:", df.shape)

# ✅ Select numeric features + target
features = ["temperature", "humidity", "wind_speed", "pm2_5", "pm10", "co", "no2"]
target = "aqi"

X = df[features]
y = df[target]

# ✅ Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Define models
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(),
    "DecisionTree": DecisionTreeRegressor(),
    "RandomForest": RandomForestRegressor()
}

# ✅ Train and evaluate
for name, model in models.items():
    print(f"\n📦 Training model: {name}")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print(f"📊 R² Score: {r2:.3f}")
    print(f"📉 MAE: {mae:.3f}")
    print(f"📈 RMSE: {rmse:.3f}")

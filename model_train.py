import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# --------------------
# Load Standardized Data from EDA
# --------------------
df_scaled = pd.read_csv("eda_outputs/standardized_data.csv")

# Features and Target
X = df_scaled[["temperature", "humidity", "wind_speed", "pm2_5", "pm10", "co", "no2"]]
y = df_scaled["aqi"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------
# Define Models
# --------------------
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.01),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

results = {}
predictions = {}

# --------------------
# Train and Evaluate
# --------------------
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    corr = np.corrcoef(y_test, y_pred)[0, 1]

    results[name] = {"MSE": mse, "R²": r2, "Correlation": corr}

# Convert results to DataFrame
results_df = pd.DataFrame(results).T
print("\n📊 Model Performance:\n")
print(results_df)

# --------------------
# Plot R² Comparison
# --------------------
plt.figure(figsize=(8, 5))
plt.bar(results_df.index, results_df["R²"], color='skyblue')
plt.title("R² Score Comparison of Models")
plt.ylabel("R² Score")
plt.xticks(rotation=30)
plt.ylim(0, 1.05)
plt.show()

# --------------------
# Plot MSE Comparison
# --------------------
plt.figure(figsize=(8, 5))
plt.bar(results_df.index, results_df["MSE"], color='salmon')
plt.title("MSE Comparison of Models")
plt.ylabel("Mean Squared Error")
plt.xticks(rotation=30)
plt.show()

# --------------------
# Save Best Model (Random Forest)
# --------------------
best_model = RandomForestRegressor(n_estimators=100, random_state=42)
best_model.fit(X, y)  # Train on full dataset
joblib.dump(best_model, "models/aqi_best_model.pkl")
print("\n✅ Best model (Random Forest) saved as models/aqi_best_model.pkl")

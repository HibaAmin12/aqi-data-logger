#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Load latest dataset
df = pd.read_csv("api.csv")  # replace with your file path

# Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True)
df = df.sort_values("timestamp")

# Remove AQI category values (1–5)
df["aqi"] = pd.to_numeric(df["aqi"], errors='coerce')
df = df[df["aqi"] > 10]  # Keep only real values
df = df.dropna()

# Encode categorical weather
df = pd.get_dummies(df, columns=["weather_main"], drop_first=True)

# Time-based features
df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.day
df["month"] = df["timestamp"].dt.month

# AQI change rate
df["aqi_change"] = df["aqi"].diff().fillna(0)

# Save as features.csv
df.to_csv("features.csv", index=False)

df.head()


# In[2]:


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import numpy as np

df = pd.read_csv("features.csv")

# Define features/labels
X = df.drop(columns=["timestamp", "aqi"])
y = df["aqi"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "aqi_model.pkl")

# Evaluate
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))


# In[ ]:


# ✅ Convert this notebook to .py script automatically
get_ipython().system('jupyter nbconvert --to script Un.ipynb')
get_ipython().system('jupyter nbconvert --to script 02_model_training.ipynb')


# In[ ]:





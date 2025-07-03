#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Run this cell once
get_ipython().system('pip install pandas numpy matplotlib seaborn scikit-learn joblib')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


# In[11]:


# Replace with your actual filename
df = pd.read_csv("api.csv")  

# Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True)

# Sort by time
df = df.sort_values("timestamp")

# Convert AQI to numeric and keep only real values (>10)
df["aqi"] = pd.to_numeric(df["aqi"], errors='coerce')
df = df[df["aqi"] > 10]
df = df.dropna()

# One-hot encode weather condition
df = pd.get_dummies(df, columns=["weather_main"], drop_first=True)

# Add time-based features
df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.day
df["month"] = df["timestamp"].dt.month

df.head()


# In[12]:


# AQI Trend Over Time
plt.figure(figsize=(12, 6))
plt.plot(df["timestamp"], df["aqi"], marker='o')
plt.title("AQI Over Time")
plt.xlabel("Date")
plt.ylabel("AQI")
plt.grid(True)
plt.show()

# Correlation Matrix
plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation")
plt.show()

# AQI Distribution
sns.histplot(df["aqi"], bins=30, kde=True)
plt.title("AQI Distribution")
plt.xlabel("AQI")
plt.ylabel("Frequency")
plt.show()


# In[13]:


X = df.drop(columns=["timestamp", "aqi"])
y = df["aqi"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[14]:


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "aqi_model.pkl")


# In[15]:


y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))

# Plot Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.plot(list(y_test.values), label="Actual", marker='o')
plt.plot(y_pred, label="Predicted", marker='x')
plt.title("Actual vs Predicted AQI")
plt.xlabel("Index")
plt.ylabel("AQI")
plt.legend()
plt.grid(True)
plt.show()


# In[16]:


# Predict using last row (latest data)
latest = df.tail(1).drop(columns=["timestamp", "aqi"])
model = joblib.load("aqi_model.pkl")
predicted_aqi = model.predict(latest)
print("Predicted AQI for Latest Entry:", predicted_aqi[0])


# In[ ]:





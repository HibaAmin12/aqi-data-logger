import hopsworks
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load secrets
api_key = os.environ["HOPSWORKS_API_KEY"]
project = os.environ["HOPSWORKS_PROJECT"]
host = os.environ["HOPSWORKS_HOST"]

# Connect to Hopsworks
project = hopsworks.login(
    api_key_value=api_key,
    project=project,
    host=host
)
fs = project.get_feature_store()

# Load feature group data
feature_group = fs.get_feature_group(name="aqi_features", version=1)
df = feature_group.read()

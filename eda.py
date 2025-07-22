# eda.py

import os
import hopsworks
import pandas as pd
import sweetviz as sv

# 🔐 Load secrets from environment
api_key = os.environ["HOPSWORKS_API_KEY"]
project_name = os.environ["HOPSWORKS_PROJECT"]
host = os.environ["HOPSWORKS_HOST"]

# ✅ Connect to Hopsworks
project = hopsworks.login(api_key_value=api_key, project=project_name, host=host)
fs = project.get_feature_store()
fg = fs.get_feature_group(name="aqi_features", version=1)

# ✅ Load data from feature store
df = fg.read()

# ✅ Save preview CSV for debugging (optional)
df.to_csv("eda_outputs/aqi_data_latest.csv", index=False)

# ✅ Create Sweetviz EDA report
report = sv.analyze(df)
report_path = "eda_report.html"
report.show_html(report_path)

print("✅ EDA report generated and saved.")

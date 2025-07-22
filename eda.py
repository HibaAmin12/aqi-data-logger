import os
import hopsworks
import pandas as pd
import sweetviz

# 🔐 Load secrets
api_key = os.environ["HOPSWORKS_API_KEY"]
project_name = os.environ["HOPSWORKS_PROJECT"]
host = os.environ["HOPSWORKS_HOST"]

# ✅ Login to Hopsworks
project = hopsworks.login(api_key_value=api_key, project=project_name, host=host)
fs = project.get_feature_store()

# ✅ Get feature group
fg = fs.get_feature_group(name="aqi_features", version=1)

# ✅ Read data
df = fg.read()
df = df.sort_values("timestamp", ascending=False).head(50)  # Just latest 50 rows if needed

# ✅ Generate EDA report
report = sweetviz.analyze(df)
report_path = "eda_report.html"
report.show_html(filepath=report_path, open_browser=False)
print(f"✅ EDA report saved to {report_path}")

# eda_report.py

import pandas as pd
from ydata_profiling import ProfileReport

# Read the AQI dataset
df = pd.read_csv("api.csv")

# Create the profile report
profile = ProfileReport(df, title="AQI EDA Report", explorative=True)

# Save as HTML
profile.to_file("eda_report.html")
print("✅ EDA report generated: eda_report.html")

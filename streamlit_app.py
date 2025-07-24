import streamlit as st
import pandas as pd
import joblib

# Load model and data
model = joblib.load("best_model.pkl")
data = pd.read_csv("api.csv")

# Preprocess features (drop target + timestamp + id)
X = data.drop(columns=["aqi", "timestamp", "id"], errors="ignore")

# UI
st.title("🌫️ AQI Prediction App")
st.write("This app uses a machine learning model to predict the Air Quality Index (AQI) based on weather and pollution data.")

# Show input data
with st.expander("📊 Show Input Data"):
    st.dataframe(X)

# Predict
if st.button("🔮 Predict AQI"):
    predictions = model.predict(X)
    data["Predicted AQI"] = predictions
    st.success("AQI prediction completed!")
    st.dataframe(data[["timestamp", "aqi", "Predicted AQI"]])

    # Save predictions
    data.to_csv("model_outputs/predictions.csv", index=False)
    st.download_button("📥 Download Predictions", data.to_csv(index=False), file_name="predictions.csv")


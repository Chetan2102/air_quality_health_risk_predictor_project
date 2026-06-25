
import streamlit as st
import pandas as pd
import pickle

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------

st.set_page_config(
    page_title="Air Quality Health Risk Predictor",
    page_icon="🌍",
    layout="wide"
)

# ----------------------------------------------------
# Custom CSS
# ----------------------------------------------------

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

h1 {
    text-align: center;
    color: #2E8B57;
}

.stButton>button {
    width: 100%;
    background-color: #2E8B57;
    color: white;
    font-size:18px;
    border-radius:8px;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Load Model Files
# ----------------------------------------------------

with open("air_quality_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

with open("feature_names.pkl", "rb") as file:
    feature_names = pickle.load(file)

# ----------------------------------------------------
# Title
# ----------------------------------------------------

st.title("🌍 Air Quality Health Risk Predictor")

st.write("""
Predict the health impact level based on air quality and
environmental conditions using a trained Support Vector Machine (SVM).
""")

# ----------------------------------------------------
# Sidebar Inputs
# ----------------------------------------------------

st.sidebar.header("Enter Air Quality Details")

AQI = st.sidebar.number_input("AQI", min_value=0.0, value=150.0)

PM10 = st.sidebar.number_input("PM10", min_value=0.0, value=100.0)

PM2_5 = st.sidebar.number_input("PM2.5", min_value=0.0, value=80.0)

NO2 = st.sidebar.number_input("NO₂", min_value=0.0, value=40.0)

SO2 = st.sidebar.number_input("SO₂", min_value=0.0, value=20.0)

O3 = st.sidebar.number_input("O₃", min_value=0.0, value=50.0)

Temperature = st.sidebar.number_input(
    "Temperature (°C)",
    value=25.0
)

Humidity = st.sidebar.slider(
    "Humidity (%)",
    0.0,
    100.0,
    50.0
)

WindSpeed = st.sidebar.number_input(
    "Wind Speed",
    min_value=0.0,
    value=5.0
)

RespiratoryCases = st.sidebar.number_input(
    "Respiratory Cases",
    min_value=0.0,
    value=10.0
)

CardiovascularCases = st.sidebar.number_input(
    "Cardiovascular Cases",
    min_value=0.0,
    value=5.0
)

HospitalAdmissions = st.sidebar.number_input(
    "Hospital Admissions",
    min_value=0.0,
    value=8.0
)

# ----------------------------------------------------
# Create Input DataFrame
# ----------------------------------------------------

input_df = pd.DataFrame(
    [[
        AQI,
        PM10,
        PM2_5,
        NO2,
        SO2,
        O3,
        Temperature,
        Humidity,
        WindSpeed,
        RespiratoryCases,
        CardiovascularCases,
        HospitalAdmissions
    ]],
    columns=feature_names
)

# ----------------------------------------------------
# Prediction
# ----------------------------------------------------

if st.button("Predict Health Impact"):

    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]

    confidence = model.predict_proba(scaled_input).max()

    class_labels = {
        0: "Very Low",
        1: "Low",
        2: "Moderate",
        3: "High",
        4: "Very High"
    }

    recommendations = {
        0: "Air quality is excellent. Outdoor activities are generally safe.",

        1: "Air quality is acceptable. Sensitive individuals should monitor symptoms.",

        2: "Limit prolonged outdoor activities if you have respiratory conditions.",

        3: "Reduce outdoor exposure and consider wearing an N95 mask.",

        4: "Avoid outdoor activities whenever possible. Stay indoors and use an air purifier if available."
    }

    # --------------------------------------------

    st.success(
        f"Predicted Health Impact: {class_labels[prediction]}"
    )

    st.subheader("Prediction Confidence")

    st.progress(float(confidence))

    st.write(f"Confidence: **{confidence:.2%}**")

    # --------------------------------------------

    st.subheader("Health Recommendation")

    st.info(recommendations[prediction])

    # --------------------------------------------

    st.subheader("Input Summary")

    st.dataframe(input_df)

# ----------------------------------------------------
# About Model
# ----------------------------------------------------

st.markdown("---")

st.subheader("About this Project")

st.write("""
**Model:** Support Vector Machine (SVM)

**Dataset:** Air Quality Health Impact Dataset

**Deployment:** Streamlit

**Model Serialization:** Pickle
""")

st.caption("Developed as an End-to-End Machine Learning Project.")


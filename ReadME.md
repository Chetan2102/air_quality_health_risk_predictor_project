# Air Quality Health Risk Predictor

## Project Overview

The **Air Quality Health Risk Predictor** is an end-to-end Machine Learning project that predicts the **Health Impact Class** based on air quality indicators, weather conditions, and health-related parameters. The project demonstrates the complete machine learning pipeline, including data preprocessing, exploratory data analysis, model training, evaluation, hyperparameter tuning, model serialization using Pickle, and deployment through Streamlit.

The final model is a **Support Vector Machine (SVM)** classifier, selected after comparing the performance of multiple machine learning algorithms.

---

## Dataset

The project uses the **Air Quality Health Impact Dataset** from Kaggle.

The dataset contains environmental and health-related features such as:

* Air Quality Index (AQI)
* PM10
* PM2.5
* Nitrogen Dioxide (NO₂)
* Sulfur Dioxide (SO₂)
* Ozone (O₃)
* Temperature
* Humidity
* Wind Speed
* Respiratory Cases
* Cardiovascular Cases
* Hospital Admissions

The target variable, **HealthImpactClass**, categorizes the overall health impact into different risk levels based on the given environmental conditions. The dataset is well-suited for multiclass classification tasks and provides a realistic scenario for developing predictive machine learning models.

# Explainable Farming AI

##  Project Overview

Explainable Farming AI is an intelligent agriculture support system that helps farmers and agricultural researchers identify the most suitable crop based on soil nutrients and environmental conditions.

Unlike traditional crop recommendation systems, this project not only predicts the crop but also explains *why* the prediction was made using Explainable AI (XAI) techniques such as SHAP (SHapley Additive Explanations).

The application is developed using Machine Learning, Streamlit, and SHAP visualization tools to create an interactive and user-friendly farming assistant.

---

#  Objectives

- Predict the best crop for cultivation
- Analyze soil and weather conditions
- Provide AI-based farming suggestions
- Improve decision transparency using Explainable AI
- Help farmers make data-driven agricultural decisions

---

#  Features

##  Crop Recommendation
Predicts the most suitable crop using machine learning algorithms trained on agricultural datasets.

## 📊 Explainable AI (XAI)
Uses SHAP visualizations to explain:
- Feature importance
- Prediction reasoning
- Impact of each soil/weather parameter

##  Smart Risk Alerts
Detects abnormal farming conditions such as:
- Low rainfall
- High temperature
- Unbalanced soil nutrients

##  Interactive Dashboard
Simple and clean Streamlit interface with:
- Sliders
- Input boxes
- Graphs
- Prediction results

##  Data Visualization
Displays:
- SHAP waterfall plots
- Feature contribution charts
- Prediction confidence

---

#  Machine Learning Workflow

``text
User Input
   ↓
Data Preprocessing
   ↓
Machine Learning Model
   ↓
Crop Prediction
   ↓
SHAP Explainability
   ↓
Final Recommendation

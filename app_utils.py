import json
import os
import pickle
import time
import urllib.parse
import urllib.request
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")


def apply_theme():
    st.markdown(
        """
        <style>
            .main-title {
                color: #2E7D32;
                text-align: center;
                font-size: 3em;
                margin-bottom: 10px;
            }
            .subtitle {
                color: #558B2F;
                text-align: center;
                font-size: 1.2em;
                margin-bottom: 30px;
            }
            .prediction-box {
                background-color: #C8E6C9;
                padding: 20px;
                border-radius: 10px;
                border-left: 5px solid #2E7D32;
                margin: 10px 0;
            }
            .confidence-box {
                background-color: #FFF9C4;
                padding: 15px;
                border-radius: 10px;
                border-left: 5px solid #F57F17;
                margin: 10px 0;
            }
            .warning-box {
                background-color: #FFCDD2;
                padding: 15px;
                border-radius: 10px;
                border-left: 5px solid #D32F2F;
                margin: 10px 0;
            }
            .insight-box {
                background-color: #E3F2FD;
                padding: 15px;
                border-radius: 10px;
                border-left: 5px solid #1976D2;
                margin: 10px 0;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _find_dataset_path():
    candidate_paths = ["data/Crop.csv", "data/crop_recommendation.csv"]
    for candidate in candidate_paths:
        if os.path.exists(candidate):
            return candidate
    return None


def _load_dataset(path):
    df = pd.read_csv(path)
    if "ph" in df.columns and "pH" not in df.columns:
        df = df.rename(columns={"ph": "pH"})
    if "PH" in df.columns and "pH" not in df.columns:
        df = df.rename(columns={"PH": "pH"})

    expected_columns = {"N", "P", "K", "pH", "temperature", "humidity", "rainfall", "label"}
    missing = expected_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing expected columns: {sorted(missing)}. "
            "Please provide a dataset with N, P, K, pH, temperature, humidity, rainfall, and label."
        )

    return df


def load_crop_info():
    crop_info_path = "data/crop_info.json"
    if not os.path.exists(crop_info_path):
        return {}
    try:
        with open(crop_info_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def get_crop_harvest_info(crop_name, crop_info):
    if not crop_info or not crop_name:
        return None
    crop_key = str(crop_name).lower()
    details = crop_info.get(crop_key)
    if not details:
        return None
    details = details.copy()
    average_days = details.get("average_harvest_days")
    if isinstance(average_days, (int, float)):
        details["estimated_harvest_date"] = (
            pd.Timestamp.now().normalize() + pd.Timedelta(days=int(average_days))
        ).strftime("%d %B %Y")
    return details


def _safe_api_request(url, headers=None):
    try:
        request = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(request, timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None


WEATHER_CODE_MAP = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    95: "Thunderstorm",
}


CITY_COORDINATES = {
    "pollachi": (10.65, 77.00),
    "chennai": (13.08, 80.27),
    "coimbatore": (11.01, 76.96),
    "madurai": (9.93, 78.12),
    "bangalore": (12.97, 77.59),
    "mumbai": (19.08, 72.88),
}


@st.cache_data(show_spinner=False)
def get_weather_for_location(location):
    if not location or not str(location).strip():
        raise ValueError("Please enter a valid location to fetch weather.")

    raw_location = str(location).strip()
    normalized = " ".join(raw_location.split()).lower()
    if normalized in {"", "none", "null"}:
        raise ValueError("Please enter a valid location to fetch weather.")

    latitude = None
    longitude = None
    display_name = raw_location

    if "," in raw_location:
        parts = [p.strip() for p in raw_location.split(",")]
        if len(parts) == 2:
            try:
                latitude = float(parts[0])
                longitude = float(parts[1])
                display_name = f"{latitude}, {longitude}"
            except ValueError:
                latitude = None
                longitude = None

    if latitude is None or longitude is None:
        if normalized in CITY_COORDINATES:
            latitude, longitude = CITY_COORDINATES[normalized]
            display_name = raw_location.title()
        else:
            encoded = urllib.parse.quote(raw_location)
            geocoding_url = (
                f"https://geocoding-api.open-meteo.com/v1/search?name={encoded}&count=5&language=en&format=json"
            )
            geocoding_data = _safe_api_request(geocoding_url)
            if not geocoding_data:
                nominatim_url = f"https://nominatim.openstreetmap.org/search?q={encoded}&format=json&limit=5"
                headers = {"User-Agent": "ExplainableFarmingAI/1.0"}
                geocoding_data = _safe_api_request(nominatim_url, headers=headers)

            if not geocoding_data:
                raise ConnectionError("Weather service is currently unavailable. Please try again later.")

            matched = None
            if isinstance(geocoding_data, dict) and "results" in geocoding_data:
                candidates = geocoding_data.get("results", [])
                for item in candidates:
                    item_name = str(item.get("name", "")).lower()
                    if normalized in item_name or any(token in item_name for token in normalized.split()):
                        matched = item
                        break
                if matched is None and candidates:
                    matched = candidates[0]
            else:
                for item in geocoding_data:
                    place_name = str(item.get("display_name", "")).lower()
                    if normalized in place_name or any(token in place_name for token in normalized.split()):
                        matched = item
                        break
                if matched is None and geocoding_data:
                    matched = geocoding_data[0]

            if matched is None:
                raise ValueError(f"No location found for '{raw_location}'. Please try a different city or coordinates.")

            if isinstance(matched, dict) and "latitude" in matched and "longitude" in matched:
                latitude = float(matched["latitude"])
                longitude = float(matched["longitude"])
                display_name = matched.get("name", raw_location)
            else:
                latitude = float(matched["lat"])
                longitude = float(matched["lon"])
                display_name = matched.get("display_name", raw_location)

    weather_url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}"
        f"&current_weather=true&hourly=relativehumidity_2m&timezone=auto"
    )
    weather_data = _safe_api_request(weather_url)
    if not weather_data:
        raise ConnectionError("Weather service is currently unavailable. Please try again later.")

    current = weather_data.get("current_weather")
    if current is None:
        raise ValueError("Weather service returned an invalid response.")

    humidity = None
    if "hourly" in weather_data and "relativehumidity_2m" in weather_data["hourly"]:
        hourly = weather_data["hourly"]
        times = hourly.get("time", [])
        values = hourly.get("relativehumidity_2m", [])
        if current.get("time") in times:
            idx = times.index(current["time"])
            humidity = values[idx]
        elif values:
            humidity = values[0]

    if humidity is None:
        raise ValueError("Unable to retrieve humidity from weather response.")

    weather_code = current.get("weathercode")
    condition = WEATHER_CODE_MAP.get(weather_code, f"Weather code {weather_code}")

    return {
        "name": display_name,
        "temperature": float(current["temperature"]),
        "humidity": float(humidity),
        "condition": condition,
    }


@st.cache_data(show_spinner=False)
def compute_model_comparison(df, feature_names):
    X = df[feature_names]
    y = LabelEncoder().fit_transform(df["label"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = []
    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

        results.append(
            {
                "Model": name,
                "Accuracy": round(accuracy, 4),
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "F1 score": round(f1, 4),
                "CV Mean": round(float(cv_scores.mean()), 4),
                "CV Std": round(float(cv_scores.std()), 4),
                "Train Time (s)": round(train_time, 3),
            }
        )

    return pd.DataFrame(results)


def render_dataset_analysis_page(df, feature_names):
    st.markdown("## 📈 Dataset Analysis")
    st.info("This page is intended for developers and interview demos to review the dataset and feature distributions.")

    st.markdown("### Dataset Overview")
    st.write(f"**Shape:** {df.shape}")
    st.write(f"**Features:** {feature_names}")
    st.write("**Data types:**")
    st.dataframe(df.dtypes.astype(str).to_frame("dtype"), use_container_width=True)

    st.markdown("**Missing values:**")
    st.dataframe(df.isna().sum().to_frame("missing_count"), use_container_width=True)

    st.markdown("**Duplicate analysis:**")
    st.write(f"Duplicate rows: {int(df.duplicated().sum())}")

    st.markdown("**Statistical summary:**")
    st.dataframe(df.describe().transpose(), use_container_width=True)

    st.markdown("**Target class distribution:**")
    target_counts = df["label"].value_counts().reset_index()
    target_counts.columns = ["label", "count"]
    st.dataframe(target_counts, use_container_width=True)
    st.bar_chart(target_counts.set_index("label"))

    st.markdown("**Correlation heatmap:**")
    corr = df[feature_names].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(corr, cmap="coolwarm")
    fig.colorbar(cax)
    ax.set_xticks(range(len(feature_names)))
    ax.set_yticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha="left")
    ax.set_yticklabels(feature_names)
    st.pyplot(fig)

    st.markdown("**Histograms:**")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i, feature in enumerate(feature_names):
        df[feature].hist(ax=axes[i], bins=20, color="#2E7D32", alpha=0.7)
        axes[i].set_title(feature)
    axes[-1].axis("off")
    st.pyplot(fig)

    st.markdown("**Boxplots for outlier analysis:**")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i, feature in enumerate(feature_names):
        axes[i].boxplot(df[feature].dropna(), patch_artist=True, boxprops=dict(facecolor="#2E7D32", alpha=0.6))
        axes[i].set_title(feature)
    axes[-1].axis("off")
    st.pyplot(fig)


def render_model_performance_page(df, feature_names):
    st.markdown("## 📊 Model Performance")
    st.info("This page is dedicated to benchmarking and comparing candidate models while keeping the production prediction experience focused.")

    st.markdown("### Algorithm benchmark")
    comparison_df = compute_model_comparison(df, feature_names)
    best_model = comparison_df.sort_values("F1 score", ascending=False).iloc[0]
    st.write(f"**Best model by F1 score:** {best_model['Model']} ({best_model['F1 score']})")
    st.write("_Random Forest remains the production model for predictions and explainability._")
    styled = comparison_df.style.highlight_max(axis=0, subset=["Accuracy", "Precision", "Recall", "F1 score", "CV Mean"])
    st.dataframe(styled, use_container_width=True)

    st.markdown("### Model comparison table")
    st.table(comparison_df)


@st.cache_resource
def load_model_and_data():
    model_path = "models/crop_model.pkl"
    encoder_path = "models/label_encoder.pkl"

    dataset_path = _find_dataset_path()
    if dataset_path is None:
        raise FileNotFoundError(
            "Dataset not found at data/Crop.csv or data/crop_recommendation.csv. "
            "Please place your dataset file at data/Crop.csv."
        )

    if os.path.exists(model_path) and os.path.exists(encoder_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(encoder_path, "rb") as f:
            encoder = pickle.load(f)
        df = _load_dataset(dataset_path)
        return model, encoder, df

    df = _load_dataset(dataset_path)

    X = df.drop("label", axis=1)
    y = df["label"]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_encoded)

    os.makedirs("models", exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(encoder_path, "wb") as f:
        pickle.dump(encoder, f)

    return model, encoder, df


def predict_crop(model, encoder, features_dict):
    feature_names = ["N", "P", "K", "pH", "temperature", "humidity", "rainfall"]
    input_df = pd.DataFrame([features_dict], columns=feature_names)
    prediction = model.predict(input_df)[0]
    confidence = model.predict_proba(input_df)[0].max() * 100
    crop_name = encoder.inverse_transform([prediction])[0]
    return crop_name, round(float(confidence), 1)


@st.cache_data(show_spinner=False)
def _get_shap_explainer(_model):
    explainer = shap.TreeExplainer(_model)
    return explainer


def explain_prediction(model, encoder, features_dict, feature_names):
    feature_df = pd.DataFrame([features_dict], columns=feature_names)
    explainer = _get_shap_explainer(model)
    shap_values = explainer.shap_values(feature_df)
    base_value = explainer.expected_value
    return shap_values, base_value, feature_df


def get_feature_importance(model, feature_names):
    importances = model.feature_importances_
    return {"features": feature_names, "importances": importances}


def assess_conditions_risk(features_dict, df):
    warnings_list = []
    N = features_dict["N"]
    P = features_dict["P"]
    K = features_dict["K"]
    pH = features_dict["pH"]
    temp = features_dict["temperature"]
    humidity = features_dict["humidity"]
    rainfall = features_dict["rainfall"]

    if N > 100:
        warnings_list.append("⚠️ Nitrogen level is high - may favor leafy growth crops")
    elif N < 30:
        warnings_list.append("⚠️ Nitrogen level is low - legumes may be a better fit")

    if P < 20:
        warnings_list.append("⚠️ Phosphorus is low - root and flowering crops may underperform")

    if K < 30:
        warnings_list.append("⚠️ Potassium is low - fruit quality may be affected")

    if pH > 7.5:
        warnings_list.append("⚠️ Soil is alkaline - consider crops that tolerate high pH")
    elif pH < 5.5:
        warnings_list.append("⚠️ Soil is acidic - monitor crop suitability carefully")

    if temp > 35:
        warnings_list.append("⚠️ Temperature is very high - heat stress may affect growth")
    elif temp < 15:
        warnings_list.append("⚠️ Temperature is low - cool-season crops may be more suitable")

    if humidity > 85:
        warnings_list.append("⚠️ Humidity is very high - fungal disease risk increases")
    elif humidity < 40:
        warnings_list.append("⚠️ Humidity is low - irrigation may be needed")

    if rainfall > 250:
        warnings_list.append("⚠️ Rainfall is very high - drainage and waterlogging may be concern")
    elif rainfall < 60:
        warnings_list.append("⚠️ Low rainfall - irrigation recommended")

    return warnings_list


def generate_insights(crop_name, features_dict, encoder):
    insights = []

    N = features_dict["N"]
    P = features_dict["P"]
    K = features_dict["K"]
    pH = features_dict["pH"]
    temp = features_dict["temperature"]
    humidity = features_dict["humidity"]
    rainfall = features_dict["rainfall"]

    if N > 100:
        insights.append("💪 High nitrogen promotes leafy growth (ideal for rice and sugarcane)")
    elif N < 30:
        insights.append("🌱 Low nitrogen - suitable for legumes (beans, peas)")

    if K > 100:
        insights.append("🍎 High potassium improves fruit quality and disease resistance")

    if pH > 7.5:
        insights.append("⚡ Alkaline soil suits chickpeas and certain vegetables")
    elif pH < 5.5:
        insights.append("🌿 Acidic soil suits rice and tea crops")
    else:
        insights.append("✅ pH is in neutral range - suitable for most crops")

    if temp > 25 and humidity > 70:
        insights.append("☀️ Warm and humid conditions favor tropical crops")
    elif temp < 20:
        insights.append("❄️ Cool climate suits temperate crops like apple and lentil")

    if rainfall > 150:
        insights.append("💧 High rainfall supports moisture-loving crops")
    else:
        insights.append("🏜️ Low rainfall requires drought-resistant varieties")

    return insights


def build_prediction_report(predicted_crop, confidence, features_dict, weather_name, weather_condition, shap_values_class, base_val, feature_names, explanation_text):
    lines = [
        "Explainable Farming AI - Crop Prediction Report",
        "=" * 42,
        f"Predicted Crop: {predicted_crop}",
        f"Confidence: {confidence:.1f}%",
        f"Selected Location: {weather_name}",
        f"Weather Condition: {weather_condition}",
        "",
        "Input Features:",
    ]
    for feature in feature_names:
        lines.append(f"- {feature}: {features_dict[feature]}")
    lines.extend(["", "SHAP Impact Summary:", explanation_text])
    return "\n".join(lines)

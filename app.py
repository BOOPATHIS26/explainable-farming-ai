import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from app_utils import (
    apply_theme,
    assess_conditions_risk,
    build_prediction_report,
    explain_prediction,
    generate_insights,
    get_crop_harvest_info,
    get_feature_importance,
    get_weather_for_location,
    load_crop_info,
    load_model_and_data,
    predict_crop,
)

apply_theme()

st.set_page_config(page_title="Explainable Farming AI", page_icon="🌾", layout="wide")

st.markdown("<h1 class='main-title'>🌾 Explainable Farming AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Smart Crop Recommendation with AI Explainability</p>", unsafe_allow_html=True)

try:
    model, encoder, df = load_model_and_data()
except FileNotFoundError as e:
    st.error(f"❌ {str(e)}")
    st.info("Please add data/Crop.csv and rerun the app.")
    st.stop()

feature_names = ["N", "P", "K", "pH", "temperature", "humidity", "rainfall"]

st.sidebar.header("🌍 Input Soil & Weather Conditions")
st.sidebar.info("Adjust the values using sliders or enter manually")

location = st.sidebar.text_input(
    "Location (city or lat,lon)",
    value=st.session_state.get("weather_location", ""),
)
if st.sidebar.button("Fetch Weather"):
    if not location or not str(location).strip():
        st.warning("Please enter a location before fetching weather.")
    else:
        try:
            weather = get_weather_for_location(location)
            st.session_state["weather_location"] = location
            st.session_state["weather_temperature"] = weather["temperature"]
            st.session_state["weather_humidity"] = weather["humidity"]
            st.session_state["weather_condition"] = weather.get("condition", "Unknown")
            st.session_state["weather_name"] = weather.get("name", location)
            st.sidebar.success(
                f"Fetched: {weather.get('name', location)} | {weather['temperature']}°C | {weather['humidity']}% humidity | {weather.get('condition', 'Unknown')}"
            )
        except ValueError as e:
            st.sidebar.error(str(e))
        except ConnectionError as e:
            st.sidebar.error(str(e))
        except Exception as e:
            st.sidebar.error(f"Weather fetch failed: {str(e)}")

default_temperature = st.session_state.get("weather_temperature", 25)
default_humidity = st.session_state.get("weather_humidity", 60)

if st.session_state.get("weather_name"):
    st.sidebar.caption(f"Last fetched: {st.session_state['weather_name']}")

N = st.sidebar.slider("Nitrogen (N) - mg/kg", min_value=0, max_value=140, value=50, step=1)
P = st.sidebar.slider("Phosphorus (P) - mg/kg", min_value=5, max_value=145, value=40, step=1)
K = st.sidebar.slider("Potassium (K) - mg/kg", min_value=5, max_value=205, value=40, step=1)
pH = st.sidebar.slider("pH Value", min_value=3.5, max_value=9.5, value=6.5, step=0.1)
temperature = st.sidebar.slider("Temperature - °C", min_value=8, max_value=43, value=int(default_temperature), step=1)
humidity = st.sidebar.slider("Humidity - %", min_value=14, max_value=99, value=int(default_humidity), step=1)
rainfall = st.sidebar.slider("Rainfall - mm", min_value=20, max_value=298, value=100, step=5)

features_dict = {
    "N": N,
    "P": P,
    "K": K,
    "pH": pH,
    "temperature": temperature,
    "humidity": humidity,
    "rainfall": rainfall,
}

input_df = pd.DataFrame([features_dict], columns=feature_names)
probabilities = model.predict_proba(input_df)[0]
ranked_indices = np.argsort(probabilities)[::-1][:3]

ranked_recommendations = []
for rank_idx in ranked_indices:
    crop_name = encoder.inverse_transform([rank_idx])[0]
    probability_percent = round(float(probabilities[rank_idx] * 100), 1)
    ranked_recommendations.append((crop_name, probability_percent))

predicted_crop, confidence = ranked_recommendations[0]

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown(
        """
        <div class='prediction-box'>
            <h2>🥇 Primary Recommendation: <span style='color: #2E7D32;'>{}</span></h2>
            <p>Based on your soil and weather conditions</p>
        </div>
        """.format(predicted_crop),
        unsafe_allow_html=True,
    )
    st.caption("The first-ranked crop is the most suitable based on the current soil and weather conditions.")
with col2:
    st.markdown(
        """
        <div class='confidence-box'>
            <h3>🏆 Top 3 Crop Recommendations</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    for rank, (crop_name, probability_percent) in enumerate(ranked_recommendations, start=1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
        st.markdown(
            f"<div style='margin-top: 0.35rem; padding: 0.4rem 0.6rem; border-radius: 6px; background-color: #F1F8E9;'>{medal} <strong>{crop_name}</strong> — {probability_percent:.1f}%</div>",
            unsafe_allow_html=True,
        )

st.progress(confidence / 100)

crop_info = load_crop_info()
crop_details = get_crop_harvest_info(predicted_crop, crop_info)
if crop_details:
    timeline_html = "<br>".join(crop_details.get("timeline", []))
    st.markdown(
        f"""
        <div class='prediction-box'>
            <h3>🌾 Recommended Crop</h3>
            <p><strong>{crop_details['crop_name']}</strong></p>
            <p><strong>⏳ Harvest Duration</strong><br>{crop_details['min_harvest_days']}–{crop_details['max_harvest_days']} Days</p>
            <p><strong>📅 Estimated Harvest Date</strong><br>{crop_details['estimated_harvest_date']}</p>
            <p><strong>🌱 Crop Timeline</strong><br>{timeline_html}</p>
            <p><strong>📝 Description</strong><br>{crop_details['description']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.info(
        "Harvest information is not available for the recommended crop right now. Please try again later."
    )

st.markdown("---")
st.markdown("<h3>⚠️ Risk Assessment</h3>", unsafe_allow_html=True)
warnings_list = assess_conditions_risk(features_dict, df)
if warnings_list:
    for warning in warnings_list:
        st.markdown(f"""
            <div class='warning-box'>
                {warning}
            </div>
        """, unsafe_allow_html=True)
else:
    st.success("✅ All conditions are within ideal ranges!")

st.markdown("---")
st.markdown("<h3>💡 Farming Insights</h3>", unsafe_allow_html=True)
insights = generate_insights(predicted_crop, features_dict, encoder)
for insight in insights:
    st.markdown(f"""
        <div class='insight-box'>
            {insight}
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("<h2>🔍 Explainable AI Analysis</h2>", unsafe_allow_html=True)
shap_values, base_value, X = explain_prediction(model, encoder, features_dict, feature_names)
class_idx = np.where(encoder.classes_ == predicted_crop)[0][0]

if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    shap_values_class = shap_values[0, :, class_idx]
    base_val = base_value[class_idx]
else:
    raise ValueError(f"Unexpected SHAP values format: type={type(shap_values)}, ndim={shap_values.ndim if isinstance(shap_values, np.ndarray) else 'N/A'}")

exp_col1, exp_col2 = st.columns(2)
with exp_col1:
    st.subheader("📈 Feature Importance (Model-wide)")
    feature_imp = get_feature_importance(model, feature_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2E7D32" if x > 0.1 else "#558B2F" for x in feature_imp["importances"]]
    ax.barh(feature_imp["features"], feature_imp["importances"], color=colors)
    ax.set_xlabel("Importance Score", fontsize=11, fontweight="bold")
    ax.set_title("Which features matter most for prediction?", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

with exp_col2:
    st.subheader("🌊 SHAP Waterfall (This Prediction)")
    explanation = shap.Explanation(
        values=shap_values_class,
        base_values=base_val,
        data=X.iloc[0].values,
        feature_names=feature_names,
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.waterfall_plot(explanation, show=False)
    st.pyplot(fig)

st.subheader("📊 How Each Input Influenced the Prediction")
shap_df = pd.DataFrame(
    {
        "Feature": feature_names,
        "Value": [features_dict[f] for f in feature_names],
        "SHAP Impact": shap_values_class,
        "Direction": ["Positive ⬆️" if x > 0 else "Negative ⬇️" for x in shap_values_class],
    }
)
shap_df["Impact_Magnitude"] = np.abs(shap_df["SHAP Impact"])
shap_df = shap_df.sort_values("Impact_Magnitude", ascending=False)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Feature**")
    for feat in shap_df["Feature"]:
        st.write(feat)
with col2:
    st.markdown("**Input Value**")
    for val in shap_df["Value"]:
        st.write(f"{val:.2f}")
with col3:
    st.markdown("**Impact Direction**")
    for impact, direction in zip(shap_df["SHAP Impact"], shap_df["Direction"]):
        color = "🟢" if impact > 0 else "🔴"
        st.write(f"{color} {impact:.3f} {direction}")

st.markdown("---")
st.markdown("<h2>🔬 Detailed Feature Analysis</h2>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Nitrogen", f"{N} mg/kg", delta="Nutrient")
with col2:
    st.metric("Phosphorus", f"{P} mg/kg", delta="Nutrient")
with col3:
    st.metric("Potassium", f"{K} mg/kg", delta="Nutrient")
with col4:
    st.metric("pH Value", f"{pH:.1f}", delta="Soil")

col5, col6, col7 = st.columns(3)
with col5:
    st.metric("Temperature", f"{temperature}°C", delta="Weather")
with col6:
    st.metric("Humidity", f"{humidity}%", delta="Weather")
with col7:
    st.metric("Rainfall", f"{rainfall} mm", delta="Weather")

report_text = build_prediction_report(
    predicted_crop,
    confidence,
    features_dict,
    st.session_state.get("weather_name", "Not provided"),
    st.session_state.get("weather_condition", "Unknown"),
    shap_values_class,
    base_val,
    feature_names,
    "\n".join([f"{row['Feature']}: {row['SHAP Impact']:.3f}" for _, row in shap_df.iterrows()]),
)
st.download_button(
    label="Download Report",
    data=report_text,
    file_name="crop_prediction_report.txt",
    mime="text/plain",
)

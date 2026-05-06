"""
Explainable Farming AI - A Streamlit application for crop recommendation with explainable AI.

Features:
- Recommends the best crop based on soil and weather conditions
- Provides SHAP-based explanations for predictions
- Shows feature importance and confidence scores
- Displays risk warnings for non-ideal conditions
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import shap
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Explainable Farming AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
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
""", unsafe_allow_html=True)

# ============================================================================
# MODEL FUNCTIONS
# ============================================================================

@st.cache_resource
def load_model_and_data():
    """Load or train the model and prepare data."""
    model_path = 'models/crop_model.pkl'
    encoder_path = 'models/label_encoder.pkl'
    data_path = 'data/crop_recommendation.csv'
    
    # Check if models exist
    if os.path.exists(model_path) and os.path.exists(encoder_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(encoder_path, 'rb') as f:
            encoder = pickle.load(f)
        df = pd.read_csv(data_path)
        return model, encoder, df
    
    # If dataset doesn't exist, raise error
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            "Please run data_generator.py first."
        )
    
    # Load and prepare data
    df = pd.read_csv(data_path)
    
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Encode labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y_encoded)
    
    # Save model and encoder
    os.makedirs('models', exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(encoder_path, 'wb') as f:
        pickle.dump(encoder, f)
    
    return model, encoder, df

def predict_crop(model, encoder, features_dict):
    """Predict crop and confidence."""
    X = np.array([list(features_dict.values())]).reshape(1, -1)
    
    # Prediction
    prediction = model.predict(X)[0]
    crop_name = encoder.inverse_transform([prediction])[0]
    
    # Confidence
    prediction_proba = model.predict_proba(X)[0]
    confidence = np.max(prediction_proba) * 100
    
    return crop_name, confidence

def explain_prediction(model, encoder, features_dict, feature_names):
    """Generate SHAP explanation.
    
    Returns SHAP values in a standardized format.
    For RandomForestClassifier with TreeExplainer:
    - Returns (1, n_features, n_classes) numpy array
    - Base values: (n_classes,) array
    """
    X = np.array([list(features_dict.values())]).reshape(1, -1)
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    base_value = explainer.expected_value
    
    return shap_values, base_value, X

def get_feature_importance(model, feature_names):
    """Get feature importance from model."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    return {
        'features': [feature_names[i] for i in indices],
        'importances': [importances[i] for i in indices]
    }

def assess_conditions_risk(features_dict, df):
    """Assess risk based on input conditions."""
    warnings_list = []
    
    N = features_dict['N']
    P = features_dict['P']
    K = features_dict['K']
    pH = features_dict['pH']
    temp = features_dict['temperature']
    humidity = features_dict['humidity']
    rainfall = features_dict['rainfall']
    
    # Check for extreme values
    if N < 20:
        warnings_list.append("⚠️ Very low Nitrogen - May reduce crop yield")
    elif N > 130:
        warnings_list.append("⚠️ Very high Nitrogen - May cause leaf burn")
    
    if pH < 4.5:
        warnings_list.append("⚠️ Soil too acidic - Consider adding lime")
    elif pH > 8.5:
        warnings_list.append("⚠️ Soil too alkaline - May lock nutrients")
    
    if temp < 10:
        warnings_list.append("⚠️ Temperature too low - Growth may be slow")
    elif temp > 40:
        warnings_list.append("⚠️ Temperature too high - Heat stress risk")
    
    if humidity < 20:
        warnings_list.append("⚠️ Very low humidity - Irrigation needed")
    elif humidity > 95:
        warnings_list.append("⚠️ Very high humidity - Disease risk")
    
    if rainfall < 25:
        warnings_list.append("⚠️ Low rainfall - Irrigation recommended")
    
    return warnings_list

def generate_insights(crop_name, features_dict, encoder):
    """Generate human-readable insights."""
    insights = []
    
    N = features_dict['N']
    P = features_dict['P']
    K = features_dict['K']
    pH = features_dict['pH']
    temp = features_dict['temperature']
    humidity = features_dict['humidity']
    rainfall = features_dict['rainfall']
    
    # Nutrient insights
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
    
    # Weather insights
    if temp > 25 and humidity > 70:
        insights.append("☀️ Warm and humid conditions favor tropical crops")
    elif temp < 20:
        insights.append("❄️ Cool climate suits temperate crops like apple and lentil")
    
    if rainfall > 150:
        insights.append("💧 High rainfall supports moisture-loving crops")
    else:
        insights.append("🏜️ Low rainfall requires drought-resistant varieties")
    
    return insights

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Title
    st.markdown("<h1 class='main-title'>🌾 Explainable Farming AI</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Smart Crop Recommendation with AI Explainability</p>", unsafe_allow_html=True)
    
    # Load model
    try:
        model, encoder, df = load_model_and_data()
    except FileNotFoundError as e:
        st.error(f"❌ {str(e)}")
        st.info("Please run the data_generator.py script first to create the dataset.")
        return
    
    feature_names = ['N', 'P', 'K', 'pH', 'temperature', 'humidity', 'rainfall']
    
    # ========================================================================
    # SIDEBAR - USER INPUTS
    # ========================================================================
    st.sidebar.header("🌍 Input Soil & Weather Conditions")
    st.sidebar.info("Adjust the values using sliders or enter manually")
    
    # Nitrogen
    N = st.sidebar.slider(
        "Nitrogen (N) - mg/kg",
        min_value=0,
        max_value=140,
        value=50,
        step=1,
        help="Nitrogen content in soil (0-140)"
    )
    
    # Phosphorus
    P = st.sidebar.slider(
        "Phosphorus (P) - mg/kg",
        min_value=5,
        max_value=145,
        value=40,
        step=1,
        help="Phosphorus content in soil (5-145)"
    )
    
    # Potassium
    K = st.sidebar.slider(
        "Potassium (K) - mg/kg",
        min_value=5,
        max_value=205,
        value=40,
        step=1,
        help="Potassium content in soil (5-205)"
    )
    
    # pH
    pH = st.sidebar.slider(
        "pH Value",
        min_value=3.5,
        max_value=9.5,
        value=6.5,
        step=0.1,
        help="Soil pH (3.5-9.5)"
    )
    
    # Temperature
    temperature = st.sidebar.slider(
        "Temperature - °C",
        min_value=8,
        max_value=43,
        value=25,
        step=1,
        help="Average temperature (8-43°C)"
    )
    
    # Humidity
    humidity = st.sidebar.slider(
        "Humidity - %",
        min_value=14,
        max_value=99,
        value=60,
        step=1,
        help="Relative humidity (14-99%)"
    )
    
    # Rainfall
    rainfall = st.sidebar.slider(
        "Rainfall - mm",
        min_value=20,
        max_value=298,
        value=100,
        step=5,
        help="Annual rainfall (20-298 mm)"
    )
    
    # Create features dictionary
    features_dict = {
        'N': N,
        'P': P,
        'K': K,
        'pH': pH,
        'temperature': temperature,
        'humidity': humidity,
        'rainfall': rainfall
    }
    
    # ========================================================================
    # MAIN CONTENT
    # ========================================================================
    
    # Make prediction
    predicted_crop, confidence = predict_crop(model, encoder, features_dict)
    
    # Display prediction
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
            <div class='prediction-box'>
                <h2>🎯 Predicted Crop: <span style='color: #2E7D32;'>{}</span></h2>
                <p>Based on your soil and weather conditions</p>
            </div>
        """.format(predicted_crop), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='confidence-box'>
                <h3>📊 Confidence Score</h3>
                <h2 style='color: #F57F17;'>{:.1f}%</h2>
                <p>Model certainty</p>
            </div>
        """.format(confidence), unsafe_allow_html=True)
    
    # Display confidence meter
    st.progress(confidence / 100)
    
    # ========================================================================
    # RISK WARNINGS
    # ========================================================================
    st.markdown("---")
    warnings_list = assess_conditions_risk(features_dict, df)
    
    if warnings_list:
        st.markdown("<h3>⚠️ Risk Assessment</h3>", unsafe_allow_html=True)
        for warning in warnings_list:
            st.markdown(f"""
                <div class='warning-box'>
                    {warning}
                </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ All conditions are within ideal ranges!")
    
    # ========================================================================
    # INSIGHTS
    # ========================================================================
    st.markdown("---")
    st.markdown("<h3>💡 Farming Insights</h3>", unsafe_allow_html=True)
    
    insights = generate_insights(predicted_crop, features_dict, encoder)
    for insight in insights:
        st.markdown(f"""
            <div class='insight-box'>
                {insight}
            </div>
        """, unsafe_allow_html=True)
    
    # ========================================================================
    # EXPLAINABILITY SECTION
    # ========================================================================
    st.markdown("---")
    st.markdown("<h2>🔍 Explainable AI Analysis</h2>", unsafe_allow_html=True)
    
    # Get SHAP explanation
    shap_values, base_value, X = explain_prediction(model, encoder, features_dict, feature_names)
    
    # Get the class index for the predicted crop name
    class_idx = np.where(encoder.classes_ == predicted_crop)[0][0]
    
    # Extract SHAP values for the predicted class
    # shap_values has shape (1, n_features, n_classes)
    # We need shape (n_features,) for the waterfall plot
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # Standard format: (n_samples, n_features, n_classes)
        shap_values_class = shap_values[0, :, class_idx]  # Shape: (n_features,)
        base_val = base_value[class_idx]  # Scalar
    else:
        raise ValueError(f"Unexpected SHAP values format: type={type(shap_values)}, ndim={shap_values.ndim if isinstance(shap_values, np.ndarray) else 'N/A'}")
    
    
    # Create columns for explanations
    exp_col1, exp_col2 = st.columns(2)
    
    # Feature Importance Bar Chart
    with exp_col1:
        st.subheader("📈 Feature Importance (Model-wide)")
        feature_imp = get_feature_importance(model, feature_names)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#2E7D32' if x > 0.1 else '#558B2F' for x in feature_imp['importances']]
        ax.barh(feature_imp['features'], feature_imp['importances'], color=colors)
        ax.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
        ax.set_title('Which features matter most for prediction?', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
    
    # SHAP Waterfall Plot
    with exp_col2:
        st.subheader("🌊 SHAP Waterfall (This Prediction)")
        
        # Create SHAP waterfall explanation
        explanation = shap.Explanation(
            values=shap_values_class,
            base_values=base_val,
            data=X[0],
            feature_names=feature_names
        )
        
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.waterfall_plot(explanation, show=False)
        st.pyplot(fig)
    
    # Feature Impact on This Prediction
    st.subheader("📊 How Each Input Influenced the Prediction")
    
    # Create a dataframe for SHAP values
    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'Value': [features_dict[f] for f in feature_names],
        'SHAP Impact': shap_values_class,
        'Direction': ['Positive ⬆️' if x > 0 else 'Negative ⬇️' for x in shap_values_class]
    })
    
    shap_df['Impact_Magnitude'] = np.abs(shap_df['SHAP Impact'])
    shap_df = shap_df.sort_values('Impact_Magnitude', ascending=False)
    
    # Display as table
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Feature**")
        for feat in shap_df['Feature']:
            st.write(feat)
    
    with col2:
        st.markdown("**Input Value**")
        for val in shap_df['Value']:
            st.write(f"{val:.2f}")
    
    with col3:
        st.markdown("**Impact Direction**")
        for impact, direction in zip(shap_df['SHAP Impact'], shap_df['Direction']):
            color = '🟢' if impact > 0 else '🔴'
            st.write(f"{color} {impact:.3f} {direction}")
    
    # ========================================================================
    # DETAILED FEATURE ANALYSIS
    # ========================================================================
    st.markdown("---")
    st.markdown("<h2>🔬 Detailed Feature Analysis</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Nitrogen", f"{N} mg/kg", delta="Nutrient" )
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
    
    # ========================================================================
    # DATASET STATISTICS
    # ========================================================================
    with st.expander("📚 Dataset Statistics & Reference"):
        st.markdown("### Crop Distribution in Training Data")
        crop_counts = df['label'].value_counts()
        st.bar_chart(crop_counts)
        
        st.markdown("### Summary Statistics")
        st.dataframe(df.describe(), use_container_width=True)
        
        st.markdown("### Feature Ranges")
        ranges_data = {
            'Feature': feature_names,
            'Min': [df[f].min() for f in feature_names],
            'Max': [df[f].max() for f in feature_names],
            'Mean': [df[f].mean() for f in feature_names],
            'Std Dev': [df[f].std() for f in feature_names]
        }
        st.dataframe(pd.DataFrame(ranges_data), use_container_width=True)

if __name__ == "__main__":
    main()

"""
Debug script to understand SHAP values structure for RandomForestClassifier
"""

import numpy as np
import pandas as pd
import pickle
import shap
from sklearn.preprocessing import LabelEncoder

# Load model and data
model_path = 'models/crop_model.pkl'
encoder_path = 'models/label_encoder.pkl'
data_path = 'data/crop_recommendation.csv'

with open(model_path, 'rb') as f:
    model = pickle.load(f)
with open(encoder_path, 'rb') as f:
    encoder = pickle.load(f)
df = pd.read_csv(data_path)

print("=" * 80)
print("MODEL INFORMATION")
print("=" * 80)
print(f"Model type: {type(model)}")
print(f"Number of classes: {len(encoder.classes_)}")
print(f"Classes: {encoder.classes_}")
print(f"Model n_classes_: {model.n_classes_}")

# Create a sample input
sample_features = df.iloc[0, :-1].values  # Exclude target
print(f"\nSample input shape: {sample_features.shape}")
print(f"Sample input: {sample_features}")

# Prepare for SHAP
X_sample = sample_features.reshape(1, -1)
print(f"Reshaped X: {X_sample.shape}")

# Create SHAP explainer
print("\n" + "=" * 80)
print("CREATING SHAP EXPLAINER")
print("=" * 80)
explainer = shap.TreeExplainer(model)

# Get SHAP values
shap_values = explainer.shap_values(X_sample)
base_value = explainer.expected_value

print(f"SHAP values type: {type(shap_values)}")
print(f"Base value type: {type(base_value)}")

# Check structure
if isinstance(shap_values, list):
    print(f"\n✓ SHAP values is a LIST with {len(shap_values)} elements")
    for i, sv in enumerate(shap_values):
        if isinstance(sv, np.ndarray):
            print(f"  Class {i}: shape = {sv.shape}, dtype = {sv.dtype}")
        else:
            print(f"  Class {i}: type = {type(sv)}")
            
elif isinstance(shap_values, np.ndarray):
    print(f"\n✓ SHAP values is a NUMPY ARRAY")
    print(f"  Shape: {shap_values.shape}")
    print(f"  Dtype: {shap_values.dtype}")
    print(f"  Ndim: {shap_values.ndim}")

if isinstance(base_value, list):
    print(f"\n✓ Base value is a LIST with {len(base_value)} elements")
    print(f"  Values: {base_value}")
elif isinstance(base_value, np.ndarray):
    print(f"\n✓ Base value is a NUMPY ARRAY")
    print(f"  Shape: {base_value.shape}")
    print(f"  Values: {base_value}")
else:
    print(f"\n✓ Base value is a SCALAR: {base_value}")

# Make prediction to show which class we'd access
print("\n" + "=" * 80)
print("PREDICTION INFORMATION")
print("=" * 80)
prediction = model.predict(X_sample)
prediction_proba = model.predict_proba(X_sample)

print(f"Raw prediction from model: {prediction[0]} (type: {type(prediction[0])})")
print(f"Prediction probabilities shape: {prediction_proba[0].shape}")

# The prediction is already the class index! Encode it back to see the name
try:
    predicted_class_name = encoder.inverse_transform([prediction[0]])[0]
    print(f"Predicted crop name: {predicted_class_name}")
except:
    print(f"Could not decode prediction, it's likely already an index: {prediction[0]}")

print(f"\nAll class probabilities: {prediction_proba[0]}")

# Show the issue
print("\n" + "=" * 80)
print("THE ISSUE & SOLUTION")
print("=" * 80)
class_idx = int(prediction[0])  # The prediction is already an index!
print(f"Predicted class index: {class_idx}")

if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    print(f"SHAP values shape: {shap_values.shape} = (n_samples, n_features, n_classes)")
    print(f"\n✓ CORRECT ACCESS for class {class_idx}:")
    print(f"  shap_values_class = shap_values[0, :, class_idx]")
    shap_values_class = shap_values[0, :, class_idx]
    print(f"  Result shape: {shap_values_class.shape}")
    print(f"  Base value for this class: {base_value[class_idx]}")

print("\n" + "=" * 80)
print("RECOMMENDED SOLUTION")
print("=" * 80)
print("""
For RandomForestClassifier with multiple classes, TreeExplainer returns:
- List of arrays: One array per class output (binary or multi-class)
  Shape of each array: (n_samples, n_features)

For a single prediction:
- shap_values is a list of length = number of classes
- To access values for predicted class: shap_values[class_idx][0]
- To get base value: base_value[class_idx]

BUT for binary classification, SHAP may only return values for the positive class!
So check: len(shap_values) == 1 or len(shap_values) == n_classes
""")

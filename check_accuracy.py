import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

# Load data
df = pd.read_csv('data/crop_recommendation.csv')
X = df.drop('label', axis=1)
y = df['label']

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {acc*100:.2f}%')

# Cross-validation
cv_scores = cross_val_score(model, X, y_encoded, cv=5)
print(f'Cross-Validation Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)')

# Per-crop accuracy
print()
print(classification_report(y_test, y_pred, target_names=encoder.classes_))
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

# Load data
df = pd.read_csv('data/crop_recommendation.csv')
X = df.drop('label', axis=1)
y = df['label']

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

print("Testing different models...\n")

# Model 1: Better Random Forest
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,       # No depth limit
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))
print(f"✅ Improved Random Forest : {rf_acc*100:.2f}%")

# Model 2: Gradient Boosting
gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
gb.fit(X_train, y_train)
gb_acc = accuracy_score(y_test, gb.predict(X_test))
print(f"✅ Gradient Boosting      : {gb_acc*100:.2f}%")

# Model 3: Try with more data
from data_generator import generate_crop_dataset
print("\nGenerating larger dataset (5000 samples)...")
df_large = generate_crop_dataset(n_samples=5000)
X_large = df_large.drop('label', axis=1)
y_large = encoder.fit_transform(df_large['label'])

X_train2, X_test2, y_train2, y_test2 = train_test_split(X_large, y_large, test_size=0.2, random_state=42)

rf2 = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
rf2.fit(X_train2, y_train2)
rf2_acc = accuracy_score(y_test2, rf2.predict(X_test2))
print(f"✅ Random Forest (5000 samples): {rf2_acc*100:.2f}%")

# Cross-validation on best model
cv = cross_val_score(rf2, X_large, y_large, cv=5)
print(f"\n🏆 Best Model CV Accuracy: {cv.mean()*100:.2f}% (+/- {cv.std()*100:.2f}%)")

print("\nPer-crop breakdown:")
y_pred2 = rf2.predict(X_test2)
print(classification_report(y_test2, y_pred2, target_names=encoder.classes_))

# Save the better model
import pickle, os
os.makedirs('models', exist_ok=True)
with open('models/crop_model.pkl', 'wb') as f:
    pickle.dump(rf2, f)
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

# Save larger dataset
os.makedirs('data', exist_ok=True)
df_large.to_csv('data/crop_recommendation.csv', index=False)

print("\n✅ Improved model and dataset saved!")
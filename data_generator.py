import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import pickle, os

def generate_better_dataset(n_samples=5000, random_state=42):
    np.random.seed(random_state)

    # Crops with MORE DISTINCT conditions (less overlap)
    crops = {
        'rice':        dict(N=(100,120), P=(55,75), K=(55,75), pH=(6.0,7.0), temp=(22,28), hum=(78,90), rain=(180,240)),
        'maize':       dict(N=(90,110), P=(35,55), K=(35,55), pH=(6.2,7.2), temp=(22,26), hum=(62,72), rain=(55,90)),
        'chickpea':    dict(N=(22,35), P=(22,38), K=(22,38), pH=(6.8,7.8), temp=(16,22), hum=(42,55), rain=(42,58)),
        'kidneybeans': dict(N=(22,35), P=(28,52), K=(22,38), pH=(6.2,6.8), temp=(22,28), hum=(52,68), rain=(68,115)),
        'pigeonpeas':  dict(N=(22,35), P=(22,38), K=(28,52), pH=(6.2,7.2), temp=(20,28), hum=(52,72), rain=(62,95)),
        'mothbeans':   dict(N=(22,35), P=(22,38), K=(22,38), pH=(7.2,8.0), temp=(28,36), hum=(18,32), rain=(18,35)),
        'mungbean':    dict(N=(22,35), P=(22,38), K=(22,38), pH=(6.2,6.8), temp=(22,28), hum=(52,68), rain=(62,95)),
        'blackgram':   dict(N=(22,35), P=(22,38), K=(22,38), pH=(6.2,7.2), temp=(24,30), hum=(62,72), rain=(62,95)),
        'lentil':      dict(N=(22,35), P=(22,38), K=(22,38), pH=(6.2,6.8), temp=(14,22), hum=(42,58), rain=(42,75)),
        'pomegranate': dict(N=(45,75), P=(22,38), K=(45,75), pH=(5.8,7.8), temp=(25,30), hum=(22,45), rain=(22,95)),
        'banana':      dict(N=(105,135), P=(45,75), K=(105,155), pH=(5.8,7.2), temp=(25,28), hum=(72,88), rain=(185,245)),
        'mango':       dict(N=(82,115), P=(22,38), K=(82,115), pH=(5.8,7.2), temp=(25,28), hum=(62,78), rain=(42,95)),
        'grapes':      dict(N=(82,115), P=(22,38), K=(82,115), pH=(6.2,7.2), temp=(22,26), hum=(42,65), rain=(42,58)),
        'watermelon':  dict(N=(82,115), P=(22,38), K=(22,38), pH=(6.2,6.8), temp=(24,32), hum=(52,78), rain=(42,95)),
        'muskmelon':   dict(N=(82,115), P=(45,75), K=(22,38), pH=(6.2,6.8), temp=(26,32), hum=(52,78), rain=(42,95)),
        'apple':       dict(N=(82,115), P=(22,38), K=(82,115), pH=(6.2,7.2), temp=(10,18), hum=(62,78), rain=(62,115)),
        'orange':      dict(N=(82,115), P=(22,38), K=(82,115), pH=(6.2,7.2), temp=(18,24), hum=(62,78), rain=(62,115)),
        'papaya':      dict(N=(105,135), P=(22,38), K=(82,115), pH=(6.2,7.2), temp=(28,34), hum=(72,88), rain=(155,245)),
        'coconut':     dict(N=(105,135), P=(45,75), K=(105,155), pH=(5.8,7.8), temp=(27,33), hum=(72,88), rain=(155,245)),
        'cotton':      dict(N=(105,135), P=(22,38), K=(45,75), pH=(6.2,7.2), temp=(22,30), hum=(62,78), rain=(42,95)),
        'sugarcane':   dict(N=(105,135), P=(45,75), K=(45,75), pH=(6.2,7.2), temp=(22,26), hum=(72,88), rain=(155,245)),
        'tobacco':     dict(N=(22,38), P=(22,38), K=(22,38), pH=(6.2,6.8), temp=(22,28), hum=(62,78), rain=(42,95)),
        'arecanut':    dict(N=(82,115), P=(22,38), K=(45,75), pH=(6.2,7.2), temp=(26,33), hum=(72,88), rain=(185,245)),
    }

    data = []
    per_crop = n_samples // len(crops)

    for crop, c in crops.items():
        for _ in range(per_crop):
            sample = {
                'N':           np.clip(np.random.normal(np.mean(c['N']),    5),   0,   140),
                'P':           np.clip(np.random.normal(np.mean(c['P']),    5),   5,   145),
                'K':           np.clip(np.random.normal(np.mean(c['K']),    5),   5,   205),
                'pH':          np.clip(np.random.normal(np.mean(c['pH']),   0.2), 3.5, 9.5),
                'temperature': np.clip(np.random.normal(np.mean(c['temp']), 1.5), 8,   43),
                'humidity':    np.clip(np.random.normal(np.mean(c['hum']),  5),   14,  99),
                'rainfall':    np.clip(np.random.normal(np.mean(c['rain']), 15),  20,  298),
                'label': crop
            }
            data.append(sample)

    return pd.DataFrame(data)


print("Generating improved dataset...")
df = generate_better_dataset(n_samples=5000)

X = df.drop('label', axis=1)
y = df['label']

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Test Accuracy: {acc*100:.2f}%")

cv = cross_val_score(model, X, y_encoded, cv=5)
print(f"✅ CV Accuracy:   {cv.mean()*100:.2f}% (+/- {cv.std()*100:.2f}%)")

print("\nPer-crop report:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# Save everything
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

with open('models/crop_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)
df.to_csv('data/crop_recommendation.csv', index=False)

print("✅ Improved model and dataset saved! Restart your Streamlit app.")
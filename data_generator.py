"""
Generate a crop recommendation dataset for training the ML model.
This script creates a realistic dataset with soil and weather features.
"""

import pandas as pd
import numpy as np
import os

def generate_crop_dataset(n_samples=2000, random_state=42):
    """
    Generate a crop recommendation dataset.
    
    Features:
    - N: Nitrogen (0-140)
    - P: Phosphorus (5-145)
    - K: Potassium (5-205)
    - pH: pH value (3.5-9.5)
    - Temperature: Temperature in Celsius (8-43)
    - Humidity: Humidity in % (14-99)
    - Rainfall: Rainfall in mm (20-298)
    
    Target: Crop type
    """
    np.random.seed(random_state)
    
    # Define crop characteristics (optimal conditions)
    crops = {
        'rice': {
            'N': (80, 120),
            'P': (40, 80),
            'K': (40, 80),
            'pH': (5.5, 7.5),
            'temperature': (20, 30),
            'humidity': (70, 90),
            'rainfall': (150, 250)
        },
        'maize': {
            'N': (80, 120),
            'P': (20, 60),
            'K': (20, 60),
            'pH': (6.0, 7.5),
            'temperature': (21, 27),
            'humidity': (60, 80),
            'rainfall': (40, 100)
        },
        'chickpea': {
            'N': (20, 40),
            'P': (20, 40),
            'K': (20, 40),
            'pH': (6.5, 8.0),
            'temperature': (15, 25),
            'humidity': (40, 60),
            'rainfall': (40, 60)
        },
        'kidneybeans': {
            'N': (20, 40),
            'P': (20, 60),
            'K': (20, 40),
            'pH': (6.0, 7.0),
            'temperature': (20, 30),
            'humidity': (50, 70),
            'rainfall': (60, 120)
        },
        'pigeonpeas': {
            'N': (20, 40),
            'P': (20, 40),
            'K': (20, 60),
            'pH': (6.0, 7.5),
            'temperature': (18, 30),
            'humidity': (50, 80),
            'rainfall': (60, 100)
        },
        'mothbeans': {
            'N': (20, 40),
            'P': (20, 40),
            'K': (20, 40),
            'pH': (7.0, 8.0),
            'temperature': (20, 35),
            'humidity': (20, 40),
            'rainfall': (20, 40)
        },
        'mungbean': {
            'N': (20, 40),
            'P': (20, 40),
            'K': (20, 40),
            'pH': (6.0, 7.0),
            'temperature': (20, 30),
            'humidity': (50, 70),
            'rainfall': (60, 100)
        },
        'blackgram': {
            'N': (20, 40),
            'P': (20, 40),
            'K': (20, 40),
            'pH': (6.0, 7.5),
            'temperature': (20, 30),
            'humidity': (50, 70),
            'rainfall': (60, 100)
        },
        'lentil': {
            'N': (20, 40),
            'P': (20, 40),
            'K': (20, 40),
            'pH': (6.0, 7.0),
            'temperature': (15, 25),
            'humidity': (40, 60),
            'rainfall': (40, 80)
        },
        'pomegranate': {
            'N': (40, 80),
            'P': (20, 40),
            'K': (40, 80),
            'pH': (5.5, 8.0),
            'temperature': (24, 29),
            'humidity': (20, 50),
            'rainfall': (20, 100)
        },
        'banana': {
            'N': (100, 140),
            'P': (40, 80),
            'K': (100, 160),
            'pH': (5.5, 7.5),
            'temperature': (24, 28),
            'humidity': (70, 90),
            'rainfall': (180, 250)
        },
        'mango': {
            'N': (80, 120),
            'P': (20, 40),
            'K': (80, 120),
            'pH': (5.5, 7.5),
            'temperature': (24, 28),
            'humidity': (60, 80),
            'rainfall': (40, 100)
        },
        'grapes': {
            'N': (80, 120),
            'P': (20, 40),
            'K': (80, 120),
            'pH': (6.0, 7.5),
            'temperature': (20, 28),
            'humidity': (40, 70),
            'rainfall': (40, 60)
        },
        'watermelon': {
            'N': (80, 120),
            'P': (20, 40),
            'K': (20, 40),
            'pH': (6.0, 7.0),
            'temperature': (21, 32),
            'humidity': (50, 80),
            'rainfall': (40, 100)
        },
        'muskmelon': {
            'N': (80, 120),
            'P': (40, 80),
            'K': (20, 40),
            'pH': (6.0, 7.0),
            'temperature': (21, 32),
            'humidity': (50, 80),
            'rainfall': (40, 100)
        },
        'apple': {
            'N': (80, 120),
            'P': (20, 40),
            'K': (80, 120),
            'pH': (6.0, 7.5),
            'temperature': (10, 21),
            'humidity': (60, 80),
            'rainfall': (60, 120)
        },
        'orange': {
            'N': (80, 120),
            'P': (20, 40),
            'K': (80, 120),
            'pH': (6.0, 7.5),
            'temperature': (17, 27),
            'humidity': (60, 80),
            'rainfall': (60, 120)
        },
        'papaya': {
            'N': (100, 140),
            'P': (20, 40),
            'K': (80, 120),
            'pH': (6.0, 7.5),
            'temperature': (21, 32),
            'humidity': (70, 90),
            'rainfall': (150, 250)
        },
        'coconut': {
            'N': (100, 140),
            'P': (40, 80),
            'K': (100, 160),
            'pH': (5.5, 8.0),
            'temperature': (24, 32),
            'humidity': (70, 90),
            'rainfall': (150, 250)
        },
        'cotton': {
            'N': (100, 140),
            'P': (20, 40),
            'K': (40, 80),
            'pH': (6.0, 7.5),
            'temperature': (20, 30),
            'humidity': (60, 80),
            'rainfall': (40, 100)
        },
        'sugarcane': {
            'N': (100, 140),
            'P': (40, 80),
            'K': (40, 80),
            'pH': (6.0, 7.5),
            'temperature': (21, 27),
            'humidity': (70, 90),
            'rainfall': (150, 250)
        },
        'tobacco': {
            'N': (20, 40),
            'P': (20, 40),
            'K': (20, 40),
            'pH': (6.0, 7.0),
            'temperature': (20, 30),
            'humidity': (60, 80),
            'rainfall': (40, 100)
        },
        'arecanut': {
            'N': (80, 120),
            'P': (20, 40),
            'K': (40, 80),
            'pH': (6.0, 7.5),
            'temperature': (24, 32),
            'humidity': (70, 90),
            'rainfall': (180, 250)
        }
    }
    
    data = []
    samples_per_crop = n_samples // len(crops)
    
    for crop, conditions in crops.items():
        for _ in range(samples_per_crop):
            # Generate data with noise around optimal conditions
            sample = {
                'N': np.random.normal(np.mean(conditions['N']), 15),
                'P': np.random.normal(np.mean(conditions['P']), 15),
                'K': np.random.normal(np.mean(conditions['K']), 15),
                'pH': np.random.normal(np.mean(conditions['pH']), 0.3),
                'temperature': np.random.normal(np.mean(conditions['temperature']), 2),
                'humidity': np.random.normal(np.mean(conditions['humidity']), 8),
                'rainfall': np.random.normal(np.mean(conditions['rainfall']), 20),
                'label': crop
            }
            data.append(sample)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Clip values to realistic ranges
    df['N'] = df['N'].clip(0, 140)
    df['P'] = df['P'].clip(5, 145)
    df['K'] = df['K'].clip(5, 205)
    df['pH'] = df['pH'].clip(3.5, 9.5)
    df['temperature'] = df['temperature'].clip(8, 43)
    df['humidity'] = df['humidity'].clip(14, 99)
    df['rainfall'] = df['rainfall'].clip(20, 298)
    
    return df

def main():
    """Generate and save the dataset."""
    print("Generating crop recommendation dataset...")
    df = generate_crop_dataset(n_samples=2000)
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save dataset
    csv_path = 'data/crop_recommendation.csv'
    df.to_csv(csv_path, index=False)
    print(f"✅ Dataset saved to {csv_path}")
    print(f"\nDataset shape: {df.shape}")
    print(f"Crops: {df['label'].unique().tolist()}")
    print(f"\nFirst few rows:\n{df.head()}")
    print(f"\nDataset statistics:\n{df.describe()}")

if __name__ == "__main__":
    main()

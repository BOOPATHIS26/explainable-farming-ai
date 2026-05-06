"""
Example configurations and use cases for Explainable Farming AI
This file demonstrates different scenarios and their expected outputs.
"""

# Example Input Scenarios
SCENARIOS = {
    "Rice Growing Region": {
        "description": "Ideal conditions for rice cultivation in monsoon region",
        "inputs": {
            "N": 80,
            "P": 40,
            "K": 40,
            "pH": 6.5,
            "temperature": 25,
            "humidity": 75,
            "rainfall": 200
        },
        "expected_crop": "rice",
        "expected_confidence_range": (75, 95),
        "explanation": "High nitrogen, warm temperature, high humidity and rainfall favor rice"
    },
    
    "Maize Belt Conditions": {
        "description": "Typical conditions in maize growing regions",
        "inputs": {
            "N": 100,
            "P": 30,
            "K": 30,
            "pH": 6.8,
            "temperature": 26,
            "humidity": 65,
            "rainfall": 70
        },
        "expected_crop": "maize",
        "expected_confidence_range": (70, 90),
        "explanation": "High nitrogen, moderate potassium, warm and moderately dry conditions suit maize"
    },
    
    "Legume Growing": {
        "description": "Conditions suitable for legume crops (beans, peas, lentils)",
        "inputs": {
            "N": 25,
            "P": 35,
            "K": 35,
            "pH": 7.0,
            "temperature": 22,
            "humidity": 55,
            "rainfall": 50
        },
        "expected_crop": "chickpea or lentil",
        "expected_confidence_range": (65, 85),
        "explanation": "Low nitrogen (legumes fix nitrogen), neutral pH, cool and dry conditions suit legumes"
    },
    
    "Tropical Fruit Region": {
        "description": "Ideal conditions for tropical fruit crops",
        "inputs": {
            "N": 120,
            "P": 60,
            "K": 100,
            "pH": 6.5,
            "temperature": 28,
            "humidity": 80,
            "rainfall": 180
        },
        "expected_crop": "banana or coconut",
        "expected_confidence_range": (70, 88),
        "explanation": "High nutrients, warm, humid, and wet conditions favor tropical fruits"
    },
    
    "Temperate Apple Growing": {
        "description": "Cool conditions suitable for apple orchards",
        "inputs": {
            "N": 100,
            "P": 30,
            "K": 100,
            "pH": 6.5,
            "temperature": 18,
            "humidity": 70,
            "rainfall": 80
        },
        "expected_crop": "apple",
        "expected_confidence_range": (75, 90),
        "explanation": "Cool temperature, moderate rainfall, and good potassium suit apple cultivation"
    },
    
    "Arid Region Dry Farming": {
        "description": "Low water availability, dry climate conditions",
        "inputs": {
            "N": 35,
            "P": 25,
            "K": 35,
            "pH": 7.5,
            "temperature": 32,
            "humidity": 30,
            "rainfall": 30
        },
        "expected_crop": "moth beans or watermelon",
        "expected_confidence_range": (60, 80),
        "explanation": "Low rainfall, high temperature, low humidity require drought-resistant crops"
    },
    
    "Sugarcane Zone": {
        "description": "Conditions for commercial sugarcane cultivation",
        "inputs": {
            "N": 120,
            "P": 60,
            "K": 60,
            "pH": 6.5,
            "temperature": 24,
            "humidity": 80,
            "rainfall": 180
        },
        "expected_crop": "sugarcane",
        "expected_confidence_range": (75, 90),
        "explanation": "High nitrogen and potassium, warm, humid conditions with good rainfall suit sugarcane"
    },
}

# Feature Value Interpretations
FEATURE_INTERPRETATIONS = {
    "N (Nitrogen)": {
        "very_low": (0, 20),
        "low": (20, 50),
        "medium": (50, 80),
        "high": (80, 120),
        "very_high": (120, 140),
        "best_for": "Rice, Sugarcane, Banana: 100-130 mg/kg",
        "risk": "Low N: Limited growth; High N: Leaf burn, excess vegetative growth"
    },
    
    "P (Phosphorus)": {
        "very_low": (5, 15),
        "low": (15, 30),
        "medium": (30, 60),
        "high": (60, 100),
        "very_high": (100, 145),
        "best_for": "Most crops: 40-80 mg/kg",
        "risk": "Low P: Poor root development; High P: Nutrient imbalance"
    },
    
    "K (Potassium)": {
        "very_low": (5, 20),
        "low": (20, 40),
        "medium": (40, 80),
        "high": (80, 140),
        "very_high": (140, 205),
        "best_for": "Fruits, Sugarcane: 100-160 mg/kg",
        "risk": "Low K: Disease susceptibility; High K: Nutrient imbalance"
    },
    
    "pH": {
        "very_acidic": (3.5, 4.5),
        "acidic": (4.5, 5.5),
        "slightly_acidic": (5.5, 6.5),
        "neutral": (6.5, 7.5),
        "alkaline": (7.5, 8.5),
        "very_alkaline": (8.5, 9.5),
        "best_for": "Most crops: 6.0-7.5",
        "risk": "Too acidic: Nutrient leaching; Too alkaline: Nutrient lockup"
    },
    
    "Temperature": {
        "very_cool": (8, 15),
        "cool": (15, 20),
        "moderate": (20, 25),
        "warm": (25, 30),
        "hot": (30, 35),
        "very_hot": (35, 43),
        "best_for": "Most crops: 20-28°C",
        "risk": "Too cold: Slow growth; Too hot: Heat stress, wilting"
    },
    
    "Humidity": {
        "very_dry": (14, 25),
        "dry": (25, 40),
        "moderate": (40, 60),
        "humid": (60, 80),
        "very_humid": (80, 99),
        "best_for": "Most crops: 60-80%",
        "risk": "Too dry: Water stress; Too humid: Disease risk, fungal infections"
    },
    
    "Rainfall": {
        "very_low": (20, 40),
        "low": (40, 80),
        "moderate": (80, 150),
        "high": (150, 200),
        "very_high": (200, 298),
        "best_for": "Most crops: 50-150 mm",
        "risk": "Too low: Drought; Too high: Waterlogging, root rot"
    },
}

# Risk Thresholds
RISK_THRESHOLDS = {
    "nitrogen": {
        "critical_low": 15,
        "critical_high": 135,
        "warning": "N < 20 or N > 130"
    },
    "pH": {
        "critical_low": 4.0,
        "critical_high": 9.0,
        "warning": "pH < 4.5 or pH > 8.5"
    },
    "temperature": {
        "critical_low": 10,
        "critical_high": 38,
        "warning": "Temp < 12 or Temp > 35"
    },
    "humidity": {
        "critical_low": 15,
        "critical_high": 98,
        "warning": "Humidity < 20 or Humidity > 95"
    },
}

# Crop Characteristics Reference
CROP_CHARACTERISTICS = {
    "rice": {
        "ideal_N": (80, 120),
        "ideal_P": (40, 80),
        "ideal_K": (40, 80),
        "ideal_pH": (5.5, 7.5),
        "ideal_temp": (20, 30),
        "ideal_humidity": (70, 90),
        "ideal_rainfall": (150, 250),
        "season": "Monsoon",
        "water_requirement": "High",
        "soil_type": "Clay loam, Clay"
    },
    
    "maize": {
        "ideal_N": (80, 120),
        "ideal_P": (20, 60),
        "ideal_K": (20, 60),
        "ideal_pH": (6.0, 7.5),
        "ideal_temp": (21, 27),
        "ideal_humidity": (60, 80),
        "ideal_rainfall": (40, 100),
        "season": "Kharif/Rabi",
        "water_requirement": "Moderate",
        "soil_type": "Well-drained loam"
    },
    
    "chickpea": {
        "ideal_N": (20, 40),
        "ideal_P": (20, 40),
        "ideal_K": (20, 40),
        "ideal_pH": (6.5, 8.0),
        "ideal_temp": (15, 25),
        "ideal_humidity": (40, 60),
        "ideal_rainfall": (40, 60),
        "season": "Rabi",
        "water_requirement": "Low",
        "soil_type": "Well-drained loam"
    },
}

def print_scenario(scenario_name: str):
    """Print details of a specific scenario."""
    if scenario_name not in SCENARIOS:
        print(f"Scenario '{scenario_name}' not found!")
        print(f"Available scenarios: {list(SCENARIOS.keys())}")
        return
    
    scenario = SCENARIOS[scenario_name]
    print(f"\n{'='*60}")
    print(f"Scenario: {scenario_name}")
    print(f"{'='*60}")
    print(f"Description: {scenario['description']}")
    print(f"\nInputs:")
    for key, value in scenario['inputs'].items():
        print(f"  {key:15s}: {value:6.1f}")
    print(f"\nExpected Output:")
    print(f"  Crop: {scenario['expected_crop']}")
    print(f"  Confidence: {scenario['expected_confidence_range'][0]}-{scenario['expected_confidence_range'][1]}%")
    print(f"\nExplanation: {scenario['explanation']}")
    print(f"{'='*60}\n")

def print_all_scenarios():
    """Print all available scenarios."""
    print("\nAvailable Test Scenarios:")
    print("-" * 60)
    for i, (name, scenario) in enumerate(SCENARIOS.items(), 1):
        print(f"{i}. {name}")
        print(f"   Description: {scenario['description']}")
    print("-" * 60)

if __name__ == "__main__":
    print_all_scenarios()
    print("\nExample: print_scenario('Rice Growing Region')")

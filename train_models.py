import os
import pickle
import time
import warnings
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


warnings.filterwarnings("ignore")

FEATURE_NAMES = ["N", "P", "K", "pH", "temperature", "humidity", "rainfall"]
TARGET_COLUMN = "label"
DATASET_CANDIDATES = [Path("data/Crop.csv"), Path("data/crop_recommendation.csv")]
MODEL_OUTPUT_PATH = Path("models/crop_model.pkl")
ENCODER_OUTPUT_PATH = Path("models/label_encoder.pkl")
COMPARISON_OUTPUT_PATH = Path("models/model_comparison.csv")


def load_dataset():
    """Load the crop dataset and keep the same column normalization as the app utilities."""
    for candidate in DATASET_CANDIDATES:
        if candidate.exists():
            df = pd.read_csv(candidate)
            if "ph" in df.columns and "pH" not in df.columns:
                df = df.rename(columns={"ph": "pH"})
            if "PH" in df.columns and "pH" not in df.columns:
                df = df.rename(columns={"PH": "pH"})
            return df

    raise FileNotFoundError("Dataset not found. Please place your dataset file at data/Crop.csv.")


def prepare_training_data(df):
    """Apply the same preprocessing logic used by the application: feature selection and label encoding."""
    if not set(FEATURE_NAMES + [TARGET_COLUMN]).issubset(df.columns):
        raise ValueError("The dataset is missing one or more required columns for training.")

    X = df[FEATURE_NAMES]
    y = df[TARGET_COLUMN]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    return X, y_encoded, encoder


def split_data(X, y):
    """Split data into train and test sets with a stable, class-aware configuration."""
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def build_model_registry():
    """Create a list of candidate classification models for benchmarking."""
    return [
        ("Logistic Regression", LogisticRegression(max_iter=500, random_state=42)),
        ("Decision Tree", DecisionTreeClassifier(random_state=42)),
        ("KNN", KNeighborsClassifier(n_neighbors=5)),
        ("Random Forest", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
        ("Gradient Boosting", GradientBoostingClassifier(random_state=42)),
    ]


def evaluate_model(name, model, X_train, X_test, y_train, y_test, X, y):
    """Fit a model, compute standard classification metrics, and collect stability statistics."""
    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - start_time

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

    return {
        "Model": name,
        "Accuracy": round(float(accuracy), 4),
        "Precision": round(float(precision), 4),
        "Recall": round(float(recall), 4),
        "F1 Score": round(float(f1), 4),
        "CV Mean": round(float(cv_scores.mean()), 4),
        "CV Std": round(float(cv_scores.std()), 4),
        "Training Time (s)": round(float(training_time), 4),
        "Confusion Matrix": cm,
        "CV Scores": cv_scores,
        "Model Object": model,
    }


def select_best_model(results):
    """Select the strongest model using cross-validation performance, accuracy, generalization, and stability."""
    tree_based_models = [result for result in results if result["Model"] in {"Decision Tree", "Random Forest", "Gradient Boosting"}]
    candidates = tree_based_models if tree_based_models else results

    def ranking_key(result):
        return (
            result["CV Mean"],
            result["Accuracy"],
            result["F1 Score"],
            -result["CV Std"],
            -result["Training Time (s)"],
        )

    return max(candidates, key=ranking_key)


def print_comparison_table(results):
    """Display a compact comparison table for all candidate models."""
    comparison_df = pd.DataFrame(
        [
            {
                "Model": result["Model"],
                "Accuracy": result["Accuracy"],
                "Precision": result["Precision"],
                "Recall": result["Recall"],
                "F1 Score": result["F1 Score"],
                "CV Mean": result["CV Mean"],
                "CV Std": result["CV Std"],
                "Training Time (s)": result["Training Time (s)"],
            }
            for result in results
        ]
    )

    print("\n" + "-" * 56)
    print("Model Comparison")
    print("-" * 56)
    print(comparison_df.to_string(index=False))
    print("-" * 56)


def print_model_summary(results):
    """Print a simple deployment-style summary for each model."""
    print("\n" + "-" * 56)
    print("Model Comparison")
    print("-" * 56)
    for result in results:
        print(f"{result['Model']:<24} : {result['Accuracy'] * 100:.2f}%")
    print("-" * 56)


def save_comparison_report(results):
    """Export the evaluation metrics to a CSV file for later review."""
    comparison_df = pd.DataFrame(
        [
            {
                "Model": result["Model"],
                "Accuracy": result["Accuracy"],
                "Precision": result["Precision"],
                "Recall": result["Recall"],
                "F1 Score": result["F1 Score"],
                "CV Mean": result["CV Mean"],
                "CV Std": result["CV Std"],
                "Training Time (s)": result["Training Time (s)"],
            }
            for result in results
        ]
    )
    COMPARISON_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(COMPARISON_OUTPUT_PATH, index=False)


def print_confusion_matrices(results):
    """Print confusion matrices for each model to support deeper inspection."""
    for result in results:
        print(f"\nConfusion Matrix - {result['Model']}")
        print(result["Confusion Matrix"])


def save_best_model(best_result, encoder):
    """Persist the selected model and label encoder using the paths expected by the app."""
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_OUTPUT_PATH.open("wb") as model_file:
        pickle.dump(best_result["Model Object"], model_file)

    with ENCODER_OUTPUT_PATH.open("wb") as encoder_file:
        pickle.dump(encoder, encoder_file)


def summarize_best_model(best_result, results):
    """Print a concise deployment summary matching the requested format."""
    print("\n" + "-" * 56)
    print("Best Model Selected")
    print(best_result["Model"])
    print("Reason:")

    reasons = []
    if best_result["CV Mean"] == max(result["CV Mean"] for result in results):
        reasons.append("Highest Cross Validation Score")
    if best_result["Accuracy"] == max(result["Accuracy"] for result in results):
        reasons.append("Highest Accuracy")
    if best_result["CV Std"] == min(result["CV Std"] for result in results):
        reasons.append("Stable Performance")
    if best_result["F1 Score"] == max(result["F1 Score"] for result in results):
        reasons.append("Best Generalization")

    for reason in reasons:
        print(f"- {reason}")

    print("Model saved successfully.")
    print("-" * 56)


def main():
    """Run the end-to-end training and evaluation workflow."""
    print("Loading dataset...")
    df = load_dataset()

    print("Applying preprocessing pipeline...")
    X, y, encoder = prepare_training_data(df)

    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Training and evaluating multiple models...")
    results = []
    for name, model in build_model_registry():
        result = evaluate_model(name, model, X_train, X_test, y_train, y_test, X, y)
        results.append(result)

    print_comparison_table(results)
    print_model_summary(results)
    save_comparison_report(results)
    print_confusion_matrices(results)

    best_result = select_best_model(results)
    save_best_model(best_result, encoder)
    summarize_best_model(best_result, results)


if __name__ == "__main__":
    main()

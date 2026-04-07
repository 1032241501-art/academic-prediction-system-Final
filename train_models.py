# =============================================================
# train_models.py
# Purpose: Train ML models and save them for use in the Streamlit app
#
# Models Trained:
#   1. Linear Regression   → Marks Prediction
#   2. Logistic Regression → Placement Prediction
#
# Run this AFTER generate_datasets.py and BEFORE app.py
# =============================================================

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Create folders to save models and charts
os.makedirs("models", exist_ok=True)
os.makedirs("charts", exist_ok=True)


# ==============================================================
# MODULE 1: MARKS PREDICTION — Linear Regression
# ==============================================================
# Why Linear Regression?
# Marks is a continuous numeric value (0–100).
# Linear Regression finds the best-fit line between inputs and marks.
# It is simple, interpretable, and perfect for this use case.
# ==============================================================

def train_marks_model():
    print("=" * 55)
    print("  MODULE 1: Marks Prediction (Linear Regression)")
    print("=" * 55)

    # --- Load Dataset ---
    df = pd.read_csv("data/student_marks.csv")
    print(f"\n  Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"  Sample:\n{df.head(3)}\n")

    # --- Features and Target ---
    X = df[["study_hours", "attendance", "previous_score"]]
    y = df["marks"]

    # --- Train-Test Split (80% train, 20% test) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Train size: {len(X_train)} | Test size: {len(X_test)}")

    # --- Feature Scaling (StandardScaler) ---
    # Scales features to have mean=0 and std=1
    # Helps the model converge better
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Train the Linear Regression Model ---
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # --- Predictions ---
    y_pred = model.predict(X_test_scaled)

    # --- Evaluation Metrics ---
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)

    print("\n  📊 Evaluation Metrics:")
    print(f"     MAE  (Mean Absolute Error)       : {mae:.2f}")
    print(f"     MSE  (Mean Squared Error)         : {mse:.2f}")
    print(f"     RMSE (Root Mean Squared Error)    : {rmse:.2f}")
    print(f"     R²   (R-Squared Score)            : {r2:.4f}")
    print(f"\n  Interpretation:")
    print(f"     R² of {r2:.2f} means the model explains {r2*100:.1f}% of variance in marks.")

    # --- Save Model and Scaler ---
    joblib.dump(model, "models/marks_model.pkl")
    joblib.dump(scaler, "models/marks_scaler.pkl")
    print("\n  [✓] marks_model.pkl saved to /models")
    print("  [✓] marks_scaler.pkl saved to /models")

    # --- Save Metrics for Dashboard ---
    metrics = {
        "mae": round(mae, 2),
        "mse": round(mse, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4)
    }
    import json
    with open("models/marks_metrics.json", "w") as f:
        json.dump(metrics, f)
    print("  [✓] marks_metrics.json saved to /models")

    # --- Plot: Actual vs Predicted Marks ---
    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, y_pred, color="steelblue", alpha=0.6, edgecolors="white", s=60)
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             color="tomato", linewidth=2, linestyle="--", label="Ideal Line")
    plt.xlabel("Actual Marks")
    plt.ylabel("Predicted Marks")
    plt.title("Marks Prediction: Actual vs Predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig("charts/marks_actual_vs_predicted.png")
    plt.close()
    print("  [✓] Chart saved: charts/marks_actual_vs_predicted.png\n")

    return metrics


# ==============================================================
# MODULE 2: PLACEMENT PREDICTION — Logistic Regression
# ==============================================================
# Why Logistic Regression?
# Placement is a binary outcome (Placed: 1 or Not Placed: 0).
# Logistic Regression is ideal for binary classification problems.
# It outputs a probability between 0 and 1, which is easy to explain.
# ==============================================================

def train_placement_model():
    print("=" * 55)
    print("  MODULE 2: Placement Prediction (Logistic Regression)")
    print("=" * 55)

    # --- Load Dataset ---
    df = pd.read_csv("data/placement_data.csv")
    print(f"\n  Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"  Sample:\n{df.head(3)}\n")
    print(f"  Class Distribution:")
    print(f"     Placed     (1): {df['placed'].sum()}")
    print(f"     Not Placed (0): {(df['placed'] == 0).sum()}\n")

    # --- Features and Target ---
    X = df[["cgpa", "skills", "projects", "communication_score"]]
    y = df["placed"]

    # --- Train-Test Split (80% train, 20% test) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train size: {len(X_train)} | Test size: {len(X_test)}")

    # --- Feature Scaling ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Train Logistic Regression Model ---
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    # --- Predictions ---
    y_pred = model.predict(X_test_scaled)

    # --- Evaluation Metrics ---
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    cm        = confusion_matrix(y_test, y_pred)

    print("\n  📊 Evaluation Metrics:")
    print(f"     Accuracy  : {accuracy*100:.2f}%")
    print(f"     Precision : {precision:.4f}")
    print(f"     Recall    : {recall:.4f}")
    print(f"     F1 Score  : {f1:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Not Placed", "Placed"]))
    print(f"  Confusion Matrix:\n{cm}")

    # --- Save Model and Scaler ---
    joblib.dump(model, "models/placement_model.pkl")
    joblib.dump(scaler, "models/placement_scaler.pkl")
    print("\n  [✓] placement_model.pkl saved to /models")
    print("  [✓] placement_scaler.pkl saved to /models")

    # --- Save Metrics for Dashboard ---
    metrics = {
        "accuracy": round(accuracy * 100, 2),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "confusion_matrix": cm.tolist()
    }
    import json
    with open("models/placement_metrics.json", "w") as f:
        json.dump(metrics, f)
    print("  [✓] placement_metrics.json saved to /models")

    # --- Plot: Confusion Matrix Heatmap ---
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Not Placed", "Placed"],
        yticklabels=["Not Placed", "Placed"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Placement Prediction: Confusion Matrix")
    plt.tight_layout()
    plt.savefig("charts/placement_confusion_matrix.png")
    plt.close()
    print("  [✓] Chart saved: charts/placement_confusion_matrix.png\n")

    return metrics


# ==============================================================
# MAIN: Run both training pipelines
# ==============================================================
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("   ACADEMIC PERFORMANCE & PLACEMENT PREDICTION SYSTEM")
    print("   Model Training Script")
    print("=" * 55 + "\n")

    # Check if datasets exist
    if not os.path.exists("data/student_marks.csv"):
        print("  [!] student_marks.csv not found. Run generate_datasets.py first!")
        exit(1)
    if not os.path.exists("data/placement_data.csv"):
        print("  [!] placement_data.csv not found. Run generate_datasets.py first!")
        exit(1)

    # Train both models
    marks_metrics     = train_marks_model()
    placement_metrics = train_placement_model()

    print("=" * 55)
    print("  ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")
    print("  You can now run:  streamlit run app.py")
    print("=" * 55 + "\n")

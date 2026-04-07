# =============================================================
# generate_datasets.py
# Purpose: Create realistic sample datasets for both ML modules
# Run this script ONCE before training models or starting the app
# =============================================================

import pandas as pd
import numpy as np
import os

# Set seed for reproducibility
np.random.seed(42)

# --------------------------------------------------
# Create the data/ directory if it doesn't exist
# --------------------------------------------------
os.makedirs("data", exist_ok=True)


# ==============================================================
# DATASET 1: Student Marks Prediction
# Features: study_hours, attendance, previous_score → marks
# Algorithm used: Linear Regression (continuous output)
# ==============================================================

def generate_marks_dataset(n=300):
    """
    Generates a realistic student marks dataset.
    Marks are calculated as a weighted combination of inputs
    with some added noise to simulate real-world variability.
    """

    # Feature 1: Daily study hours (1 to 10)
    study_hours = np.round(np.random.uniform(1, 10, n), 1)

    # Feature 2: Attendance percentage (50% to 100%)
    attendance = np.round(np.random.uniform(50, 100, n), 1)

    # Feature 3: Previous exam score (30 to 95)
    previous_score = np.round(np.random.uniform(30, 95, n), 1)

    # Target: Marks (realistic formula + noise)
    # More study + high attendance + good previous score → higher marks
    noise = np.random.normal(0, 3, n)
    marks = (
        3.5 * study_hours
        + 0.25 * attendance
        + 0.40 * previous_score
        + noise
    )

    # Clip marks to valid range [0, 100]
    marks = np.clip(np.round(marks, 1), 0, 100)

    # Build DataFrame
    df = pd.DataFrame({
        "study_hours": study_hours,
        "attendance": attendance,
        "previous_score": previous_score,
        "marks": marks
    })

    return df


# ==============================================================
# DATASET 2: Student Placement Prediction
# Features: cgpa, skills, projects, communication_score → placed
# Algorithm used: Logistic Regression (binary classification)
# ==============================================================

def generate_placement_dataset(n=300):
    """
    Generates a realistic placement prediction dataset.
    Placement (1/0) is determined by a probability score
    derived from the student's profile inputs.
    """

    # Feature 1: CGPA (5.0 to 10.0)
    cgpa = np.round(np.random.uniform(5.0, 10.0, n), 2)

    # Feature 2: Number of technical skills known (1 to 10)
    skills = np.random.randint(1, 11, n)

    # Feature 3: Number of projects completed (0 to 8)
    projects = np.random.randint(0, 9, n)

    # Feature 4: Communication score (1 to 10)
    communication_score = np.round(np.random.uniform(1, 10, n), 1)

    # Target: Placed (1 = Yes, 0 = No)
    # A score is computed and passed through a sigmoid to get probability
    score = (
        0.5 * cgpa
        + 0.3 * skills
        + 0.4 * projects
        + 0.3 * communication_score
        - 4.0  # Bias term to balance classes
    )

    # Sigmoid function: converts score to probability
    probability = 1 / (1 + np.exp(-score))

    # Random placement outcome based on probability
    placed = (np.random.rand(n) < probability).astype(int)

    # Build DataFrame
    df = pd.DataFrame({
        "cgpa": cgpa,
        "skills": skills,
        "projects": projects,
        "communication_score": communication_score,
        "placed": placed
    })

    return df


# --------------------------------------------------
# Main: Generate and save both datasets
# --------------------------------------------------
if __name__ == "__main__":
    print("Generating datasets...")

    # Generate marks dataset
    marks_df = generate_marks_dataset(n=300)
    marks_df.to_csv("data/student_marks.csv", index=False)
    print(f"  [✓] student_marks.csv saved — {len(marks_df)} records")
    print(f"      Columns: {list(marks_df.columns)}")
    print(f"      Marks range: {marks_df['marks'].min()} – {marks_df['marks'].max()}\n")

    # Generate placement dataset
    placement_df = generate_placement_dataset(n=300)
    placement_df.to_csv("data/placement_data.csv", index=False)
    print(f"  [✓] placement_data.csv saved — {len(placement_df)} records")
    print(f"      Columns: {list(placement_df.columns)}")
    placed_count = placement_df['placed'].sum()
    print(f"      Placed: {placed_count} | Not Placed: {len(placement_df) - placed_count}\n")

    print("Datasets generated successfully in the /data folder.")

# =============================================================
# clean_real_data.py
# Purpose: Clean and convert real Kaggle datasets into the
#          exact format required by this project
#
# Input files (place in data/raw/ folder):
#   - Student_Performance.csv         (Kaggle - marks dataset)
#   - Placement_Data_Full_Class.csv   (Kaggle - placement dataset)
#
# Output files (saved to data/ folder):
#   - student_marks.csv
#   - placement_data.csv
#
# Run: python clean_real_data.py
# =============================================================

import pandas as pd
import numpy as np
import os

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data", exist_ok=True)

print("=" * 60)
print("  REAL DATA CLEANING SCRIPT")
print("=" * 60)


# ==============================================================
# PART 1: Clean Student Performance Dataset → student_marks.csv
#
# Original Kaggle columns:
#   Hours Studied           → study_hours
#   Previous Scores         → previous_score
#   Sleep Hours             (dropped — not in our model)
#   Extracurricular         (dropped)
#   Sample Question Papers  → used to boost attendance estimate
#   Performance Index       → marks
#
# Attendance is NOT in this dataset — we derive a realistic
# estimate from Hours Studied + Sample Papers Practiced
# ==============================================================

def clean_marks_dataset():
    print("\n[ PART 1 ] Cleaning Student Performance Dataset...")

    filepath = "data/raw/Student_Performance.csv"

    # --- Check file exists ---
    if not os.path.exists(filepath):
        print(f"  ❌ File not found: {filepath}")
        print("  Please download from Kaggle and place in data/raw/ folder.")
        print("  Link: https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression")
        return False

    # --- Load raw data ---
    df = pd.read_csv(filepath)
    print(f"  Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"  Original columns: {list(df.columns)}")

    # --- Rename columns to match project format ---
    df.rename(columns={
        "Hours Studied":                   "study_hours",
        "Previous Scores":                 "previous_score",
        "Sample Question Papers Practiced":"sample_papers",
        "Performance Index":               "marks",
        "Sleep Hours":                     "sleep_hours",
        "Extracurricular Activities":      "extracurricular"
    }, inplace=True)

    # --- Drop rows with missing values ---
    before = len(df)
    df.dropna(subset=["study_hours", "previous_score", "marks"], inplace=True)
    print(f"  Dropped {before - len(df)} rows with missing values")

    # --- Derive attendance column ---
    # Since this dataset has no attendance column, we estimate it:
    # Base: 55 + study contribution + sample papers contribution + noise
    # This gives a realistic 50–100 range that correlates with study hours
    np.random.seed(42)
    df["attendance"] = (
        55
        + (df["study_hours"] / df["study_hours"].max()) * 30
        + (df["sample_papers"] / df["sample_papers"].max()) * 10
        + np.random.normal(0, 3, len(df))
    )
    df["attendance"] = df["attendance"].clip(50, 100).round(1)

    # --- Scale marks to 0–100 range if needed ---
    # Performance Index in this dataset is already 10–100
    df["marks"] = df["marks"].clip(0, 100).round(1)

    # --- Scale study_hours to realistic range (1–10) ---
    df["study_hours"] = df["study_hours"].clip(1, 10).round(1)

    # --- Keep only required columns ---
    df_clean = df[["study_hours", "attendance", "previous_score", "marks"]].copy()

    # --- Final validation ---
    print(f"  Final shape: {df_clean.shape}")
    print(f"  study_hours  range: {df_clean['study_hours'].min()} – {df_clean['study_hours'].max()}")
    print(f"  attendance   range: {df_clean['attendance'].min()} – {df_clean['attendance'].max()}")
    print(f"  prev_score   range: {df_clean['previous_score'].min()} – {df_clean['previous_score'].max()}")
    print(f"  marks        range: {df_clean['marks'].min()} – {df_clean['marks'].max()}")
    print(f"  Missing values: {df_clean.isnull().sum().sum()}")

    # --- Save ---
    df_clean.to_csv("data/student_marks.csv", index=False)
    print(f"  ✅ Saved: data/student_marks.csv ({len(df_clean)} rows)")
    return True


# ==============================================================
# PART 2: Clean Campus Placement Dataset → placement_data.csv
#
# Original Kaggle columns:
#   degree_p         → cgpa  (degree percentage → converted to /10)
#   etest_p          → used to estimate skills score
#   workex           → used to estimate projects count
#   mba_p            → used to estimate communication score
#   status           → placed (Placed=1, Not Placed=0)
#
# Note: This dataset has ~215 rows. We augment to 1000+ rows
# using realistic jitter so the model trains more robustly.
# ==============================================================

def clean_placement_dataset():
    print("\n[ PART 2 ] Cleaning Campus Placement Dataset...")

    filepath = "data/raw/Placement_Data_Full_Class.csv"

    # --- Check file exists ---
    if not os.path.exists(filepath):
        print(f"  ❌ File not found: {filepath}")
        print("  Please download from Kaggle and place in data/raw/ folder.")
        print("  Link: https://www.kaggle.com/datasets/benroshan/factors-affecting-campus-placement")
        return False

    # --- Load raw data ---
    df = pd.read_csv(filepath)
    print(f"  Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"  Original columns: {list(df.columns)}")

    # IMPORTANT:
    # Do NOT use df.dropna(inplace=True) on the full dataset,
    # because 'salary' is often missing for Not Placed students.
    # That would remove many/all class 0 rows.

    # --- Keep only relevant columns for this project ---
    required_columns = ["degree_p", "etest_p", "workex", "mba_p", "status"]
    df = df[required_columns].copy()

    # --- Drop missing values only in required columns ---
    before = len(df)
    df.dropna(subset=required_columns, inplace=True)
    print(f"  Dropped {before - len(df)} rows with missing values in required columns only")

    # --- Debug: show original status distribution before mapping ---
    print("\n  Status distribution in cleaned raw data:")
    print(df["status"].value_counts())

    # --- Map 'status' to binary placed column ---
    df["placed"] = df["status"].apply(
        lambda x: 1 if str(x).strip().lower() == "placed" else 0
    )

    # --- Convert degree percentage to CGPA scale (out of 10) ---
    df["cgpa"] = (df["degree_p"] / 10).clip(4.0, 10.0).round(2)

    # --- Derive 'skills' from employability test percentage ---
    df["skills"] = ((df["etest_p"] / 100) * 9 + 1).clip(1, 10).round(0).astype(int)

    # --- Derive 'projects' from work experience ---
    np.random.seed(42)
    df["projects"] = df["workex"].apply(
        lambda x: np.random.randint(3, 7) if str(x).strip().lower() == "yes"
        else np.random.randint(0, 4)
    )

    # --- Derive 'communication_score' from MBA percentage ---
    df["communication_score"] = ((df["mba_p"] / 100) * 9 + 1).clip(1, 10).round(1)

    # --- Keep only required columns ---
    df_clean = df[["cgpa", "skills", "projects", "communication_score", "placed"]].copy()

    print(f"\n  Class distribution before augmentation:")
    print(df_clean["placed"].value_counts())

    # --- Safety check ---
    if df_clean["placed"].nunique() < 2:
        print("  ❌ ERROR: Only one class found in 'placed' column.")
        print("  Logistic Regression requires both 0 and 1 classes.")
        return False

    # --- Augment dataset to 1200 rows ---
    print(f"\n  Augmenting dataset to 1200 rows with realistic jitter...")
    augmented_rows = []
    np.random.seed(99)

    while len(df_clean) + len(augmented_rows) < 1200:
        row = df_clean.sample(1).iloc[0]

        new_row = {
            "cgpa": round(float(np.clip(row["cgpa"] + np.random.normal(0, 0.3), 4.0, 10.0)), 2),
            "skills": int(np.clip(row["skills"] + np.random.randint(-1, 2), 1, 10)),
            "projects": int(np.clip(row["projects"] + np.random.randint(-1, 2), 0, 8)),
            "communication_score": round(float(np.clip(row["communication_score"] + np.random.normal(0, 0.5), 1.0, 10.0)), 1),
            "placed": int(row["placed"])
        }

        augmented_rows.append(new_row)

    df_augmented = pd.DataFrame(augmented_rows)
    df_final = pd.concat([df_clean, df_augmented], ignore_index=True)
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n  Final shape: {df_final.shape}")
    print(f"  cgpa                range: {df_final['cgpa'].min()} – {df_final['cgpa'].max()}")
    print(f"  skills              range: {df_final['skills'].min()} – {df_final['skills'].max()}")
    print(f"  projects            range: {df_final['projects'].min()} – {df_final['projects'].max()}")
    print(f"  communication_score range: {df_final['communication_score'].min()} – {df_final['communication_score'].max()}")

    print(f"\n  Class distribution after augmentation:")
    print(df_final["placed"].value_counts())

    print(f"\n  Missing values: {df_final.isnull().sum().sum()}")

    # --- Save ---
    df_final.to_csv("data/placement_data.csv", index=False)
    print(f"  ✅ Saved: data/placement_data.csv ({len(df_final)} rows)")
    return True


# ==============================================================
# MAIN
# ==============================================================
if __name__ == "__main__":
    print("\nPlace your Kaggle CSV files in the data/raw/ folder first.")
    print("Then this script will clean and convert them.\n")

    marks_ok     = clean_marks_dataset()
    placement_ok = clean_placement_dataset()

    print("\n" + "=" * 60)
    if marks_ok and placement_ok:
        print("  ✅ BOTH DATASETS CLEANED SUCCESSFULLY!")
        print("  Next step: python train_models.py")
    else:
        print("  ⚠️  Some files were missing. Check messages above.")
        print("  Download the datasets from Kaggle and try again.")
    print("=" * 60 + "\n")

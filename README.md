# 🎓 Academic Performance & Placement Prediction System

A beginner-friendly **Machine Learning mini project** built using **Python, Streamlit, Pandas, and Scikit-learn**.

This application predicts:

1. **Student Academic Performance / Marks**
2. **Student Placement Status**

It also provides:
- improvement recommendations
- skill suggestions
- dashboard insights
- model evaluation metrics

---

## 📌 Project Modules

- **Marks Prediction** using Linear Regression
- **Placement Prediction** using Logistic Regression
- **Recommendations Module**
- **Skill Suggestion Module**
- **Dashboard for Summary Insights**
- **About Project Page**

---

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

---

## 📁 Folder Structure

```bash
academic_prediction_system/
│
├── app.py
├── utils.py
├── train_models.py
├── clean_real_data.py
├── generate_datasets.py
├── requirements.txt
├── README.md
├── PROJECT_REPORT.md
│
├── data/
│   ├── student_marks.csv
│   └── placement_data.csv
│
├── models/
│   ├── marks_model.pkl
│   ├── marks_scaler.pkl
│   ├── placement_model.pkl
│   ├── placement_scaler.pkl
│   ├── marks_metrics.json
│   └── placement_metrics.json
│
└── charts/
    ├── marks_actual_vs_predicted.png
    └── placement_confusion_matrix.png
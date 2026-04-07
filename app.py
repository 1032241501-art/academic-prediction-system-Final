# =============================================================
# app.py
# Project: Academic Performance & Placement Prediction System
# Tech Stack: Python, Streamlit, Scikit-learn, Pandas, Matplotlib
# Run: streamlit run app.py
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

from utils import (
    get_marks_recommendations,
    get_placement_recommendations,
    get_skill_suggestions,
    placement_probability_label,
    SKILL_CATALOG
)

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Academic & Placement Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>
.main-title {
    font-size: 2.3rem;
    font-weight: 800;
    color: #2F80ED;
    text-align: center;
    margin-bottom: 0.4rem;
}
.subtitle {
    font-size: 1.05rem;
    color: #b0b0b0;
    text-align: center;
    margin-bottom: 1.5rem;
}
.section-subtitle {
    font-size: 0.98rem;
    color: #c7c7c7;
    margin-bottom: 1rem;
}
.footer-text {
    font-size: 0.85rem;
    color: #9aa0a6;
    text-align: center;
    margin-top: 2rem;
}
.small-note {
    font-size: 0.88rem;
    color: #9aa0a6;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# PROJECT DETAILS FOR CREDITS
# --------------------------------------------------
STUDENT_NAME = "Aayush Chetan Gore"
COLLEGE_NAME = "Thakur College of Engineering and Technology"
SUBJECT_NAME = "BAI (Basics of Artificial Intelligence)"

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------
def file_exists(path):
    return os.path.exists(path)

def check_required_files():
    required_files = [
        "data/student_marks.csv",
        "data/placement_data.csv",
        "models/marks_model.pkl",
        "models/marks_scaler.pkl",
        "models/placement_model.pkl",
        "models/placement_scaler.pkl",
        "models/marks_metrics.json",
        "models/placement_metrics.json",
    ]
    missing_files = [file for file in required_files if not file_exists(file)]
    return missing_files

def show_missing_files_error(missing_files):
    st.error("❌ Some required files are missing.")
    st.markdown("### Missing Files")
    for file in missing_files:
        st.markdown(f"- `{file}`")

    st.markdown("### How to Fix")
    st.code(
        "python clean_real_data.py\n"
        "python train_models.py\n"
        "streamlit run app.py",
        language="bash"
    )
    st.info("If you are using generated sample data instead of real data, run:")
    st.code(
        "python generate_datasets.py\n"
        "python train_models.py\n"
        "streamlit run app.py",
        language="bash"
    )
    st.stop()

# --------------------------------------------------
# LOAD MODELS / DATA
# --------------------------------------------------
@st.cache_resource
def load_marks_model():
    model = joblib.load("models/marks_model.pkl")
    scaler = joblib.load("models/marks_scaler.pkl")
    return model, scaler

@st.cache_resource
def load_placement_model():
    model = joblib.load("models/placement_model.pkl")
    scaler = joblib.load("models/placement_scaler.pkl")
    return model, scaler

@st.cache_data
def load_datasets():
    marks_df = pd.read_csv("data/student_marks.csv")
    placement_df = pd.read_csv("data/placement_data.csv")
    return marks_df, placement_df

@st.cache_data
def load_metrics():
    with open("models/marks_metrics.json", "r") as f:
        marks_metrics = json.load(f)
    with open("models/placement_metrics.json", "r") as f:
        placement_metrics = json.load(f)
    return marks_metrics, placement_metrics

# --------------------------------------------------
# INITIAL FILE CHECK
# --------------------------------------------------
missing_files = check_required_files()
if missing_files:
    show_missing_files_error(missing_files)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
def render_sidebar():
    st.sidebar.image("https://img.icons8.com/color/96/graduation-cap.png", width=80)
    st.sidebar.title("🎓 Navigation")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Go to:",
        [
            "🏠 Home",
            "📊 Marks Prediction",
            "💼 Placement Prediction",
            "💡 Recommendations",
            "📈 Dashboard",
            "ℹ️ About Project"
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Mini Project**\n\n"
        "Academic Performance & Placement Prediction System\n\n"
        f"**Subject:** {SUBJECT_NAME}\n\n"
        "**Algorithms Used:**\n"
        "- Linear Regression\n"
        "- Logistic Regression\n\n"
        "**Domain:** Education + Machine Learning"
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Developed by: {STUDENT_NAME}")
    st.sidebar.caption(f"College: {COLLEGE_NAME}")

    return page

# --------------------------------------------------
# PAGE 1: HOME
# --------------------------------------------------
def render_home():
    st.markdown('<div class="main-title">🎓 Academic Performance & Placement Prediction System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">A beginner-friendly machine learning mini project to predict student marks and placement outcomes</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    This integrated system helps students and colleges estimate:
    - **Expected academic marks**
    - **Placement readiness**
    - **Improvement recommendations**
    - **Skill suggestions for better career outcomes**
    """)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📚 Marks Prediction")
        st.write(
            "Predicts expected marks using **Linear Regression** based on study hours, attendance, and previous exam score."
        )

    with col2:
        st.markdown("### 💼 Placement Prediction")
        st.write(
            "Predicts placement status using **Logistic Regression** based on CGPA, technical skills, projects, and communication score."
        )

    with col3:
        st.markdown("### 💡 Recommendations")
        st.write(
            "Provides easy-to-understand, rule-based suggestions to improve both academic and placement performance."
        )

    st.markdown("---")
    st.markdown("### 🎯 Project Objective")
    st.write(
        "The main objective of this project is to help students understand their academic and placement status early, "
        "so that they can take corrective action in time. It also helps college administrators view summary insights through a dashboard."
    )

    st.markdown("### 🛠️ How to Use This App")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        **Step 1:** Open **Marks Prediction**  
        Enter study hours, attendance, and previous score.

        **Step 2:** Open **Placement Prediction**  
        Enter CGPA, technical skills, projects, and communication score.
        """)

    with c2:
        st.markdown("""
        **Step 3:** Open **Recommendations**  
        View personalized improvement suggestions.

        **Step 4:** Open **Dashboard**  
        Explore dataset insights, charts, and model performance metrics.
        """)

    st.markdown("---")
    st.success("✅ Models and datasets are loaded successfully. Use the sidebar to explore the modules.")

# --------------------------------------------------
# PAGE 2: MARKS PREDICTION
# --------------------------------------------------
def render_marks_prediction():
    st.markdown("## 📊 Marks Prediction Module")
    st.markdown("**Algorithm Used:** Linear Regression")
    st.caption("This model predicts continuous marks in the range 0–100.")

    model, scaler = load_marks_model()
    marks_metrics, _ = load_metrics()

    st.markdown("---")
    st.markdown("### Enter Student Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        study_hours = st.slider(
            "📚 Daily Study Hours",
            min_value=0.5, max_value=12.0, value=5.0, step=0.5,
            help="Recommended range: 6 to 8 hours per day"
        )

    with col2:
        attendance = st.slider(
            "🏫 Attendance Percentage (%)",
            min_value=30.0, max_value=100.0, value=75.0, step=1.0,
            help="Recommended range: 80% to 100%"
        )

    with col3:
        previous_score = st.slider(
            "📝 Previous Exam Score",
            min_value=0.0, max_value=100.0, value=60.0, step=1.0,
            help="Recommended range: 65 to 100"
        )

    st.markdown("---")

    if st.button("🔮 Predict Expected Marks", use_container_width=True):
        input_data = np.array([[study_hours, attendance, previous_score]])
        input_scaled = scaler.transform(input_data)
        predicted_marks = model.predict(input_scaled)[0]
        predicted_marks = np.clip(predicted_marks, 0, 100)

        st.session_state["marks_inputs"] = {
            "study_hours": study_hours,
            "attendance": attendance,
            "previous_score": previous_score,
            "predicted_marks": predicted_marks
        }

        st.markdown("### 🎯 Prediction Result")
        m1, m2, m3 = st.columns(3)
        m1.metric("Study Hours", f"{study_hours:.1f} hrs/day")
        m2.metric("Attendance", f"{attendance:.1f}%")
        m3.metric("Previous Score", f"{previous_score:.1f}")

        if predicted_marks >= 75:
            st.success(f"🌟 Predicted Marks: **{predicted_marks:.1f} / 100** — Excellent performance.")
        elif predicted_marks >= 60:
            st.info(f"👍 Predicted Marks: **{predicted_marks:.1f} / 100** — Good performance.")
        elif predicted_marks >= 40:
            st.warning(f"⚠️ Predicted Marks: **{predicted_marks:.1f} / 100** — Average performance, improvement needed.")
        else:
            st.error(f"❌ Predicted Marks: **{predicted_marks:.1f} / 100** — Needs urgent improvement.")

        st.info(
            f"Model Used: Linear Regression | "
            f"MAE: {marks_metrics.get('mae', 'N/A')} | "
            f"RMSE: {marks_metrics.get('rmse', 'N/A')} | "
            f"R² Score: {marks_metrics.get('r2', 'N/A')}"
        )

        st.markdown("### 📊 Your Input vs Ideal Ranges")
        categories = [
            "Study Hours\n(ideal: 6–8)",
            "Attendance\n(ideal: 80–100)",
            "Previous Score\n(ideal: 65–100)"
        ]
        your_values = [study_hours, attendance, previous_score]
        ideal_values = [7.0, 85.0, 70.0]

        fig, ax = plt.subplots(figsize=(7, 3.8))
        x = np.arange(len(categories))
        bars1 = ax.bar(x - 0.2, your_values, 0.35, label="Your Values", color="#2F80ED")
        bars2 = ax.bar(x + 0.2, ideal_values, 0.35, label="Ideal Values", color="#6FCF97")

        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_title("Your Input vs Ideal Ranges")
        ax.legend(fontsize=8)

        for bar in bars1:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.7,
                f"{bar.get_height():.1f}",
                ha="center",
                fontsize=8
            )

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.info("💡 Visit the Recommendations page for personalized academic improvement tips.")

# --------------------------------------------------
# PAGE 3: PLACEMENT PREDICTION
# --------------------------------------------------
def render_placement_prediction():
    st.markdown("## 💼 Placement Prediction Module")
    st.markdown("**Algorithm Used:** Logistic Regression")
    st.caption("This model predicts whether a student is likely to be placed or not placed.")

    model, scaler = load_placement_model()
    _, placement_metrics = load_metrics()

    st.markdown("---")
    st.markdown("### Enter Student Profile Details")

    col1, col2 = st.columns(2)

    with col1:
        cgpa = st.slider(
            "🎓 CGPA",
            min_value=4.0, max_value=10.0, value=7.0, step=0.1,
            help="Recommended range: 7.0 to 10.0"
        )
        skills = st.slider(
            "💻 Number of Technical Skills",
            min_value=0, max_value=15, value=4,
            help="Examples: Python, Java, SQL, DSA, Web Development"
        )

    with col2:
        projects = st.slider(
            "🔨 Number of Projects Completed",
            min_value=0, max_value=15, value=2,
            help="Recommended minimum: 2 to 3 projects"
        )
        communication_score = st.slider(
            "🗣️ Communication Score (1–10)",
            min_value=1.0, max_value=10.0, value=6.0, step=0.5,
            help="Recommended range: 7 to 10"
        )

    st.markdown("---")

    if st.button("🔮 Predict Placement Status", use_container_width=True):
        input_data = np.array([[cgpa, skills, projects, communication_score]])
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        st.session_state["placement_inputs"] = {
            "cgpa": cgpa,
            "skills": skills,
            "projects": projects,
            "communication_score": communication_score,
            "prediction": prediction,
            "probability": probability
        }

        st.markdown("### 🎯 Prediction Result")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("CGPA", f"{cgpa:.1f}")
        p2.metric("Skills", f"{skills}")
        p3.metric("Projects", f"{projects}")
        p4.metric("Communication", f"{communication_score:.1f}/10")

        if prediction == 1:
            st.success(
                f"✅ Prediction: **PLACED** | Probability: **{probability:.1%}** | "
                f"Confidence Level: **{placement_probability_label(probability)}**"
            )
        else:
            st.error(
                f"❌ Prediction: **NOT PLACED** | Probability: **{probability:.1%}** | "
                f"Confidence Level: **{placement_probability_label(probability)}**"
            )

        st.info(
            f"Model Used: Logistic Regression | "
            f"Accuracy: {placement_metrics.get('accuracy', 'N/A')}% | "
            f"Precision: {placement_metrics.get('precision', 'N/A')} | "
            f"Recall: {placement_metrics.get('recall', 'N/A')} | "
            f"F1 Score: {placement_metrics.get('f1', 'N/A')}"
        )

        st.markdown("### 📊 Placement Probability Meter")
        st.progress(int(probability * 100))
        st.caption(f"Current probability of placement: {probability:.1%}")

        st.markdown("### 📊 Your Profile Overview")
        categories = ["CGPA\n(/10)", "Skills\n(/10)", "Projects\n(/10)", "Communication\n(/10)"]
        values = [cgpa, min(skills, 10), min(projects, 10), communication_score]

        fig, ax = plt.subplots(figsize=(7, 3.8))
        bars = ax.barh(
            categories,
            values,
            color=["#2F80ED", "#27AE60", "#F2C94C", "#EB5757"],
            alpha=0.9
        )
        ax.set_xlim(0, 10)
        ax.set_title("Student Profile Scores (normalized to /10)")
        ax.axvline(x=7, color="#6FCF97", linestyle="--", linewidth=2, label="Target: 7+")
        ax.legend(fontsize=8)

        for bar in bars:
            ax.text(
                bar.get_width() + 0.1,
                bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.1f}",
                va="center",
                fontsize=9
            )

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.info("💡 Visit the Recommendations page for placement improvement suggestions.")

# --------------------------------------------------
# PAGE 4: RECOMMENDATIONS
# --------------------------------------------------
def render_recommendations():
    st.markdown("## 💡 Recommendations & Improvement Tips")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs([
        "📚 Marks Improvement",
        "💼 Placement Improvement",
        "🛠️ Skill Suggestions"
    ])

    with tab1:
        st.markdown("### 📚 Marks Improvement Recommendations")
        if "marks_inputs" in st.session_state:
            d = st.session_state["marks_inputs"]
            st.info(
                f"Based on your inputs: Study Hours = {d['study_hours']:.1f}, "
                f"Attendance = {d['attendance']:.1f}%, "
                f"Previous Score = {d['previous_score']:.1f}, "
                f"Predicted Marks = {d['predicted_marks']:.1f}"
            )
            recs = get_marks_recommendations(
                d["study_hours"],
                d["attendance"],
                d["previous_score"],
                d["predicted_marks"]
            )
            for rec in recs:
                st.markdown(f"- {rec}")
        else:
            st.warning("Please generate a marks prediction first to view personalized recommendations.")

    with tab2:
        st.markdown("### 💼 Placement Improvement Recommendations")
        if "placement_inputs" in st.session_state:
            d = st.session_state["placement_inputs"]
            st.info(
                f"Based on your profile: CGPA = {d['cgpa']:.1f}, "
                f"Skills = {d['skills']}, "
                f"Projects = {d['projects']}, "
                f"Communication = {d['communication_score']:.1f}/10, "
                f"Placement Probability = {d['probability']:.1%}"
            )
            recs = get_placement_recommendations(
                d["cgpa"],
                d["skills"],
                d["projects"],
                d["communication_score"],
                d["prediction"],
                d["probability"]
            )
            for rec in recs:
                st.markdown(f"- {rec}")
        else:
            st.warning("Please generate a placement prediction first to view personalized recommendations.")

    with tab3:
        st.markdown("### 🛠️ Skill Suggestions for Better Placement")
        skill_count = st.slider("How many technical skills do you currently know?", 0, 15, 3)
        suggestions = get_skill_suggestions(skill_count)

        for category, skill_list in suggestions.items():
            st.markdown(f"**{category}**")
            cols = st.columns(3)
            for i, skill in enumerate(skill_list):
                cols[i % 3].markdown(f"- {skill}")
            st.markdown("")

        st.markdown("---")
        st.markdown("### 📚 Full Skill Catalog")
        for cat, skills in SKILL_CATALOG.items():
            with st.expander(f"📂 {cat}"):
                for skill in skills:
                    st.markdown(f"- {skill}")

# --------------------------------------------------
# PAGE 5: DASHBOARD
# --------------------------------------------------
def render_dashboard():
    st.markdown("## 📈 College Dashboard — Overview & Model Metrics")
    st.markdown("---")

    marks_df, placement_df = load_datasets()
    marks_metrics, placement_metrics = load_metrics()

    st.markdown("### 📊 Summary Statistics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Students (Marks)", len(marks_df))
    c2.metric("Total Students (Placement)", len(placement_df))
    c3.metric("Likely Placed", int(placement_df["placed"].sum()))
    c4.metric("Likely Not Placed", int((placement_df["placed"] == 0).sum()))

    st.markdown("---")
    st.markdown("### 📊 Data Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(marks_df["marks"], bins=20, color="#2F80ED", edgecolor="white", alpha=0.9)
        ax.axvline(
            marks_df["marks"].mean(),
            color="#EB5757",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {marks_df['marks'].mean():.1f}"
        )
        ax.set_title("Distribution of Student Marks")
        ax.set_xlabel("Marks")
        ax.set_ylabel("Number of Students")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        placed_count = int(placement_df["placed"].sum())
        not_placed_count = int((placement_df["placed"] == 0).sum())

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(
            [placed_count, not_placed_count],
            labels=["Placed", "Not Placed"],
            autopct="%1.1f%%",
            colors=["#27AE60", "#EB5757"],
            startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2}
        )
        ax.set_title("Placement Distribution")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    col3, col4 = st.columns(2)

    with col3:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(
            marks_df["study_hours"],
            marks_df["marks"],
            color="#2F80ED",
            alpha=0.5,
            edgecolors="white",
            s=40
        )
        ax.set_title("Study Hours vs Marks")
        ax.set_xlabel("Study Hours")
        ax.set_ylabel("Marks")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col4:
        placed_cgpa = placement_df[placement_df["placed"] == 1]["cgpa"]
        not_placed_cgpa = placement_df[placement_df["placed"] == 0]["cgpa"]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot(
            [not_placed_cgpa, placed_cgpa],
            tick_labels=["Not Placed", "Placed"],
            patch_artist=True,
            boxprops=dict(facecolor="#D6EAF8", color="#2F80ED"),
            medianprops=dict(color="#EB5757", linewidth=2)
        )
        ax.set_title("CGPA Distribution by Placement Status")
        ax.set_ylabel("CGPA")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.markdown("### 🤖 Model Performance Metrics")

    mcol1, mcol2 = st.columns(2)

    with mcol1:
        st.markdown("#### Marks Prediction Model")
        a1, a2 = st.columns(2)
        a1.metric("MAE", marks_metrics.get("mae", "N/A"))
        a2.metric("RMSE", marks_metrics.get("rmse", "N/A"))
        a3, a4 = st.columns(2)
        a3.metric("MSE", marks_metrics.get("mse", "N/A"))
        a4.metric("R² Score", marks_metrics.get("r2", "N/A"))

    with mcol2:
        st.markdown("#### Placement Prediction Model")
        b1, b2 = st.columns(2)
        b1.metric("Accuracy", f"{placement_metrics.get('accuracy', 'N/A')}%")
        b2.metric("Precision", placement_metrics.get("precision", "N/A"))
        b3, b4 = st.columns(2)
        b3.metric("Recall", placement_metrics.get("recall", "N/A"))
        b4.metric("F1 Score", placement_metrics.get("f1", "N/A"))

        if "confusion_matrix" in placement_metrics:
            cm = np.array(placement_metrics["confusion_matrix"])
            fig, ax = plt.subplots(figsize=(4.5, 3.2))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Not Placed", "Placed"],
                yticklabels=["Not Placed", "Placed"],
                ax=ax
            )
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.markdown("---")
    st.markdown("### 🗂️ Dataset Preview")

    tab1, tab2 = st.tabs(["📚 Student Marks Dataset", "💼 Placement Dataset"])

    with tab1:
        st.dataframe(marks_df.head(10), use_container_width=True)
        st.caption(
            f"Records: {len(marks_df)} | "
            f"Average Marks: {marks_df['marks'].mean():.1f}"
        )

    with tab2:
        st.dataframe(placement_df.head(10), use_container_width=True)
        st.caption(
            f"Records: {len(placement_df)} | "
            f"Placement Rate: {placement_df['placed'].mean() * 100:.1f}%"
        )

# --------------------------------------------------
# PAGE 6: ABOUT PROJECT
# --------------------------------------------------
def render_about():
    st.markdown("## ℹ️ About This Project")
    st.markdown("---")

    st.markdown(f"""
### 🎓 Project Title
**Academic Performance & Placement Prediction System using Machine Learning**

### 👨‍🎓 Student Information
- **Developed by:** {STUDENT_NAME}
- **College:** {COLLEGE_NAME}
- **Subject:** {SUBJECT_NAME}
- **Project Type:** Mini Project

---

### 📌 Problem Statement
Students and institutions often do not have an early method to estimate academic performance and placement readiness.  
This project solves that problem by using Machine Learning models to make predictions based on simple educational and profile-related inputs.

---

### 🎯 Objectives
- Predict student marks using study-related inputs
- Predict placement status using placement-related inputs
- Provide recommendation-based improvement suggestions
- Suggest useful technical and soft skills for placement readiness
- Display summary insights in a dashboard for student/college use

---

### ⚙️ Algorithms Used

#### 1. Linear Regression
Used for **Marks Prediction** because marks are continuous numeric values.

#### 2. Logistic Regression
Used for **Placement Prediction** because placement status is a binary outcome:
- 1 = Placed
- 0 = Not Placed

---

### 🧠 How Recommendation Logic Works
The recommendation module is **rule-based**.
It checks the entered values and predicted result, then gives easy-to-understand suggestions such as:
- increase study hours
- improve attendance
- build more projects
- learn more technical skills
- improve communication and interview preparation

---

### 🛠️ Technologies Used
- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

---

### 🔮 Future Scope
- Add real-time database integration
- Include more student features such as internships and certifications
- Compare multiple machine learning algorithms
- Add downloadable student reports
- Deploy for wider institutional use
""")

# --------------------------------------------------
# MAIN ROUTER
# --------------------------------------------------
def main():
    page = render_sidebar()

    if page == "🏠 Home":
        render_home()
    elif page == "📊 Marks Prediction":
        render_marks_prediction()
    elif page == "💼 Placement Prediction":
        render_placement_prediction()
    elif page == "💡 Recommendations":
        render_recommendations()
    elif page == "📈 Dashboard":
        render_dashboard()
    elif page == "ℹ️ About Project":
        render_about()

    st.markdown("---")
    st.markdown(
        f'<div class="footer-text">Mini Project | {SUBJECT_NAME} | Developed by {STUDENT_NAME}</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
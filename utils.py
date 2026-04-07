# =============================================================
# utils.py
# Purpose: Utility functions for recommendations and skill suggestions
# =============================================================

def get_marks_recommendations(study_hours, attendance, previous_score, predicted_marks):
    recommendations = []

    if study_hours < 3:
        recommendations.append("📚 Increase your daily study hours to at least 4–5 hours for better improvement.")
    elif study_hours < 5:
        recommendations.append("📚 Try to study at least 6 hours daily to improve consistency and understanding.")
    else:
        recommendations.append("✅ Your study hours are in a good range. Maintain consistency and avoid distractions.")

    if attendance < 60:
        recommendations.append("🏫 Your attendance is very low. Try to maintain at least 75% attendance regularly.")
    elif attendance < 75:
        recommendations.append("🏫 Improve your attendance and aim for 80% or above for better classroom understanding.")
    else:
        recommendations.append("✅ Your attendance is good. Regular class presence supports better academic performance.")

    if previous_score < 40:
        recommendations.append("📝 Revise previous weak topics thoroughly and focus more on conceptual understanding.")
        recommendations.append("🧪 Take regular mock tests and solve practice papers to track your progress.")
    elif previous_score < 60:
        recommendations.append("📝 Practice more questions daily and strengthen the topics where you lost marks earlier.")
    else:
        recommendations.append("✅ Your previous score is a good foundation. Build on it with regular revision.")

    if predicted_marks < 40:
        recommendations.append("⚠️ Your predicted marks are low. You should follow a strict study timetable and seek help from teachers if needed.")
    elif predicted_marks < 60:
        recommendations.append("📈 Your predicted marks are average. With focused revision, you can improve significantly.")
    elif predicted_marks < 75:
        recommendations.append("👍 Your predicted marks are good. Focus on accuracy and time management to score even higher.")
    else:
        recommendations.append("🌟 Your predicted marks are excellent. Keep up the same effort and consistency.")

    recommendations.append("💡 Tip: Use active recall, short notes, and self-testing instead of only passive reading.")
    return recommendations


def get_placement_recommendations(cgpa, skills, projects, communication_score, prediction, probability):
    recommendations = []

    if cgpa < 6.0:
        recommendations.append("📊 Your CGPA is currently low. Focus on improving semester performance because many companies prefer 6.5+.")
    elif cgpa < 7.5:
        recommendations.append("📊 Try to improve your CGPA above 7.5 to become eligible for more placement opportunities.")
    else:
        recommendations.append("✅ Your CGPA is in a strong range. Maintain it consistently.")

    if skills < 3:
        recommendations.append("💻 You need more technical skills. Start with important placement skills like Python, SQL, and Java.")
    elif skills < 5:
        recommendations.append("💻 Add 2–3 more technical skills to strengthen your placement profile.")
    else:
        recommendations.append("✅ Your technical skill count is good. Focus on improving depth in your strongest skills.")

    if projects < 2:
        recommendations.append("🔨 Build at least 2–3 meaningful projects to demonstrate practical knowledge.")
        recommendations.append("🌐 Try uploading your projects to GitHub so they can be shown in resumes and interviews.")
    elif projects < 4:
        recommendations.append("🔨 You have a decent start with projects. Add one more advanced or domain-based project.")
    else:
        recommendations.append("✅ Your project portfolio is strong. Ensure your projects are clearly documented and explained.")

    if communication_score < 5:
        recommendations.append("🗣️ Communication needs strong improvement. Practice speaking, mock interviews, and group discussions regularly.")
    elif communication_score < 7:
        recommendations.append("🗣️ Work on improving communication and confidence for HR and technical interview rounds.")
    else:
        recommendations.append("✅ Your communication score is good. Keep practicing interview answers and self-introduction.")

    if prediction == 1:
        recommendations.append(f"🎉 You are likely to be placed with a predicted probability of {probability:.0%}. Keep preparing and apply confidently.")
    else:
        recommendations.append(f"⚠️ Your current placement probability is lower at around {probability:.0%}. Follow the above suggestions to improve your chances.")
        recommendations.append("📋 Practice aptitude tests, coding rounds, resume preparation, and mock interviews regularly.")

    recommendations.append("💡 Tip: Keep your resume, LinkedIn, and GitHub updated for better placement visibility.")
    return recommendations


SKILL_CATALOG = {
    "Programming Languages": [
        "Python 🐍", "Java ☕", "C++ 💻", "JavaScript 🌐"
    ],
    "Database & Query": [
        "SQL 🗄️", "MySQL", "MongoDB 🍃"
    ],
    "Core CS Fundamentals": [
        "Data Structures & Algorithms 🔢", "Operating Systems", "Computer Networks"
    ],
    "Web & App Development": [
        "HTML/CSS", "React.js ⚛️", "Node.js", "Django/Flask"
    ],
    "Data & AI": [
        "Machine Learning Basics 🤖", "Data Analysis with Pandas", "Power BI / Tableau 📊"
    ],
    "Soft Skills": [
        "Communication Skills 🗣️", "Aptitude & Reasoning 🧠",
        "Resume Writing ✍️", "Interview Preparation 🎯"
    ],
    "Tools & Platforms": [
        "Git & GitHub 🐙", "Linux Basics 🐧", "MS Excel 📋"
    ]
}


def get_skill_suggestions(current_skill_count):
    if current_skill_count < 3:
        return {
            "🚀 Start Here (Essential Skills)": [
                "Python 🐍", "SQL 🗄️", "Data Structures & Algorithms 🔢",
                "Communication Skills 🗣️", "Aptitude & Reasoning 🧠", "Git & GitHub 🐙"
            ],
            "📌 Add Next": [
                "Java ☕", "HTML/CSS", "MS Excel 📋", "Resume Writing ✍️"
            ]
        }
    elif current_skill_count < 6:
        return {
            "📈 Recommended to Add Now": [
                "Machine Learning Basics 🤖",
                "React.js ⚛️",
                "MySQL",
                "Interview Preparation 🎯",
                "Power BI / Tableau 📊"
            ],
            "🔧 Good to Know": [
                "Linux Basics 🐧",
                "Django/Flask",
                "MongoDB 🍃",
                "Computer Networks",
                "Node.js"
            ]
        }
    else:
        return {
            "🌟 Level Up Your Profile": [
                "Cloud Basics (AWS/GCP/Azure)",
                "System Design",
                "Data Analysis with Pandas",
                "Advanced SQL",
                "API Development"
            ],
            "🏆 Stand Out With": [
                "Open Source Contributions",
                "Competitive Programming 🏆",
                "Hackathon Projects",
                "Technical Blogging"
            ]
        }


def placement_probability_label(probability):
    if probability >= 0.80:
        return "Very High 🟢"
    elif probability >= 0.60:
        return "High 🟡"
    elif probability >= 0.40:
        return "Moderate 🟠"
    else:
        return "Low 🔴"
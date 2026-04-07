# Academic Performance & Placement Prediction System
## Mini Project Report — BAI Subject

---

## 1. Project Abstract

The **Academic Performance and Placement Prediction System** is a Machine Learning-based mini project developed using Python and Streamlit. The system integrates two prediction modules: one for predicting student marks using **Linear Regression** and another for predicting student placement outcomes using **Logistic Regression**. The application also features a rule-based recommendation engine that provides personalized suggestions for academic and career improvement, along with a simple dashboard for college/admin use. The entire system is built to be beginner-friendly, locally deployable, and suitable for a college-level AI/ML mini project presentation.

---

## 2. Problem Statement

Students often lack awareness of how their current study habits and academic profile will affect their future performance — both in exams and in campus placements. College administrators similarly struggle to identify at-risk students early. This project addresses both problems by building a simple, interactive machine learning system that:
- Predicts a student's expected exam marks based on study hours, attendance, and previous scores
- Predicts whether a student is likely to be placed based on CGPA, skills, projects, and communication
- Provides actionable recommendations to help students improve

---

## 3. Objectives

1. To build a marks prediction module using Linear Regression
2. To build a placement prediction module using Logistic Regression
3. To provide rule-based recommendations for academic and placement improvement
4. To suggest relevant technical skills for better employability
5. To create a simple dashboard for college/admin overview
6. To develop a clean and interactive UI using Streamlit
7. To keep the project beginner-friendly and easy to explain in a viva

---

## 4. Modules of the Project

### Module 1: Dataset Generation (`generate_datasets.py`)
- Generates realistic synthetic datasets for both prediction tasks
- `student_marks.csv` — contains study_hours, attendance, previous_score, marks
- `placement_data.csv` — contains cgpa, skills, projects, communication_score, placed
- Uses NumPy's random functions to simulate real-world student data

### Module 2: Model Training (`train_models.py`)
- Loads datasets, preprocesses features using StandardScaler
- Trains Linear Regression model for marks prediction
- Trains Logistic Regression model for placement prediction
- Evaluates models and saves metrics to JSON
- Saves trained models as `.pkl` files using joblib

### Module 3: Marks Prediction (Streamlit Page)
- Accepts study_hours, attendance, previous_score as inputs
- Loads saved Linear Regression model and scaler
- Outputs predicted marks with color-coded feedback
- Displays comparison chart (your values vs ideal values)

### Module 4: Placement Prediction (Streamlit Page)
- Accepts CGPA, skills, projects, communication_score as inputs
- Loads saved Logistic Regression model and scaler
- Predicts placement outcome (Placed / Not Placed)
- Shows placement probability with a progress bar

### Module 5: Recommendations (`utils.py`)
- Rule-based recommendation engine for marks improvement
- Rule-based recommendation engine for placement improvement
- Curated skill suggestion catalog based on student's skill count

### Module 6: Dashboard
- Displays dataset statistics and charts
- Visualizes marks distribution, placement ratio, CGPA comparisons
- Shows model evaluation metrics (MAE, RMSE, R², Accuracy, F1)
- Provides dataset preview tables

---

## 5. Algorithm Explanation

### Linear Regression (Marks Prediction)
Linear Regression is a supervised learning algorithm used to predict continuous numeric values. It assumes a linear relationship between input features and the target variable.

**Mathematical Formula:**
```
y = β0 + β1·x1 + β2·x2 + β3·x3
```
Where:
- y = predicted marks
- x1 = study_hours, x2 = attendance, x3 = previous_score
- β0 = intercept (bias), β1, β2, β3 = learned coefficients

**Why it's suitable here:**
Marks is a continuous value ranging from 0 to 100. Linear Regression learns the weights for each input feature and produces a numerical output — perfect for regression tasks.

**Evaluation Metrics:**
- **MAE (Mean Absolute Error):** Average absolute difference between actual and predicted marks
- **MSE (Mean Squared Error):** Average of squared differences (penalizes large errors more)
- **RMSE (Root MSE):** Square root of MSE — same unit as marks, easier to interpret
- **R² Score:** Proportion of variance in marks explained by the model (closer to 1 = better)

---

### Logistic Regression (Placement Prediction)
Logistic Regression is a supervised learning algorithm for binary classification. Despite its name, it is a classification algorithm, not a regression one. It uses the **sigmoid function** to output a probability between 0 and 1.

**Mathematical Formula:**
```
P(placed = 1) = 1 / (1 + e^(-z))
z = β0 + β1·cgpa + β2·skills + β3·projects + β4·communication_score
```

**Decision Rule:**
- If P ≥ 0.5 → Predict "Placed" (1)
- If P < 0.5 → Predict "Not Placed" (0)

**Why it's suitable here:**
Placement is a binary outcome (yes/no). Logistic Regression handles binary classification problems well and also provides an interpretable probability score.

**Evaluation Metrics:**
- **Accuracy:** Percentage of correctly classified students
- **Precision:** Of students predicted as placed, how many actually were?
- **Recall:** Of students who were actually placed, how many did we catch?
- **F1 Score:** Harmonic mean of Precision and Recall — best for imbalanced classes
- **Confusion Matrix:** Table showing True Positives, False Positives, True Negatives, False Negatives

---

## 6. Future Scope

1. **Real Database Integration:** Connect to a MySQL or MongoDB database for live student records
2. **More Features:** Include internships, certifications, backlogs, extracurriculars
3. **Advanced Models:** Compare performance with Random Forest, SVM, or XGBoost
4. **Deep Learning Extension:** Use neural networks for improved accuracy on larger datasets
5. **Student Login Portal:** Individual student login to track progress over time
6. **Automated Reports:** Generate downloadable PDF reports per student
7. **Cloud Deployment:** Deploy on Streamlit Cloud, Heroku, or AWS for institutional use
8. **Real Dataset Training:** Train on actual campus placement and academic records

---

## 7. How This Project Relates to AI/ML

This project demonstrates several core concepts of Artificial Intelligence and Machine Learning:

| Concept | How It's Used |
|---------|---------------|
| Supervised Learning | Both Linear and Logistic Regression are supervised algorithms trained on labeled data |
| Feature Engineering | Selecting relevant features (study hours, CGPA, etc.) that affect the target |
| Model Training | Fitting the model on training data to learn patterns |
| Model Evaluation | Using metrics like R², Accuracy, F1 to measure model quality |
| Prediction | Making real-time predictions based on new user inputs |
| Data Preprocessing | StandardScaler normalizes features for better model performance |
| Train-Test Split | Splitting data to evaluate model on unseen data and avoid overfitting |
| Classification vs Regression | Understanding when to use each type of model |
| Rule-Based AI | The recommendation engine uses IF-THEN rules — a classic AI approach |

---

## 8. Viva Questions and Answers

**Q1. What is the difference between Linear Regression and Logistic Regression?**
> Linear Regression predicts continuous numeric values (like marks). Logistic Regression predicts categorical/binary outcomes (like placed/not placed). Logistic Regression uses a sigmoid function to output probabilities.

**Q2. Why did you use Linear Regression for marks prediction?**
> Because marks is a continuous variable (0–100). Linear Regression finds the best linear relationship between the input features and the target value, which is appropriate for this case.

**Q3. Why did you use Logistic Regression for placement prediction?**
> Because placement is a binary outcome (placed = 1 or not placed = 0). Logistic Regression is designed for binary classification and also provides a probability score, making the result more interpretable.

**Q4. What is StandardScaler and why is it used?**
> StandardScaler normalizes features to have a mean of 0 and standard deviation of 1. This is important because features like CGPA (0–10) and communication score (1–10) have different ranges. Scaling ensures no single feature dominates the model.

**Q5. What is a Train-Test Split?**
> It's dividing the dataset into two parts — one for training (80%) and one for testing (20%). The model learns from training data and is evaluated on unseen test data to check if it generalizes well.

**Q6. What is R² Score?**
> R² (R-squared) measures how well the model explains the variance in the target variable. An R² of 0.85 means the model explains 85% of the variation in marks — the closer to 1, the better.

**Q7. What is a Confusion Matrix?**
> A confusion matrix shows the count of True Positives (correctly predicted placed), True Negatives (correctly predicted not placed), False Positives, and False Negatives. It helps evaluate the quality of a classification model.

**Q8. What is F1 Score?**
> F1 Score is the harmonic mean of Precision and Recall. It is useful when the dataset is imbalanced (more placed than not-placed, or vice versa). It balances both false positives and false negatives.

**Q9. What is overfitting? How is it avoided here?**
> Overfitting is when a model performs well on training data but poorly on new data. It's avoided here by using a train-test split (testing on unseen data) and using simple models (Linear and Logistic Regression) that are less prone to overfitting.

**Q10. What is the sigmoid function?**
> The sigmoid function converts any real number into a value between 0 and 1. Formula: `f(x) = 1 / (1 + e^(-x))`. In Logistic Regression, it converts the weighted sum of inputs into a probability.

**Q11. How does the recommendation engine work?**
> It uses simple IF-THEN rules. For example: if study_hours < 3, suggest increasing study time. If CGPA < 6.0, suggest improving grades. It's a classic rule-based AI approach — simple but effective for this use case.

**Q12. What is Streamlit?**
> Streamlit is an open-source Python framework for building interactive web applications without needing HTML, CSS, or JavaScript knowledge. It converts Python scripts into shareable web apps with minimal code.

**Q13. What are the datasets used in this project?**
> Two synthetic (programmatically generated) datasets are used:
> 1. `student_marks.csv` — 300 records with study_hours, attendance, previous_score, marks
> 2. `placement_data.csv` — 300 records with cgpa, skills, projects, communication_score, placed

**Q14. Can this project be extended to use real data?**
> Yes. The CSV files can be replaced with real student data. The model training and app code would remain the same — only the data source changes.

**Q15. What is Precision vs Recall?**
> Precision = Of all students predicted as placed, how many actually were placed?
> Recall = Of all students who were actually placed, how many did the model correctly identify?
> High precision means fewer false alarms; high recall means fewer missed placements.

---

## References

1. Scikit-learn Documentation — https://scikit-learn.org/stable/
2. Streamlit Documentation — https://docs.streamlit.io/
3. Pandas Documentation — https://pandas.pydata.org/
4. "Introduction to Machine Learning with Python" — Andreas Müller & Sarah Guido
5. "Hands-On Machine Learning with Scikit-Learn and TensorFlow" — Aurélien Géron

# 💉 Diabetes Prediction Project

This machine learning project predicts whether a patient has diabetes using the **Pima Indians Diabetes Dataset**. It uses **logistic regression**, and includes data cleaning, visualization, model training, and evaluation.

---

## 📌 Overview

The goal is to build a binary classification model that can determine if a person is likely to have diabetes based on key medical indicators such as glucose level, BMI, age, etc.

---

## 📂 Dataset

- **Name**: Pima Indians Diabetes Dataset
- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Features**:
  - Pregnancies
  - Glucose
  - BloodPressure
  - SkinThickness
  - Insulin
  - BMI
  - DiabetesPedigreeFunction
  - Age
  - Outcome (Target: 1 = Diabetes, 0 = No Diabetes)

---

## 🛠️ Technologies Used

- Python
- NumPy & Pandas
- Seaborn & Matplotlib (for data visualization)
- Scikit-learn (for ML model and evaluation)
- Joblib (for model saving)

---

## 🔎 Project Highlights

- Cleaned dataset by handling missing/invalid values
- Performed Exploratory Data Analysis (EDA) using visualizations
- Built a **Logistic Regression model**
- Evaluated model with:
  - Accuracy Score
  - Confusion Matrix
  - Classification Report
- Saved trained model for future use

---

## 📈 Model Performance

- Accuracy: ~76–78%
- Evaluation metrics included: Precision, Recall, F1-score

---

## 📁 Folder Structure
diabetes_prediction/
├── main.py # Main script
├── diabetes.csv # Dataset
├── diabetes_model.pkl # Saved model (optional)
├── requirements.txt # Python packages
└── README.md # Project description




---

## 🚀 How to Run the Project

1. Clone the repo:
   ```bash
   git clone https://github.com/Ramanjot005/diabetes-prediction.git
   cd diabetes-prediction
2.python -m venv venv
venv\Scripts\activate   # On Windows

3.pip install -r requirements.txt
4.python main.py

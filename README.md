# Diabetes Risk Analytics & Predictive Modeling Platform

**AICTE | IBM SkillsBuild Data Analytics with AI Internship Project**  
**Author:** Ayush Gadkari  
**GitHub Repository:** https://github.com/ayushgadkari/diabetes-analytics-project

---

## 📌 Project Overview
An end-to-end data analytics dashboard and machine learning platform built to convert clinical measurements into early diabetes diagnostic risk intelligence using a dataset of 100,000 patient records.

## 🔗 Dataset Details
* **Dataset Name:** Diabetes Prediction Dataset
* **Source Link:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)
* **Total Records:** 100,000 patient rows
* **Key Features:** `gender`, `age`, `hypertension`, `heart_disease`, `smoking_history`, `bmi`, `HbA1c_level`, `blood_glucose_level`, `diabetes`

---

## 🔑 Key Features

* **Clinical Executive Overview:** Multi-metric tracking across patient records including prevalence rates, average HbA1c, and blood glucose metrics.
* **Biomarker Correlation Engine:** Interactive Plotly scatter plots and box plots analyzing multi-variable relationships between BMI, HbA1c, and Blood Glucose.
* **Risk Cohort Segmentation:** Instant filtering of critical patient populations based on diagnostic thresholds (HbA1c ≥ 6.5% or Blood Glucose ≥ 200 mg/dL).
* **AI Predictive Model:** Random Forest Classifier delivering real-time patient risk assessments and actionable clinical prompts.

---

## 🛠️ Tech Stack

* **Language:** Python 3.9+
* **Framework / Dashboard:** Streamlit
* **Data Processing:** Pandas, NumPy
* **Visualization Engine:** Plotly Express
* **Machine Learning:** Scikit-Learn (Random Forest Classification)

---

## 🚀 Installation & Running Instructions

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/ayushgadkari/diabetes-analytics-project.git](https://github.com/ayushgadkari/diabetes-analytics-project.git)
   cd diabetes-analytics-project
   
2. Install Required Dependencies:
   ```bash
   pip install -r requirements.txt

3. Run the Streamlit Dashboard:
   ```bash
   streamlit run AyushGadkari_DiabetesAnalytics.py

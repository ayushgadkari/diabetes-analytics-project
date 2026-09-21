import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Page Configuration
st.set_page_config(
    page_title="Diabetes Risk Analytics & Prediction Platform",
    page_icon="🩺",
    layout="wide"
)

# ---------------------------------------------------------
# 1. DATA ENGINE
# ---------------------------------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("diabetes_prediction_dataset.csv")
    except FileNotFoundError:
        df = pd.read_csv("diabetes_prediction_dataset_2.csv")
    return df

df = load_data()

# Preprocessing & Encoding
gender_map = {'Female': 0, 'Male': 1, 'Other': 2}
smoking_map = {'never': 0, 'No Info': 1, 'current': 2, 'former': 3, 'ever': 4, 'not current': 5}

df_encoded = df.copy()
df_encoded['gender_enc'] = df_encoded['gender'].map(gender_map).fillna(0).astype(float)
df_encoded['smoking_enc'] = df_encoded['smoking_history'].map(smoking_map).fillna(1).astype(float)

feature_cols = ['gender_enc', 'age', 'hypertension', 'heart_disease', 'smoking_enc', 'bmi', 'HbA1c_level', 'blood_glucose_level']

# Ensure numeric types
for col in ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']:
    df_encoded[col] = df_encoded[col].astype(float)

# ---------------------------------------------------------
# 2. SIDEBAR FILTERS
# ---------------------------------------------------------
st.sidebar.title("🔍 Filter Dataset")

gender_filter = st.sidebar.multiselect(
    "Select Gender:",
    options=list(df['gender'].unique()),
    default=list(df['gender'].unique())
)

min_age, max_age = int(df['age'].min()), int(df['age'].max())
age_range = st.sidebar.slider("Select Age Range:", min_age, max_age, (min_age, max_age))

smoking_filter = st.sidebar.multiselect(
    "Select Smoking History:",
    options=list(df['smoking_history'].unique()),
    default=list(df['smoking_history'].unique())
)

filtered_df = df[
    (df['gender'].isin(gender_filter)) &
    (df['age'].between(age_range[0], age_range[1])) &
    (df['smoking_history'].isin(smoking_filter))
]

# ---------------------------------------------------------
# 3. DASHBOARD MAIN INTERFACE
# ---------------------------------------------------------
st.title("🩺 Diabetes Analytics & AI Risk Prediction Engine")
st.markdown("Transforming clinical biomarker records into early diabetes risk identification and predictive insights.")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Clinical Overview", 
    "🧪 Biomarker Analysis", 
    "⚠️ High-Risk Segmentation", 
    "🤖 AI Risk Evaluator"
])

# TAB 1: CLINICAL OVERVIEW
with tab1:
    st.subheader("Key Performance Indicators")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_records = len(filtered_df)
    diabetic_cases = int(filtered_df['diabetes'].sum())
    prevalence = (diabetic_cases / total_records * 100) if total_records > 0 else 0.0
    avg_hba1c = float(filtered_df['HbA1c_level'].mean()) if total_records > 0 else 0.0
    avg_glucose = float(filtered_df['blood_glucose_level'].mean()) if total_records > 0 else 0.0
    
    col1.metric("Total Sample Size", f"{total_records:,}")
    col2.metric("Diabetic Patients", f"{diabetic_cases:,}")
    col3.metric("Prevalence Rate", f"{prevalence:.1f}%")
    col4.metric("Avg HbA1c Level", f"{avg_hba1c:.2f}%")
    col5.metric("Avg Blood Glucose", f"{avg_glucose:.1f} mg/dL")
    
    st.markdown("---")
    
    col_a, col_b = st.columns(2)
    with col_a:
        fig_diag = px.pie(
            filtered_df, names='diabetes',
            title="Diabetes Prevalence Distribution (0 = Non-Diabetic, 1 = Diabetic)",
            color='diabetes', color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
            hole=0.4
        )
        st.plotly_chart(fig_diag, use_container_width=True)
        
    with col_b:
        fig_age = px.histogram(
            filtered_df, x='age', color='diabetes',
            barmode='overlay', title="Age Distribution by Diabetes Diagnosis",
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
        )
        st.plotly_chart(fig_age, use_container_width=True)

# TAB 2: BIOMARKER ANALYSIS
with tab2:
    st.subheader("Clinical Biomarker Correlations")
    col_c, col_d = st.columns(2)
    
    with col_c:
        sample_size = min(2000, len(filtered_df))
        sample_df = filtered_df.sample(sample_size, random_state=42) if sample_size > 0 else filtered_df
        fig_scatter = px.scatter(
            sample_df, x='blood_glucose_level', y='HbA1c_level', color='diabetes',
            size='bmi', title="Blood Glucose vs HbA1c Level (Bubble Size = BMI)",
            color_discrete_map={0: '#3498db', 1: '#e74c3c'}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with col_d:
        fig_box = px.box(
            filtered_df, x='diabetes', y='bmi', color='diabetes',
            title="BMI Distribution Across Diagnostic Groups",
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
        )
        st.plotly_chart(fig_box, use_container_width=True)

# TAB 3: HIGH-RISK SEGMENTATION
with tab3:
    st.subheader("Critical Risk Cohort Identification")
    
    high_risk_df = filtered_df[
        (filtered_df['HbA1c_level'] >= 6.5) | 
        (filtered_df['blood_glucose_level'] >= 200)
    ].sort_values(by=['HbA1c_level', 'blood_glucose_level'], ascending=False)
    
    st.markdown(f"**High Diagnostic Risk Cohort (HbA1c ≥ 6.5% OR Blood Glucose ≥ 200 mg/dL):** Found `{len(high_risk_df):,}` patient records.")
    st.dataframe(
        high_risk_df[['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes']],
        use_container_width=True
    )

# TAB 4: AI RISK EVALUATOR
with tab4:
    st.subheader("Random Forest Predictive Risk Engine")
    st.write("Evaluate patient risk probabilities in real time using a machine learning model trained on historical clinical features.")
    
    X = df_encoded[feature_cols]
    y = df_encoded['diabetes']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    @st.cache_resource
    def train_model(X_tr, y_tr):
        clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        clf.fit(X_tr, y_tr)
        return clf

    model = train_model(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    st.success(f"Model Trained Successfully! Validation Accuracy: {acc*100:.2f}%")
    
    st.markdown("#### Patient Clinical Measurements Input")
    col_i1, col_i2, col_i3 = st.columns(3)
    
    in_gender = col_i1.selectbox("Gender", options=['Female', 'Male', 'Other'])
    in_age = col_i1.number_input("Age (Years)", min_value=1, max_value=120, value=45)
    in_hyper = col_i1.selectbox("Hypertension Status", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    in_heart = col_i2.selectbox("Heart Disease History", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    in_smoke = col_i2.selectbox("Smoking History", options=['never', 'No Info', 'current', 'former', 'ever', 'not current'])
    in_bmi = col_i2.number_input("BMI (kg/m²)", min_value=10.0, max_value=95.0, value=27.3)
    
    in_hba1c = col_i3.number_input("HbA1c Level (%)", min_value=3.5, max_value=9.0, value=5.5)
    in_glucose = col_i3.number_input("Blood Glucose Level (mg/dL)", min_value=80, max_value=300, value=130)
    
    if st.button("Calculate Diabetes Risk"):
        input_data = pd.DataFrame([[
            float(gender_map[in_gender]),
            float(in_age),
            float(in_hyper),
            float(in_heart),
            float(smoking_map[in_smoke]),
            float(in_bmi),
            float(in_hba1c),
            float(in_glucose)
        ]], columns=feature_cols)
        
        prob = float(model.predict_proba(input_data)[0][1])
        
        st.markdown("---")
        if prob >= 0.5:
            st.error(f"🚨 **High Risk Flagged!** Estimated Diabetes Probability: **{prob*100:.1f}%**")
            st.warning("**Recommended Clinical Action:** Refer patient for targeted oral glucose tolerance test (OGTT) and continuous glycemic monitoring.")
        else:
            st.success(f"✅ **Low Risk Identified.** Estimated Diabetes Probability: **{prob*100:.1f}%**")
            st.info("**Recommended Clinical Action:** Maintain standard annual metabolic wellness screening.")
import streamlit as st
import pandas as pd
import pickle

# PAGE CONFIG & UI DESIGN 
st.set_page_config(page_title="Student Placement Predictor", layout="wide")
st.title("Placement & Salary Prediction Engine")
st.markdown("Predict student employability and estimated salary package using machine learning.")

# LOAD MODELS (CACHED)
@st.cache_resource
def load_models():
    with open("models/best_classification_model.pkl", "rb") as f:
        clf_model = pickle.load(f)
    with open("models/best_regression_model.pkl", "rb") as f:
        reg_model = pickle.load(f)
    return clf_model, reg_model

clf_pipeline, reg_pipeline = load_models()

# FRONTEND: SIDEBAR INPUTS
st.sidebar.header("Input Student Data")

def user_input_features():
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    ssc_percentage = st.sidebar.slider("10th Grade Percentage (SSC)", 40.0, 100.0, 75.0)
    hsc_percentage = st.sidebar.slider("12th Grade Percentage (HSC)", 40.0, 100.0, 75.0)
    degree_percentage = st.sidebar.slider("Degree Percentage", 40.0, 100.0, 75.0)
    cgpa = st.sidebar.slider("CGPA", 4.0, 10.0, 7.5)
    entrance_exam_score = st.sidebar.slider("Entrance Exam Score", 0.0, 100.0, 70.0)
    technical_skill_score = st.sidebar.slider("Technical Skill Score", 0.0, 100.0, 80.0)
    soft_skill_score = st.sidebar.slider("Soft Skill Score", 0.0, 100.0, 80.0)
    internship_count = st.sidebar.number_input("Internship Count", 0, 5, 1)
    live_projects = st.sidebar.number_input("Live Projects", 0, 10, 2)
    work_experience_months = st.sidebar.number_input("Work Experience (Months)", 0, 60, 0)
    certifications = st.sidebar.number_input("Certifications", 0, 10, 1)
    attendance_percentage = st.sidebar.slider("Attendance Percentage", 0.0, 100.0, 85.0)
    backlogs = st.sidebar.number_input("Backlogs", 0, 10, 0)
    extracurricular_activities = st.sidebar.selectbox("Extracurricular Activities", ["Yes", "No"])

    # Create raw dataframe
    data = {
        'gender': gender, 'ssc_percentage': ssc_percentage, 'hsc_percentage': hsc_percentage,
        'degree_percentage': degree_percentage, 'cgpa': cgpa, 'entrance_exam_score': entrance_exam_score,
        'technical_skill_score': technical_skill_score, 'soft_skill_score': soft_skill_score,
        'internship_count': internship_count, 'live_projects': live_projects, 
        'work_experience_months': work_experience_months, 'certifications': certifications,
        'attendance_percentage': attendance_percentage, 'backlogs': backlogs,
        'extracurricular_activities': extracurricular_activities
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# BACKEND: FEATURE ENGINEERING & INFERENCE
st.subheader("Student Profile Summary")
st.write(input_df)

if st.button("Predict Placement & Salary"):
    # REPLICATE FEATURE ENGINEERING
    input_df["academic_composite"] = input_df[["ssc_percentage", "hsc_percentage", "degree_percentage"]].mean(axis=1)
    input_df["skill_composite"] = input_df[["technical_skill_score", "soft_skill_score"]].mean(axis=1)
    input_df["experience_score"] = (input_df["internship_count"] * 3) + (input_df["live_projects"] * 2) + (input_df["work_experience_months"] * 0.5)

    # PREDICTION LOGIC
    placement_pred = clf_pipeline.predict(input_df)[0]
    
    if placement_pred == 1:
        salary_pred = reg_pipeline.predict(input_df)[0]
        st.success("**Result: PLACED**")
        st.metric(label="Estimated Salary Package", value=f"{salary_pred:.2f} LPA")
        st.balloons()
    else:
        st.error("**Result: NOT PLACED**")
        
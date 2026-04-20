import streamlit as st
import requests

st.set_page_config(page_title="Decoupled Predictor", layout="centered")
st.title("Decoupled Predictor Client")

st.header("Input Student Data")
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    cgpa = st.number_input("CGPA", 4.0, 10.0, 7.5)
    ssc_percentage = st.number_input("SSC %", 40.0, 100.0, 75.0)
    hsc_percentage = st.number_input("HSC %", 40.0, 100.0, 75.0)
    degree_percentage = st.number_input("Degree %", 40.0, 100.0, 75.0)
    attendance_percentage = st.number_input("Attendance %", 0.0, 100.0, 85.0)
    backlogs = st.number_input("Backlogs", 0, 10, 0)
    
with col2:
    technical_skill_score = st.number_input("Technical Skill Score", 0.0, 100.0, 80.0)
    soft_skill_score = st.number_input("Soft Skill Score", 0.0, 100.0, 80.0)
    entrance_exam_score = st.number_input("Entrance Exam Score", 0.0, 100.0, 70.0)
    internship_count = st.number_input("Internships", 0, 5, 1)
    live_projects = st.number_input("Live Projects", 0, 10, 2)
    work_experience_months = st.number_input("Work Experience (Months)", 0, 60, 0)
    certifications = st.number_input("Certifications", 0, 10, 1)
    extracurricular_activities = st.selectbox("Extracurriculars", ["Yes", "No"])

if st.button("Send Request to FastAPI"):
    # Build the JSON payload matching the Pydantic schema
    payload = {
        "gender": gender, "ssc_percentage": ssc_percentage, "hsc_percentage": hsc_percentage,
        "degree_percentage": degree_percentage, "cgpa": cgpa, "entrance_exam_score": entrance_exam_score,
        "technical_skill_score": technical_skill_score, "soft_skill_score": soft_skill_score,
        "internship_count": internship_count, "live_projects": live_projects, 
        "work_experience_months": work_experience_months, "certifications": certifications,
        "attendance_percentage": attendance_percentage, "backlogs": backlogs,
        "extracurricular_activities": extracurricular_activities
    }

    try:
        response = requests.post("http://localhost:8000/predict", json=payload)
        response.raise_for_status() # Check for HTTP errors
        
        result = response.json()
        
        if result["placement_code"] == 1:
            st.success(f"API Response: {result['placement_status']}")
            st.info(f"Estimated Salary: {result['estimated_salary_lpa']:.2f} LPA")
        else:
            st.error(f"API Response: {result['placement_status']}")
            
    except requests.exceptions.ConnectionError:
        st.error("Failed to connect to FastAPI. Is your uvicorn server running on localhost:8000?")
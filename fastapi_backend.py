from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle

app = FastAPI(title="Placement Prediction API")

# Load models on startup
with open("models/best_classification_model.pkl", "rb") as f:
    clf_model = pickle.load(f)
with open("models/best_regression_model.pkl", "rb") as f:
    reg_model = pickle.load(f)

# The strict Data Contract
class StudentData(BaseModel):
    gender: str
    ssc_percentage: float
    hsc_percentage: float
    degree_percentage: float
    cgpa: float
    entrance_exam_score: float
    technical_skill_score: float
    soft_skill_score: float
    internship_count: int
    live_projects: int
    work_experience_months: int
    certifications: int
    attendance_percentage: float
    backlogs: int
    extracurricular_activities: str

@app.post("/predict")
def predict_placement(data: StudentData):
    # Convert incoming JSON to DataFrame
    df = pd.DataFrame([data.dict()])
    
    # Feature Engineering
    df["academic_composite"] = df[["ssc_percentage", "hsc_percentage", "degree_percentage"]].mean(axis=1)
    df["skill_composite"] = df[["technical_skill_score", "soft_skill_score"]].mean(axis=1)
    df["experience_score"] = (df["internship_count"] * 3) + (df["live_projects"] * 2) + (df["work_experience_months"] * 0.5)

    # Inference Logic
    placement = int(clf_model.predict(df)[0])
    
    if placement == 1:
        salary = float(reg_model.predict(df)[0])
        status = "Placed"
    else:
        salary = 0.0
        status = "Not Placed"

    return {"placement_status": status, "placement_code": placement, "estimated_salary_lpa": salary}
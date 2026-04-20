import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# PAGE CONFIG
st.set_page_config(
    page_title="Placement Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CUSTOM CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* Main background */
    .stApp {
        background: #0f0f14;
        color: #e8e6df;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #16161e !important;
        border-right: 1px solid #2a2a36;
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #c8b8f8;
        font-family: 'DM Serif Display', serif;
    }

    /* Headings */
    h1, h2, h3 { font-family: 'DM Serif Display', serif !important; }
    h1 { color: #f5f2ea !important; font-size: 2.4rem !important; }
    h2 { color: #c8b8f8 !important; }
    h3 { color: #b8d4f8 !important; }

    /* Hero banner */
    .hero-banner {
        background: #1a1030;
        border: 1px solid #2e2a45;
        border-radius: 16px;
        padding: 36px 40px 28px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    }
    .hero-title {
        font-family: 'DM Serif Display', serif;
        font-size: 2.6rem;
        color: #f5f2ea;
        margin: 0 0 6px;
        line-height: 1.15;
    }
    .hero-sub {
        font-size: 1.05rem;
        color: #8a8a9e;
        font-weight: 300;
        margin: 0;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(160,100,255,0.15);
        border: 1px solid rgba(160,100,255,0.35);
        color: #c8a0ff;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 500;
        letter-spacing: 0.08em;
        margin-bottom: 14px;
        text-transform: uppercase;
    }

    /* Result cards */
    .result-card {
        background: #1a1a24;
        border: 1px solid #2e2a45;
        border-radius: 14px;
        padding: 26px 28px;
        text-align: center;
        transition: transform 0.2s;
    }
    .result-card:hover { transform: translateY(-2px); }
    .result-card .label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #6a6a80;
        margin-bottom: 8px;
    }
    .result-card .value {
        font-family: 'DM Serif Display', serif;
        font-size: 2.4rem;
        font-weight: 400;
    }
    .result-card .sub {
        font-size: 0.82rem;
        color: #6a6a80;
        margin-top: 6px;
    }
    .placed   { color: #6ee7b7; border-color: rgba(110,231,183,0.3) !important; }
    .notplaced { color: #fca5a5; border-color: rgba(252,165,165,0.3) !important; }
    .salary-val { color: #93c5fd; }

    /* Metric pill */
    .metric-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: #1e1e2c;
        border: 1px solid #2e2a45;
        border-radius: 24px;
        padding: 8px 18px;
        font-size: 0.88rem;
        margin: 4px;
    }
    .pill-dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        display: inline-block;
    }

    /* Feature bar */
    .feature-row {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 8px;
        font-size: 0.85rem;
    }
    .feature-name { color: #8a8a9e; min-width: 180px; }
    .feature-bar-bg {
        flex: 1;
        background: #1e1e2c;
        border-radius: 4px;
        height: 6px;
        overflow: hidden;
    }
    .feature-bar-fill {
        height: 100%;
        border-radius: 4px;
        background: linear-gradient(90deg, #7c5cbf, #a78bfa);
    }
    .feature-val { color: #c8b8f8; min-width: 40px; text-align: right; font-size: 0.8rem; }

    /* Divider */
    .section-divider {
        border: none;
        border-top: 1px solid #2a2a36;
        margin: 24px 0;
    }

    /* Stmetric override */
    [data-testid="stMetric"] {
        background: #1a1a24;
        border: 1px solid #2e2a45;
        border-radius: 12px;
        padding: 16px 20px;
    }

    /* Button */
    .stButton > button {
        background: #7c3aed;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 28px;
        font-size: 1rem;
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
        letter-spacing: 0.02em;
        width: 100%;
        cursor: pointer;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.88; }

    /* Slider, number input */
    [data-testid="stSlider"] { padding: 4px 0; }

    /* Expander */
    details summary { color: #8a8a9e !important; font-size: 0.9rem; }

    /* Info box */
    .info-box {
        background: rgba(79,70,229,0.1);
        border-left: 3px solid #6366f1;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        font-size: 0.88rem;
        color: #a5b4fc;
        margin: 12px 0;
    }
    .warn-box {
        background: rgba(245,158,11,0.1);
        border-left: 3px solid #f59e0b;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        font-size: 0.88rem;
        color: #fcd34d;
        margin: 12px 0;
    }
</style>
""", unsafe_allow_html=True)

# LOAD MODELS
@st.cache_resource(show_spinner=False)
def load_models():
    models = {}
    paths = {
        "clf": "models/best_classification_model.pkl",
        "reg": "models/best_regression_model.pkl",
    }
    for key, path in paths.items():
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[key] = pickle.load(f)
        else:
            models[key] = None
    return models

# FEATURE ENGINEERING 
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["academic_composite"] = df[["ssc_percentage", "hsc_percentage", "degree_percentage"]].mean(axis=1)
    df["skill_composite"]    = df[["technical_skill_score", "soft_skill_score"]].mean(axis=1)
    df["experience_score"]   = (
        df["internship_count"] * 3
        + df["live_projects"] * 2
        + df["work_experience_months"] * 0.5
    )
    return df

CATEGORICAL_COLS = ["gender", "extracurricular_activities"]
TARGET_CLF = "placement_status"
TARGET_REG = "salary_package_lpa"

def get_input_df(inputs: dict) -> pd.DataFrame:
    df = pd.DataFrame([inputs])
    df = build_features(df)
    return df

# SIDEBAR — INPUT FORM
with st.sidebar:
    st.markdown("## Student Profile")
    st.markdown('<div class="info-box">Fill in the student\'s academic and experience details to generate a placement prediction.</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### Demographics")
    gender = st.selectbox("Gender", ["Male", "Female"])
    extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"])

    st.markdown("### Academic Performance")
    ssc_pct    = st.slider("SSC Percentage (%)",    40.0, 100.0, 70.0, 0.5)
    hsc_pct    = st.slider("HSC Percentage (%)",    40.0, 100.0, 72.0, 0.5)
    degree_pct = st.slider("Degree Percentage (%)", 40.0, 100.0, 68.0, 0.5)
    cgpa       = st.slider("CGPA (10-point scale)",  4.0,  10.0,  7.5, 0.1)
    entrance   = st.slider("Entrance Exam Score",   20,   100,   65,   1)
    attendance = st.slider("Attendance (%)",         40.0, 100.0, 80.0, 0.5)
    backlogs   = st.number_input("Number of Backlogs", 0, 20, 2, 1)

    st.markdown("### Skills & Experience")
    tech_skill  = st.slider("Technical Skill Score",  20, 100, 65, 1)
    soft_skill  = st.slider("Soft Skill Score",        20, 100, 70, 1)
    internships = st.number_input("Internship Count",  0, 10, 1, 1)
    projects    = st.number_input("Live Projects",     0, 15, 2, 1)
    work_exp    = st.number_input("Work Experience (months)", 0, 60, 6, 1)
    certs       = st.number_input("Certifications",   0, 20, 2, 1)

    st.markdown("---")
    predict_btn = st.button("   Generate Prediction")

# MAIN AREA
st.markdown("""
<div class="hero-banner">
  <div class="hero-title">Placement Predictor</div>
</div>
""", unsafe_allow_html=True)

models = load_models()
clf_loaded = models["clf"] is not None
reg_loaded = models["reg"] is not None

if not clf_loaded or not reg_loaded:
    missing = []
    if not clf_loaded: missing.append("`models/best_classification_model.pkl`")
    if not reg_loaded: missing.append("`models/best_regression_model.pkl`")
    st.markdown(f'<div class="warn-box"> Model file(s) not found: {", ".join(missing)}<br>Run <code>python ScikitLearn_Pipeline.py</code> first to generate models.</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["Prediction", "Feature Analysis"])

with tab1:
    if predict_btn:
        if not clf_loaded:
            st.error("Classification model not loaded. Run the pipeline script first.")
        else:
            raw_inputs = {
                "gender":                   gender,
                "ssc_percentage":           ssc_pct,
                "hsc_percentage":           hsc_pct,
                "degree_percentage":        degree_pct,
                "cgpa":                     cgpa,
                "entrance_exam_score":      entrance,
                "technical_skill_score":    tech_skill,
                "soft_skill_score":         soft_skill,
                "internship_count":         internships,
                "live_projects":            projects,
                "work_experience_months":   work_exp,
                "certifications":           certs,
                "attendance_percentage":    attendance,
                "backlogs":                 backlogs,
                "extracurricular_activities": extracurricular,
            }

            input_df = get_input_df(raw_inputs)

            # Classification
            placement_pred = int(models["clf"].predict(input_df)[0])
            placement_prob = float(models["clf"].predict_proba(input_df)[0][1])

            # Regression (only if placed)
            salary_pred = None
            if placement_pred == 1 and reg_loaded:
                salary_pred = float(models["reg"].predict(input_df)[0])

            # Results Display 
            placed_class = "placed" if placement_pred == 1 else "notplaced"
            placed_label = "Placed" if placement_pred == 1 else "Not Placed"
            placed_color = "#6ee7b7" if placement_pred == 1 else "#fca5a5"

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"""
                <div class="result-card {placed_class}">
                    <div class="label">Placement Status</div>
                    <div class="value {placed_class}">{placed_label}</div>
                    <div class="sub">Model decision</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                prob_pct = f"{placement_prob*100:.1f}%"
                st.markdown(f"""
                <div class="result-card">
                    <div class="label">Placement Probability</div>
                    <div class="value" style="color:#a78bfa;">{prob_pct}</div>
                    <div class="sub">Confidence score</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                if salary_pred is not None:
                    st.markdown(f"""
                    <div class="result-card">
                        <div class="label">Estimated Salary</div>
                        <div class="value salary-val">₹ {salary_pred:.2f} LPA</div>
                        <div class="sub">Annual package</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    msg = "N/A — Not placed" if placement_pred == 0 else "Regression model not loaded"
                    st.markdown(f"""
                    <div class="result-card">
                        <div class="label">Estimated Salary</div>
                        <div class="value" style="color:#4a4a60; font-size:1.4rem;">—</div>
                        <div class="sub">{msg}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Probability Gauge 
            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
            st.markdown("#### Probability Breakdown")

            fig, ax = plt.subplots(figsize=(8, 1.1))
            fig.patch.set_facecolor('#0f0f14')
            ax.set_facecolor('#0f0f14')
            ax.barh(0, 1, color='#1e1e2c', height=0.5, zorder=1)
            ax.barh(0, placement_prob, color='#6ee7b7' if placement_pred==1 else '#fca5a5', height=0.5, zorder=2)
            ax.set_xlim(0, 1)
            ax.set_yticks([])
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'], color='#6a6a80', fontsize=9)
            ax.tick_params(colors='#6a6a80')
            for spine in ax.spines.values(): spine.set_visible(False)
            ax.axvline(0.5, color='#2e2a45', linewidth=1, linestyle='--', zorder=3)
            ax.text(placement_prob, 0, f'  {placement_prob*100:.1f}%', va='center', color='#f5f2ea', fontsize=10, fontweight='bold', zorder=4)
            st.pyplot(fig, use_container_width=True)
            plt.close()

            # Input summary 
            with st.expander("View submitted inputs"):
                disp = pd.DataFrame([raw_inputs]).T.rename(columns={0: "Value"})
                st.dataframe(disp, use_container_width=True)

    else:
        st.markdown("""
        <div style="text-align:center; padding:60px 20px; color:#4a4a60;">
            <div style="font-size:0.9rem;">Fill in the student profile in the sidebar and click <strong style="color:#7c5cbf;">Generate Prediction</strong></div>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.markdown("### Feature Importance Overview")
    st.markdown('<div class="info-box">Based on your current slider values, here is how each input factor contributes to the prediction model.</div>', unsafe_allow_html=True)

    feature_data = {
        "CGPA":                  cgpa / 10,
        "Technical Skill Score": tech_skill / 100,
        "Soft Skill Score":      soft_skill / 100,
        "SSC Percentage":        ssc_pct / 100,
        "HSC Percentage":        hsc_pct / 100,
        "Degree Percentage":     degree_pct / 100,
        "Attendance":            attendance / 100,
        "Certifications":        min(certs / 10, 1.0),
        "Internships":           min(internships / 5, 1.0),
        "Live Projects":         min(projects / 8, 1.0),
        "Work Experience":       min(work_exp / 36, 1.0),
        "Backlogs (inverse)":    1 - min(backlogs / 15, 1.0),
    }

    bars_html = ""
    for name, val in sorted(feature_data.items(), key=lambda x: -x[1]):
        pct = int(val * 100)
        color = "#6ee7b7" if val > 0.7 else ("#f59e0b" if val > 0.4 else "#fca5a5")
        bars_html += f"""
        <div class="feature-row">
            <span class="feature-name">{name}</span>
            <div class="feature-bar-bg">
                <div class="feature-bar-fill" style="width:{pct}%; background:{color};"></div>
            </div>
            <span class="feature-val">{pct}%</span>
        </div>"""
    st.markdown(bars_html, unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("### Composite Scores")

    academic = (ssc_pct + hsc_pct + degree_pct) / 3
    skill    = (tech_skill + soft_skill) / 2
    exp_sc   = internships * 3 + projects * 2 + work_exp * 0.5

    c1, c2, c3 = st.columns(3)
    c1.metric("Academic Composite", f"{academic:.1f}%", delta=f"{academic - 70:.1f} vs avg")
    c2.metric("Skill Composite",    f"{skill:.1f}/100",  delta=f"{skill - 65:.1f} vs avg")
    c3.metric("Experience Score",   f"{exp_sc:.1f} pts", delta=f"{exp_sc - 15:.1f} vs avg")

import streamlit as st
import pickle
import numpy as np
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="Admissions Predictor Portal",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Custom Styling & CSS ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
        font-family: 'Inter', sans-serif;
    }
    .header-container {
        background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .header-container h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .header-container p {
        font-size: 1rem;
        opacity: 0.9;
    }
    .result-admit {
        background-color: #ecfdf5;
        border: 1px solid #10b981;
        color: #065f46;
        padding: 1.25rem;
        border-radius: 0.75rem;
        text-align: center;
        font-size: 1.25rem;
        font-weight: 600;
        margin-top: 1rem;
    }
    .result-waitlist {
        background-color: #fffbeb;
        border: 1px solid #f59e0b;
        color: #92400e;
        padding: 1.25rem;
        border-radius: 0.75rem;
        text-align: center;
        font-size: 1.25rem;
        font-weight: 600;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load the Model ---
@st.cache_resource
def load_model():
    with open('admission_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file 'admission_model.pkl' not found! Please place it in the same directory as app.py.")
    st.stop()

# --- App Header ---
st.markdown("""
    <div class="header-container">
        <h1>🎓 Admissions Outcome Predictor</h1>
        <p>Evaluate candidate profiles instantly using machine learning</p>
    </div>
""", unsafe_allow_html=True)

# --- Input Form ---
with st.form("prediction_form"):
    st.markdown("### 👤 Personal & Academic Details")
    
    # නව ඇතුළත් කිරීම් (Student Name & Specialization Added)
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        student_name = st.text_input("Student Full Name", placeholder="e.g., John Doe")
    with col_p2:
        specialization = st.selectbox("Intended Specialization", ["Finance", "Marketing", "Business Analytics", "Supply Chain", "General Management", "Other"])

    st.markdown("---")
    st.markdown("### 📊 Candidate Profile Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gpa = st.number_input("GPA", min_value=0.0, max_value=4.0, value=3.5, step=0.01, format="%.2f")
        gmat = st.number_input("GMAT Score", min_value=200, max_value=800, value=650, step=10)
        work_exp = st.number_input("Work Experience (Years)", min_value=0.0, max_value=30.0, value=2.0, step=0.5)
        international = st.selectbox("International Student?", ["No", "Yes"])

    with col2:
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        race = st.selectbox("Race / Ethnicity", ["Asian", "Black", "Hispanic", "White", "Other"])
        major = st.selectbox("Undergraduate Major", ["Business", "Computer Science", "Engineering", "Humanities", "Sciences", "Other"])
        work_industry = st.selectbox("Work Industry", ["Consulting", "Education", "Finance", "Healthcare", "Technology", "Other"])

    submit_button = st.form_submit_button(label="Predict Outcome", use_container_width=True)

# --- Prediction Logic with Encoding ---
if submit_button:
    gender_map = {"Female": 0, "Male": 1, "Other": 2}
    international_map = {"No": 0, "Yes": 1}
    
    race_list = ["Asian", "Black", "Hispanic", "Other", "White"]
    major_list = ["Business", "Computer Science", "Engineering", "Humanities", "Other", "Sciences"]
    industry_list = ["Consulting", "Education", "Finance", "Healthcare", "Other", "Technology"]

    # मॉडेलसाठी आवश्यक डेटा फ्रम तयार करणे (Model Features)
    input_data = pd.DataFrame({
        'gender': [gender_map.get(gender, 0)],
        'international': [international_map.get(international, 0)],
        'gpa': [float(gpa)],
        'major': [major_list.index(major) if major in major_list else 0],
        'race': [race_list.index(race) if race in race_list else 0],
        'gmat': [float(gmat)],
        'work_exp': [float(work_exp)],
        'work_industry': [industry_list.index(work_industry) if work_industry in industry_list else 0]
    })

    try:
        expected_features = ['gender', 'international', 'gpa', 'major', 'race', 'gmat', 'work_exp', 'work_industry']
        input_data = input_data[expected_features]

        prediction = model.predict(input_data)[0]
        
        # नाव टाकले असल्यास आउटपुटमध्ये दिसेल
        name_display = f" for <b>{student_name}</b>" if student_name else ""
        spec_display = f" ({specialization} Specialization)" if specialization else ""

        if str(prediction).lower() == "admit":
            st.markdown(f"""
                <div class="result-admit">
                    🎉 Prediction{name_display}{spec_display}: <b>ADMIT</b>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="result-waitlist">
                    ⏳ Prediction{name_display}{spec_display}: <b>WAITLIST</b>
                </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Error during prediction: {e}")

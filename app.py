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
    /* Main Background & Font */
    .main {
        background-color: #f8f9fa;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Styling */
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

    /* Card Containers */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.75rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }

    /* Result Box Styling */
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
    # Ensure your model file is named 'model.pkl' or update the filename here
    with open('admission_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file 'model.pkl' not found! Please place it in the same directory as app.py.")
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
    st.markdown("### Candidate Profile Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gpa = st.number_input("GPA", min_value=0.0, max_value=4.0, value=3.5, step=0.01, format="%.2f")
        gmat = st.number_input("GMAT Score", min_value=200, max_value=800, value=650, step=10)
        work_exp = st.number_input("Work Experience (Years)", min_value=0.0, max_value=30.0, value=2.0, step=0.5)
        international = st.selectbox("International Student?", ["No", "Yes"])

    with col2:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        race = st.selectbox("Race / Ethnicity", ["Asian", "Black", "Hispanic", "White", "Other"])
        major = st.selectbox("Undergraduate Major", ["Business", "Engineering", "Computer Science", "Humanities", "Sciences", "Other"])
        work_industry = st.selectbox("Work Industry", ["Finance", "Technology", "Consulting", "Healthcare", "Education", "Other"])

    submit_button = st.form_submit_button(label="Predict Outcome", use_container_width=True)

# --- Prediction Logic ---
if submit_button:
    # Note: Depending on how your pipeline/encoder was trained, categorical inputs 
    # may need proper encoding. If your model expects raw values or encoded columns, 
    # adapt this DataFrame array construction to match your training features.
    
    # Example input dataframe mapping your exact features:
    input_data = pd.DataFrame({
        'gender': [gender],
        'international': [1 if international == "Yes" else 0],
        'gpa': [gpa],
        'major': [major],
        'race': [race],
        'gmat': [gmat],
        'work_exp': [work_exp],
        'work_industry': [work_industry]
    })

    try:
        # Predict class and probabilities
        prediction = model.predict(input_data)[0]
        
        # Display Result
        if str(prediction).lower() == "admit":
            st.markdown(f"""
                <div class="result-admit">
                    🎉 Prediction: <b>ADMIT</b>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="result-waitlist">
                    ⏳ Prediction: <b>WAITLIST</b>
                </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.info("Tip: Ensure your input formatting matches the exact schema and encoding expected by your pickled model.")

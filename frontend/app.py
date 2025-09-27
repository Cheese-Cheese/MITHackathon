import streamlit as st
from PIL import Image
import requests
import io
import base64
import pandas as pd
import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="WoundCare AI",
    page_icon="🩹",
    layout="wide"
)

# --- Custom CSS (No Changes Here) ---
st.markdown("""
<style>
    /* ... (your existing CSS goes here, no changes needed) ... */
</style>
""", unsafe_allow_html=True)


# --- Backend URLs ---
BACKEND_URL_PREDICT = 'http://127.0.0.1:5000/predict'
BACKEND_URL_HISTORY = 'http://127.0.0.1:5000/history'
BACKEND_URL_PATIENTS = 'http://127.0.0.1:5000/patients'

# --- API Functions ---
def analyze_wound(patient_id, uploaded_file):
    """Sends image to backend for analysis and fetches history."""
    with st.spinner('Analyzing wound...'):
        try:
            files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            data = {'patient_id': patient_id}
            response = requests.post(BACKEND_URL_PREDICT, files=files, data=data)
            if response.status_code == 200:
                st.session_state.analysis_results = response.json()
                fetch_history(patient_id) 
            else:
                st.error(f"Error from server: {response.status_code}")
        except requests.exceptions.RequestException as e:
            st.error(f"Connection Error: {e}")

def fetch_history(patient_id):
    """Fetches historical data for a patient."""
    try:
        response = requests.get(BACKEND_URL_HISTORY, params={'patient_id': patient_id})
        if response.status_code == 200:
            st.session_state.history_data = response.json()
        else:
            st.error("Could not fetch history.")
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {e}")

# --- UI Components ---
def display_header():
    st.markdown("""
    <div class="header">
        <div class="logo-section">
            <div class="logo">W+</div>
            <div class="app-title">WoundCare AI</div>
        </div>
        <div class="status-indicator">AI System Online</div>
    </div>
    """, unsafe_allow_html=True)

# --- Main Application View (Refactored into a function) ---
def analysis_dashboard(patient_id_to_show):
    st.header(f"Dashboard for Patient: `{patient_id_to_show}`")
    
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        st.subheader("🎛️ New Analysis")
        uploaded_file = st.file_uploader("Upload New Wound Image", type=["jpg", "jpeg", "png"], key=f"uploader_{patient_id_to_show}")
        
        if st.button("Analyze Wound", use_container_width=True, type="primary", disabled=(uploaded_file is None)):
            analyze_wound(patient_id_to_show, uploaded_file)
        
        if 'history_data' in st.session_state and st.session_state.history_data:
            st.subheader("📈 Healing Score")
            history = st.session_state.history_data
            if len(history) > 1:
                latest = history[-1]['area']
                previous = history[-2]['area']
                change = ((latest - previous) / previous) * -100 if previous > 0 else 0
                st.metric("Change Since Last Scan", f"{change:.1f}%", f"{change:.1f}% improvement")
            else:
                st.info("Two scans needed for a healing score.")

    with col2:
        st.subheader("🖼️ Wound Image")
        if 'analysis_results' in st.session_state:
            st.image(base64.b64decode(st.session_state.analysis_results['mask']), caption='Predicted Wound Mask')
        elif uploaded_file:
            st.image(uploaded_file, caption='Uploaded Image')
        else:
            st.info("Upload an image to see the analysis here.")
            
    with col3:
        st.subheader("📊 Results & History")
        if 'analysis_results' in st.session_state:
            res = st.session_state.analysis_results
            st.metric("Wound Area (% of image)", f"{res.get('area', 0)}%")
            if res.get('tissue_analysis'):
                st.bar_chart(res['tissue_analysis'])
            st.markdown("---")

        if 'history_data' in st.session_state:
            st.subheader("Healing History")
            history = st.session_state.history_data
            if history:
                df = pd.DataFrame(history)
                df['date'] = pd.to_datetime(df['timestamp'], unit='s')
                st.line_chart(df, x='date', y='area')
            else:
                st.info("No history found for this patient yet.")

# --- Page Views ---
def doctor_view():
    st.title("👨‍⚕️ Doctor Dashboard")
    try:
        response = requests.get(BACKEND_URL_PATIENTS)
        if response.status_code == 200:
            patient_list = response.json()
            if not patient_list:
                st.warning("No patient data found in the system yet.")
                return
            
            selected_patient = st.selectbox("Select a Patient to View", options=patient_list)
            
            if selected_patient:
                if st.session_state.get("current_patient") != selected_patient:
                    st.session_state.current_patient = selected_patient
                    if 'history_data' in st.session_state: del st.session_state.history_data
                    if 'analysis_results' in st.session_state: del st.session_state.analysis_results
                    fetch_history(selected_patient)

                analysis_dashboard(selected_patient)
        else:
            st.error("Could not retrieve patient list from the server.")
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {e}")

# --- UPDATED: This function now uses st.form ---
def patient_view():
    st.title("👤 Patient Portal")
    
    with st.form("patient_form"):
        patient_id_input = st.text_input("Enter Your Patient ID and press Enter")
        submitted = st.form_submit_button("Load My Data")
        
        if submitted:
            if patient_id_input:
                st.session_state.patient_id = patient_id_input
                if 'history_data' in st.session_state: del st.session_state.history_data
                if 'analysis_results' in st.session_state: del st.session_state.analysis_results
                fetch_history(patient_id_input)
            else:
                st.warning("Please enter a Patient ID.")
            
    if 'patient_id' in st.session_state:
        analysis_dashboard(st.session_state.patient_id)
        
def login_page():
    st.title("Welcome to WoundCare AI")
    st.write("Please select your role from the sidebar to begin.")
    st.info("Note: This is a demo and does not have a secure login system.")

# --- Main App Router ---
display_header()

if 'role' not in st.session_state:
    st.session_state.role = None
if 'current_patient' not in st.session_state:
    st.session_state.current_patient = None

with st.sidebar:
    st.title("Navigation")
    role = st.radio("Select Your Role:", ["Home", "Doctor", "Patient"], index=0)
    
    if role != st.session_state.role:
        st.session_state.role = role
        keys_to_clear = ['analysis_results', 'history_data', 'patient_id', 'current_patient']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

if st.session_state.role == "Doctor":
    doctor_view()
elif st.session_state.role == "Patient":
    patient_view()
else:
    login_page()
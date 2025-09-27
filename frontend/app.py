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

# --- Custom CSS ---
st.markdown("""
<style>
    /* Main App Styling */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Card Styling */
    .st-emotion-cache-183lzff { /* This is a common class for Streamlit containers */
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Header Styling */
    .header {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 1rem 1.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .logo-section {
        display: flex;
        align-items: center;
        gap: 15px;
    }
    
    .logo {
        width: 50px;
        height: 50px;
        background: linear-gradient(135deg, #2563eb, #1e40af);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        font-size: 20px;
    }
    
    .app-title {
        font-size: 28px;
        font-weight: 700;
        color: #111827;
    }
    
    .status-indicator {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        background: rgba(22, 163, 74, 0.1);
        border: 1px solid #16a34a;
        border-radius: 20px;
        color: #16a34a;
        font-size: 14px;
        font-weight: 500;
    }
    /* Style for sidebar buttons */
    .stButton>button {
        width: 100%;
        border-radius: 0.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        background-color: transparent;
        margin-bottom: 0.5rem;
    }
    .stButton>button:hover {
        background-color: rgba(255, 255, 255, 0.1);
        border-color: #764ba2;
    }
</style>
""", unsafe_allow_html=True)


# --- Backend URLs ---
BACKEND_URL_PREDICT = 'http://127.0.0.1:5000/predict'
BACKEND_URL_HISTORY = 'http://127.0.0.1:5000/history'
BACKEND_URL_PATIENTS = 'http://127.0.0.1:5000/patients'

# --- API Functions ---
def analyze_wound(patient_id, uploaded_file, is_diabetic):
    st.session_state.uploaded_image_data = uploaded_file.getvalue()
    with st.spinner('Analyzing wound...'):
        try:
            files = {'file': (uploaded_file.name, st.session_state.uploaded_image_data, uploaded_file.type)}
            data = {'patient_id': patient_id, 'is_diabetic': str(is_diabetic)}
            response = requests.post(BACKEND_URL_PREDICT, files=files, data=data)
            if response.status_code == 200:
                st.session_state.analysis_results = response.json()
                fetch_history(patient_id) 
            else:
                # Show the specific error message from the backend's JSON response
                error_detail = response.json().get('error', 'Unknown error')
                st.error(f"Error from server ({response.status_code}): {error_detail}")
        except requests.exceptions.RequestException as e:
            st.error(f"Connection Error: {e}")

def fetch_history(patient_id):
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
        <div class="logo-section"><div class="logo">W+</div><div class="app-title">WoundCare AI</div></div>
        <div class="status-indicator">AI System Online</div>
    </div>
    """, unsafe_allow_html=True)

# --- Main Application View (Dashboard) ---
def analysis_dashboard(patient_id_to_show):
    st.header(f"Dashboard for Patient: `{patient_id_to_show}`")
    
    warning_placeholder = st.empty()
    infection_placeholder = st.empty()

    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        st.subheader("🎛️ New Analysis")
        is_diabetic_checkbox = st.checkbox("Patient has diabetes", key=f"diabetic_{patient_id_to_show}")
        uploaded_file = st.file_uploader("Upload New Wound Image", type=["jpg", "jpeg", "png"], key=f"uploader_{patient_id_to_show}")
        if st.button("Analyze Wound", use_container_width=True, type="primary", disabled=(uploaded_file is None)):
            analyze_wound(patient_id_to_show, uploaded_file, is_diabetic_checkbox)
        
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
        st.subheader("🖼️ Image & Analysis")
        if 'analysis_results' in st.session_state and 'uploaded_image_data' in st.session_state:
            sub_col_orig, sub_col_mask = st.columns(2)
            with sub_col_orig:
                st.image(st.session_state.uploaded_image_data, caption='Original Image', use_column_width=True)
            with sub_col_mask:
                st.image(base64.b64decode(st.session_state.analysis_results['mask']), caption='Predicted Wound Mask', use_column_width=True)
        elif uploaded_file:
            st.image(uploaded_file, caption='Image Pending Analysis', use_column_width=True)
        else:
            st.info("Upload an image to see the analysis here.")
            
    with col3:
        st.subheader("📊 Results & History")
        if 'analysis_results' in st.session_state:
            results = st.session_state.analysis_results
            
            healing_warning = results.get('warning')
            infection_warning = results.get('infection_warning')
            
            if infection_warning:
                infection_placeholder.error(f"🚨 **INFECTION SUSPECTED:** {infection_warning}", icon="🚨")
            
            if healing_warning:
                if "Alert" in healing_warning:
                    warning_placeholder.error(f"🚨 {healing_warning}", icon="🚨")
                else:
                    warning_placeholder.warning(f"⚠️ {healing_warning}", icon="⚠️")
            
            st.metric("Wound Area (% of image)", f"{results.get('area', 0)}%")
            
            st.write("Wound Component Classification:")
            sub_col_red, sub_col_pus = st.columns(2)
            with sub_col_red:
                st.metric("Redness Score", f"{results.get('redness_score', 0)}%")
            with sub_col_pus:
                st.metric("Pus/Slough Score", f"{results.get('pus_score', 0)}%")

            if results.get('tissue_analysis'):
                st.subheader("Tissue Type Analysis")
                st.bar_chart(results['tissue_analysis'])
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
            patient_list_data = response.json()
            if not patient_list_data:
                st.warning("No patient data found in the system yet.")
                return
            
            def format_patient_name(patient):
                status_icon = "🟢 "
                if patient['status'] == 1: status_icon = "⚠️ "
                elif patient['status'] == 2: status_icon = "🚨 "
                
                diabetic_icon = "🩸" if patient['is_diabetic'] else ""
                
                return f"{status_icon}{patient['id']} {diabetic_icon}"
            
            patient_display_names = [format_patient_name(p) for p in patient_list_data]
            selected_display_name = st.selectbox("Select a Patient to View", options=patient_display_names)
            
            if selected_display_name:
                selected_patient_id = selected_display_name.split(" ")[1]

                if st.session_state.get("current_patient") != selected_patient_id:
                    st.session_state.current_patient = selected_patient_id
                    keys_to_clear = ['history_data', 'analysis_results', 'uploaded_image_data']
                    for key in keys_to_clear:
                        if key in st.session_state: del st.session_state[key]
                    fetch_history(selected_patient_id)
                analysis_dashboard(selected_patient_id)
        else:
            st.error("Could not retrieve patient list from the server.")
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {e}")

def patient_view():
    st.title("👤 Patient Portal")
    with st.form("patient_form"):
        patient_id_input = st.text_input("Enter Your Patient ID and press Enter")
        submitted = st.form_submit_button("Load My Data")
        if submitted:
            if patient_id_input:
                st.session_state.patient_id = patient_id_input
                keys_to_clear = ['history_data', 'analysis_results', 'uploaded_image_data']
                for key in keys_to_clear:
                    if key in st.session_state: del st.session_state[key]
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
    st.session_state.role = "Home"

with st.sidebar:
    st.title("Navigation")
    previous_role = st.session_state.role

    if st.button("🏠 Home", use_container_width=True):
        st.session_state.role = "Home"
    if st.button("👨‍⚕️ Doctor Dashboard", use_container_width=True):
        st.session_state.role = "Doctor"
    if st.button("👤 Patient Portal", use_container_width=True):
        st.session_state.role = "Patient"

    if st.session_state.role != previous_role:
        keys_to_clear = ['analysis_results', 'history_data', 'patient_id', 'current_patient', 'uploaded_image_data']
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
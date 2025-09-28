import streamlit as st
from PIL import Image
import requests
import io
import base64
import pandas as pd
import datetime
from streamlit_image_comparison import image_comparison

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
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .st-emotion-cache-183lzff {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
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
    .logo-section { display: flex; align-items: center; gap: 15px; }
    .logo {
        width: 50px; height: 50px;
        background: linear-gradient(135deg, #2563eb, #1e40af);
        border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        color: white; font-weight: bold; font-size: 20px;
    }
    .app-title { font-size: 28px; font-weight: 700; color: #111827; }
    .status-indicator {
        display: flex; align-items: center; gap: 8px;
        padding: 8px 16px;
        background: rgba(22, 163, 74, 0.1);
        border: 1px solid #16a34a;
        border-radius: 20px;
        color: #16a34a; font-size: 14px; font-weight: 500;
    }
    .stButton>button {
        width: 100%; border-radius: 0.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        background-color: transparent; margin-bottom: 0.5rem;
    }
    .stButton>button:hover { background-color: rgba(255, 255, 255, 0.1); border-color: #764ba2; }
</style>
""", unsafe_allow_html=True)


# --- Backend URLs ---
BACKEND_URL_PREDICT = 'http://127.0.0.1:5000/predict'
BACKEND_URL_HISTORY = 'http://127.0.0.1:5000/history'
BACKEND_URL_PATIENTS = 'http://127.0.0.1:5000/patients'
BACKEND_URL_TRAJECTORY = 'http://127.0.0.1:5000/predict_trajectory'

# --- API Functions ---
def analyze_wound(patient_id, uploaded_file, is_diabetic):
    st.session_state.uploaded_image_data = uploaded_file.getvalue()
    with st.spinner('Analyzing wound... This may take a moment.'):
        try:
            files = {'file': (uploaded_file.name, st.session_state.uploaded_image_data, uploaded_file.type)}
            data = {'patient_id': patient_id, 'is_diabetic': str(is_diabetic)}
            response = requests.post(BACKEND_URL_PREDICT, files=files, data=data)
            if response.status_code == 200:
                st.session_state.analysis_results = response.json()
                fetch_history(patient_id) 
            else:
                st.error(f"Error from server ({response.status_code}): {response.json().get('error', 'Unknown error')}")
        except requests.exceptions.JSONDecodeError:
            st.error("Error: Backend returned an invalid response. Check the backend terminal for a traceback.")
        except requests.exceptions.RequestException as e:
            st.error(f"Connection Error: Could not reach the backend. Is it running? Details: {e}")

def fetch_history(patient_id):
    try:
        response = requests.get(BACKEND_URL_HISTORY, params={'patient_id': patient_id})
        if response.status_code == 200:
            st.session_state.history_data = response.json()
        else:
            st.error("Could not fetch history.")
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {e}")

def fetch_trajectory(patient_id):
    try:
        response = requests.get(BACKEND_URL_TRAJECTORY, params={'patient_id': patient_id})
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Could not fetch trajectory: {response.json().get('error', 'Unknown error')}")
            return []
    except requests.exceptions.RequestException as e:
        print(f"Connection Error fetching trajectory: {e}")
        return []

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
        uploaded_file = st.file_uploader(
            "Upload New Wound Image",
            type=["jpg", "jpeg", "png"],
            key=f"uploader_{patient_id_to_show}",
            help="For accurate area measurement, please place a 1cm x 1cm green object next to the wound."
        )
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
        st.subheader("🖼️ Interactive Analysis")
        if 'analysis_results' in st.session_state and 'uploaded_image_data' in st.session_state:
            original_image = Image.open(io.BytesIO(st.session_state.uploaded_image_data))
            mask_image = Image.open(io.BytesIO(base64.b64decode(st.session_state.analysis_results['mask'])))
            image_comparison(
                img1=original_image, img2=mask_image,
                label1="Original Image", label2="AI Mask",
                width=700, starting_position=50, show_labels=True,
            )
        elif uploaded_file:
            st.image(uploaded_file, caption='Image Pending Analysis')
        else:
            st.info("Upload an image to see the interactive analysis.")
            
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
            st.metric("Wound Area (cm²)", f"{results.get('area_cm2', 0):.2f} cm²")
            st.metric("Wound Area (% of image)", f"{results.get('area_percent', 0)}%")
            st.write("Color Analysis:")
            sub_col_red, sub_col_pus = st.columns(2)
            sub_col_red.metric("Redness Score", f"{results.get('redness_score', 0)}%")
            sub_col_pus.metric("Pus/Slough Score", f"{results.get('pus_score', 0)}%")
            if results.get('tissue_analysis'):
                st.subheader("Tissue Type Analysis")
                st.bar_chart(results['tissue_analysis'])
            st.markdown("---")

        if 'history_data' in st.session_state:
            st.subheader("Healing Trend Charts")
            history = st.session_state.history_data
            if history:
                df_actual = pd.DataFrame(history)
                df_actual['date'] = pd.to_datetime(df_actual['timestamp'], unit='s')
                df_actual['type'] = 'Actual'

                trajectory_data = fetch_trajectory(patient_id_to_show)
                if trajectory_data:
                    df_predicted = pd.DataFrame(trajectory_data)
                    df_predicted['date'] = pd.to_datetime(df_predicted['timestamp'], unit='s')
                    df_predicted['type'] = 'Predicted'
                    
                    # --- FIX: Only combine the columns needed for the chart ---
                    df_combined = pd.concat([
                        df_actual[['date', 'area', 'type']],
                        df_predicted[['date', 'area', 'type']]
                    ])
                else:
                    df_combined = df_actual

                st.write("**Wound Area (cm²)**")
                st.line_chart(df_combined, x='date', y='area', color='type')
                
                st.write("**Color Analysis (%)**")
                st.line_chart(df_actual, x='date', y=['redness_score', 'pus_score'])

                st.write("**Tissue Composition (%)**")
                st.area_chart(df_actual, x='date', y=['healthy_tissue', 'infected_tissue'])
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
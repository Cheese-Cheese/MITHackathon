import streamlit as st
from PIL import Image
import requests
import io
import base64
import pandas as pd
import datetime
from streamlit_image_comparison import image_comparison
from fpdf import FPDF
import matplotlib.pyplot as plt

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
    
    .main-metric {
        text-align: center;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: rgba(255, 255, 255, 0.05);
        margin-bottom: 1.5rem;
    }
    .main-metric-label {
        font-size: 1.1rem;
        color: #e5e7eb;
    }
    .main-metric-value {
        font-size: 3rem;
        font-weight: 600;
        line-height: 1.2;
        color: #ffffff;
    }
    .main-metric-delta {
        font-size: 1rem;
        font-weight: 500;
    }
    .delta-positive { color: #34D399; }
    .delta-negative { color: #F87171; }

</style>
""", unsafe_allow_html=True)


# --- Backend URLs ---
BACKEND_URL_PREDICT = 'http://127.0.0.1:5000/predict'
BACKEND_URL_HISTORY = 'http://127.0.0.1:5000/history'
BACKEND_URL_PATIENTS = 'http://127.0.0.1:5000/patients'
BACKEND_URL_TRAJECTORY = 'http://127.0.0.1:5000/predict_trajectory'

# --- PDF Generation Function ---
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'WoundCare AI - Analysis Report', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_report(patient_id, analysis_results, history_data, trajectory_data, original_image_bytes):
    pdf = PDF()
    pdf.add_page()
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(40, 10, 'Patient ID:')
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, patient_id, 0, 1)
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(40, 10, 'Analysis Date:')
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 0, 1)
    pdf.ln(10)

    try:
        original_image = Image.open(io.BytesIO(original_image_bytes))
        mask_image = Image.open(io.BytesIO(base64.b64decode(analysis_results['mask'])))
        
        pdf.image(original_image, x=10, y=pdf.get_y(), w=80, title="Original Image")
        pdf.image(mask_image, x=110, y=pdf.get_y(), w=80, title="Masked Image")
        pdf.ln(85) 
    except Exception as e:
        pdf.cell(0, 10, f"Error loading images for PDF: {e}", 0, 1)

    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Analysis Summary', 0, 1, 'L')
    
    metrics = {
        "Wound Area (cm²)": f"{analysis_results.get('area_cm2', 0):.2f} cm²",
        "Wound Area (% of image)": f"{analysis_results.get('area_percent', 0)}%",
        "Redness Score": f"{analysis_results.get('redness_score', 0)}%",
        "Pus/Slough Score": f"{analysis_results.get('pus_score', 0)}%"
    }
    for key, value in metrics.items():
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(60, 8, key + ":")
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 8, value, 0, 1)
    
    for warning_key, warning_type in [('infection_warning', 'INFECTION'), ('warning', 'HEALING')]:
        if analysis_results.get(warning_key):
            pdf.set_font('Arial', 'B', 10)
            pdf.set_text_color(220, 50, 50)
            pdf.multi_cell(0, 5, f"ALERT ({warning_type}): {analysis_results[warning_key]}", 0, 'L')
            pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    if history_data:
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Healing Trend Chart', 0, 1, 'L')
        
        df_actual = pd.DataFrame(history_data)
        df_actual['date'] = pd.to_datetime(df_actual['timestamp'], unit='s')
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df_actual['date'], df_actual['area'], marker='o', linestyle='-', label='Actual Area')
        
        if trajectory_data:
            df_pred = pd.DataFrame(trajectory_data)
            df_pred['date'] = pd.to_datetime(df_pred['timestamp'], unit='s')
            ax.plot(df_pred['date'], df_pred['area'], marker='x', linestyle='--', label='Predicted Area')

        ax.set_xlabel("Date")
        ax.set_ylabel("Wound Area (cm²)")
        ax.set_title("Wound Area Over Time")
        ax.grid(True)
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        buf.seek(0)
        pdf.image(buf, x=10, w=190)
        plt.close(fig)

    return pdf.output(dest='S')

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
    
    data_col, image_col = st.columns([1.5, 2]) 

    with data_col:
        st.subheader("🎛️ Controls & Results")
        
        is_diabetic_checkbox = st.checkbox("Patient has diabetes", key=f"diabetic_{patient_id_to_show}")
        uploaded_file = st.file_uploader(
            "Upload New Wound Image",
            type=["jpg", "jpeg", "png"],
            key=f"uploader_{patient_id_to_show}",
            help="For accurate area measurement, please place a 1cm x 1cm green object next to the wound."
        )
        if st.button("Analyze Wound", use_container_width=True, type="primary", disabled=(uploaded_file is None)):
            analyze_wound(patient_id_to_show, uploaded_file, is_diabetic_checkbox)
        
        st.markdown("---")
        
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
            
            # --- UPDATED: Reorganized metrics section ---
            if 'history_data' in st.session_state and len(st.session_state.history_data) > 1:
                history = st.session_state.history_data
                latest_area = history[-1]['area']
                previous_area = history[-2]['area']
                improvement = ((previous_area - latest_area) / previous_area) * 100 if previous_area > 0 else 0
                delta_color_class = "delta-positive" if improvement >= 0 else "delta-negative"
                delta_symbol = "↑" if improvement >= 0 else "↓"
                metric_html = f"""
                <div class="main-metric">
                    <div class="main-metric-label">Improvement Since Last Scan</div>
                    <div class="main-metric-value">{improvement:.1f}%</div>
                    <div class="main-metric-delta {delta_color_class}">{delta_symbol} {abs(improvement):.1f}%</div>
                </div>
                """
                st.markdown(metric_html, unsafe_allow_html=True)
            else:
                 # --- NEW: Placeholder for when there's not enough data ---
                metric_html = """
                <div class="main-metric">
                    <div class="main-metric-label">Improvement Since Last Scan</div>
                    <div class="main-metric-value">--.--%</div>
                    <div class="main-metric-delta">(2 or more scans required)</div>
                </div>
                """
                st.markdown(metric_html, unsafe_allow_html=True)

            st.write("##### Key Metrics")
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric("Wound Area (cm²)", f"{results.get('area_cm2', 0):.2f} cm²")
                st.metric("Redness Score", f"{results.get('redness_score', 0)}%")
            with metric_col2:
                st.metric("Wound Area (% of image)", f"{results.get('area_percent', 0)}%")
                st.metric("Pus/Slough Score", f"{results.get('pus_score', 0)}%")
            
            st.markdown("---")

        if 'history_data' in st.session_state:
            st.subheader("📈 Healing Trends")
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
                    df_combined = pd.concat([df_actual[['date', 'area', 'type']], df_predicted[['date', 'area', 'type']]])
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
        
        report_disabled = 'analysis_results' not in st.session_state
        if not report_disabled:
            st.markdown("---")
            with st.spinner("Generating Report..."):
                report_data = create_report(
                    patient_id=patient_id_to_show,
                    analysis_results=st.session_state.get('analysis_results', {}),
                    history_data=st.session_state.get('history_data', []),
                    trajectory_data=fetch_trajectory(patient_id_to_show),
                    original_image_bytes=st.session_state.get('uploaded_image_data')
                )
            st.download_button(
                label="📄 Download Report (PDF)",
                data=io.BytesIO(report_data),
                file_name=f"WoundReport_{patient_id_to_show}_{datetime.date.today()}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

    with image_col:
        st.subheader("🖼️ Interactive Analysis")
        # --- FIX: Use uploaded_file variable directly for immediate display ---
        # The key is checking the local 'uploaded_file' variable, not just session_state
        if 'analysis_results' in st.session_state and 'uploaded_image_data' in st.session_state:
            original_image = Image.open(io.BytesIO(st.session_state.uploaded_image_data))
            mask_image = Image.open(io.BytesIO(base64.b64decode(st.session_state.analysis_results['mask'])))
            image_comparison(
                img1=original_image, img2=mask_image,
                label1="Original Image", label2="AI Mask",
                width=700, starting_position=50, show_labels=True,
            )
        elif uploaded_file is not None:
            # When an image is uploaded but not yet analyzed, show it here
            st.image(uploaded_file, caption='Image Pending Analysis')
        else:
            st.info("Upload an image to see the interactive analysis.")

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
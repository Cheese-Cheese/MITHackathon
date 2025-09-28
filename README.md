# WoundCare AI 🩹

WoundCare AI is a full-stack, intelligent clinical support system designed to provide objective, data-driven analysis of chronic wounds. This tool assists healthcare professionals and patients in tracking the healing process with precision, using a suite of computer vision and machine learning models.

### 💡 The Problem

Traditional wound care monitoring is often subjective, inconsistent, and lacks precise data. This can lead to delayed interventions, especially for high-risk patients like those with diabetes, where slow healing can result in severe complications. Key challenges include inaccurate area measurements due to camera distance and the difficulty in objectively assessing tissue health.

### ✨ Our Solution

WoundCare AI replaces guesswork with quantitative analysis. By uploading an image of a wound with a simple 1cm² green calibration patch, the application provides instant feedback on:

* **Absolute Wound Area (cm²):** For clinically accurate size tracking.
* **Tissue Health:** AI-powered analysis of tissue types.
* **Healing Trajectory:** Predictive forecasting of the wound's healing path.
* **Early Warnings:** Proactive alerts for stalled healing, regression, or potential infection.

---

## 🚀 Key Features

* **Dual AI Model Analysis:**
  * **Segmentation (U-Net):** Accurately outlines the wound's precise boundaries.
  * **Classification (MobileNetV2):** Analyzes the tissue inside the wound for health vs. potential necrosis.
* **Absolute Area Calibration:** By detecting a 1cm² green patch, the system calculates the wound's true area in cm², eliminating inaccuracies from camera distance.
* **Intelligent Early Warning System:**
  * Flags stalled healing or an increase in wound size.
  * Uses a stricter threshold for diabetic patients for enhanced safety.
  * Features a rule-based **Infection Proxy** that alerts clinicians based on trends in redness, pus, and area.
* **Predictive Healing Trajectory:** A time-series model forecasts the wound's healing path, helping identify patients falling behind schedule.
* **Interactive Analysis View:** A user-friendly slider allows for a direct visual comparison between the original wound image and the AI-generated mask.
* **Role-Based Dashboards:** Separate, tailored views for Doctors (overview of all patients) and Patients (viewing their own data).
* **PDF Report Generation:** One-click generation of a comprehensive PDF report for a patient's electronic health record (EHR).
* **Visual Healing Timeline:** An image gallery tracks the wound's visual progress over time, complementing the data charts.

---

## 🛠️ Tech Stack

* **Backend:** Python, Flask, SQLite
* **Frontend:** Python, Streamlit
* **AI & Computer Vision:**
  * TensorFlow / Keras
  * Scikit-learn
  * OpenCV
  * Pandas / NumPy
* **Key UI Libraries:**
  * `streamlit-image-comparison`
  * `fpdf2`

---

## ⚙️ Setup and Installation

This project consists of two main parts: the **backend** (Flask API) and the **frontend** (Streamlit App). They must be run in two separate terminals.

### Prerequisites

* Python 3.8+
* `pip` and `venv`

### 1. Backend Setup

First, set up the server that runs the AI models.

```bash
# 1. Navigate to the backend directory
cd backend

# 2. Create and activate a Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# 3. Install the required packages
pip install tensorflow numpy flask flask-cors Pillow opencv-python scikit-learn pandas

# 4. Place the trained model files in this folder:
#    - wound_segmentation_model_v1.keras
#    - tissue_classifier_model_v1.keras

# 5. Run the Flask server
python api.py
```

The backend server will start on http://127.0.0.1:5000. Leave this terminal running.

2. Frontend Setup

Now, start the user-facing web application in a new terminal.

```bash
# 1. Navigate to the frontend directory
cd frontend

# 2. Create and activate its own virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# 3. Install the required packages
pip install streamlit Pillow requests pandas fpdf2 streamlit-image-comparison matplotlib

# 4. Run the Streamlit app
streamlit run app.py
```

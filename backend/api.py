# Updated Backend API with CORS and Debugging
import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import base64
import sqlite3
import time
import os

print("--- [DEBUG] Starting api.py script execution. ---")

app = Flask(__name__)
CORS(app)

DB_FILE = 'wound_data.db'

def init_db():
    if not os.path.exists(DB_FILE):
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE measurements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                area REAL NOT NULL
            )
        ''')
        conn.commit()
        conn.close()
        print("--- [DEBUG] Database initialized. ---")

init_db()
print("--- [DEBUG] Database check complete. ---")

# --- Load All Models ---
model_segmentation = None
model_classifier = None
try:
    print("--- [DEBUG] Attempting to load SEGMENTATION model... ---")
    model_segmentation = tf.keras.models.load_model('wound_segmentation_model_v1.keras')
    print("--- [SUCCESS] Segmentation model loaded successfully! ---")
except Exception as e:
    print(f"--- [FATAL ERROR] Could not load segmentation model: {e} ---")
    print("--- [ACTION] Make sure 'wound_segmentation_model_v1.keras' is in the backend folder! ---")
    
try:
    print("--- [DEBUG] Attempting to load CLASSIFIER model... ---")
    model_classifier = tf.keras.models.load_model('tissue_classifier_model_v1.keras')
    class_names = ['Healthy Tissue', 'Infected/Necrotic'] 
    print("--- [SUCCESS] Classifier model loaded successfully! ---")
except Exception as e:
    print(f"--- [FATAL ERROR] Could not load classifier model: {e} ---")
    print("--- [ACTION] Make sure 'tissue_classifier_model_v1.keras' is in the backend folder! ---")

print("--- [DEBUG] Model loading block finished. ---")


# --- Define Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    if model_segmentation is None or model_classifier is None:
        return jsonify({'error': 'One or more models are not loaded'}), 500
    
    if 'file' not in request.files: return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    patient_id = request.form.get('patient_id', 'default_patient')

    try:
        image = Image.open(file.stream).convert('RGB')
        image_resized = image.resize((224, 224))
        image_array = np.array(image_resized) / 255.0
        image_batch = np.expand_dims(image_array, axis=0)

        predicted_mask = model_segmentation.predict(image_batch)
        predicted_mask_binary = (predicted_mask > 0.5).astype(np.uint8)
        wound_area = np.sum(predicted_mask_binary) / (224 * 224) * 100

        predictions = model_classifier.predict(image_batch)
        scores = tf.nn.softmax(predictions[0])
        tissue_results = {class_names[i]: float(scores[i]) * 100 for i in range(len(class_names))}
        
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO measurements (patient_id, timestamp, area) VALUES (?, ?, ?)",
                       (patient_id, time.time(), wound_area))
        conn.commit()
        conn.close()

        mask_image_display = predicted_mask_binary * 255
        mask_image = Image.fromarray(mask_image_display.squeeze(), mode='L')
        buffered = io.BytesIO()
        mask_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify({
            'mask': img_str,
            'area': round(wound_area, 2),
            'tissue_analysis': tissue_results
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- Define History Endpoint ---
@app.route('/history', methods=['GET'])
def history():
    patient_id = request.args.get('patient_id')
    if not patient_id: return jsonify({'error': 'Patient ID is required'}), 400
    
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, area FROM measurements WHERE patient_id = ? ORDER BY timestamp ASC", (patient_id,))
    rows = cursor.fetchall()
    conn.close()
    
    history_data = [{'timestamp': r[0], 'area': r[1]} for r in rows]
    return jsonify(history_data)

# --- Define Patients Endpoint ---
@app.route('/patients', methods=['GET'])
def get_all_patients():
    """Returns a list of all unique patient IDs."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT patient_id FROM measurements ORDER BY patient_id ASC")
    rows = cursor.fetchall()
    conn.close()
    
    patient_ids = [row[0] for row in rows]
    return jsonify(patient_ids)


if __name__ == '__main__':
    print("--- [DEBUG] Entering main execution block. ---")
    if model_segmentation is None or model_classifier is None:
        print("--- [CRITICAL] SERVER NOT STARTED because one or more AI models failed to load. ---")
        print("--- [CRITICAL] Please check the FATAL ERROR messages above. ---")
    else:
        print("--- [INFO] Starting Flask server on http://0.0.0.0:5000 ---")
        print("--- [INFO] Press CTRL+C to stop the server. ---")
        app.run(host='0.0.0.0', port=5000)
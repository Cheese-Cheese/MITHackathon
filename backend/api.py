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
import cv2
import json

app = Flask(__name__)
CORS(app)

DB_FILE = 'wound_data.db'

def init_db():
    # ... (this function is unchanged)
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS measurements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            timestamp REAL NOT NULL,
            area REAL NOT NULL,
            warning_level INTEGER NOT NULL DEFAULT 0,
            is_diabetic INTEGER NOT NULL DEFAULT 0,
            redness_score REAL NOT NULL DEFAULT 0,
            pus_score REAL NOT NULL DEFAULT 0,
            tissue_analysis TEXT 
        )
    ''')
    conn.commit()
    conn.close()
    print("Database ready with advanced analysis columns.")

init_db()

def analyze_wound_colors(image_array_rgb, mask_array):
    # ... (this function is unchanged)
    if np.sum(mask_array) == 0: return 0, 0
    image_array_bgr = cv2.cvtColor(image_array_rgb, cv2.COLOR_RGB2BGR)
    wound_only_bgr = cv2.bitwise_and(image_array_bgr, image_array_bgr, mask=mask_array)
    hsv_wound = cv2.cvtColor(wound_only_bgr, cv2.COLOR_BGR2HSV)
    total_wound_pixels = np.count_nonzero(mask_array)
    lower_red1, upper_red1 = np.array([0, 70, 50]), np.array([10, 255, 255])
    mask_red1 = cv2.inRange(hsv_wound, lower_red1, upper_red1)
    lower_red2, upper_red2 = np.array([170, 70, 50]), np.array([180, 255, 255])
    mask_red2 = cv2.inRange(hsv_wound, lower_red2, upper_red2)
    red_mask = mask_red1 + mask_red2
    redness_score = (np.count_nonzero(red_mask) / total_wound_pixels) * 100
    lower_pus, upper_pus = np.array([20, 70, 50]), np.array([60, 255, 255])
    pus_mask = cv2.inRange(hsv_wound, lower_pus, upper_pus)
    pus_score = (np.count_nonzero(pus_mask) / total_wound_pixels) * 100
    return round(redness_score, 2), round(pus_score, 2)

def check_healing_progress(patient_id, is_diabetic):
    # ... (this function is unchanged)
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT area FROM measurements WHERE patient_id = ? ORDER BY timestamp DESC LIMIT 2", (patient_id,))
    rows = cursor.fetchall()
    conn.close()
    if len(rows) < 2: return 0, None
    latest_area, previous_area = rows[0][0], rows[1][0]
    stall_threshold = 1.0 if is_diabetic else 2.0
    if latest_area > previous_area:
        return 2, f"Alert: Wound area has increased from {previous_area:.2f}% to {latest_area:.2f}%."
    improvement = ((previous_area - latest_area) / previous_area) * 100 if previous_area > 0 else 0
    if improvement < stall_threshold:
        return 1, f"Warning: Healing has stalled. Improvement is only {improvement:.2f}%."
    return 0, None

def check_infection_proxy(patient_id):
    # ... (this function is unchanged)
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row 
    cursor = conn.cursor()
    cursor.execute("SELECT area, redness_score, pus_score FROM measurements WHERE patient_id = ? ORDER BY timestamp DESC LIMIT 2", (patient_id,))
    rows = cursor.fetchall()
    conn.close()
    if len(rows) < 2: return None
    latest, previous = dict(rows[0]), dict(rows[1])
    if (latest['redness_score'] > previous['redness_score'] and 
        latest['pus_score'] > 5.0 and latest['area'] >= previous['area']):
        return "High suspicion of infection based on trends."
    return None

# --- UPDATED: Initialize models to None before try block ---
model_segmentation = None
model_classifier = None
class_names = []

try:
    model_segmentation = tf.keras.models.load_model('wound_segmentation_model_v1.keras')
    model_classifier = tf.keras.models.load_model('tissue_classifier_model_v1.keras')
    class_names = ['Healthy Tissue', 'Infected/Necrotic'] 
    print("All models loaded successfully!")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load models. Please ensure .keras files are in the backend folder. Details: {e}")

# --- API Endpoints ---
@app.route('/predict', methods=['POST'])
def predict():
    if not all([model_segmentation, model_classifier]):
        return jsonify({'error': 'One or more AI models failed to load on the server. Check the backend terminal.'}), 500
    
    # ... (The rest of the predict function is unchanged)
    file = request.files['file']
    patient_id = request.form.get('patient_id', 'default_patient')
    is_diabetic_str = request.form.get('is_diabetic', 'false')
    is_diabetic_bool = True if is_diabetic_str.lower() == 'true' else False
    try:
        image = Image.open(file.stream).convert('RGB')
        image_resized = image.resize((224, 224))
        image_array_rgb = np.array(image_resized)
        seg_batch = np.expand_dims(image_array_rgb / 255.0, axis=0)
        class_batch = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(image_array_rgb, axis=0))
        predicted_mask = model_segmentation.predict(seg_batch)
        predicted_mask_binary = (predicted_mask[0, :, :, 0] > 0.5).astype(np.uint8)
        wound_area = np.sum(predicted_mask_binary) / (224 * 224) * 100
        redness, pus = analyze_wound_colors(image_array_rgb, predicted_mask_binary)
        predictions = model_classifier.predict(class_batch)
        scores = tf.nn.softmax(predictions[0])
        tissue_results = {class_names[i]: float(scores[i]) * 100 for i in range(len(class_names))}
        warning_level, warning_message = check_healing_progress(patient_id, is_diabetic_bool)
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        tissue_results_json = json.dumps(tissue_results)
        cursor.execute("""
            INSERT INTO measurements (patient_id, timestamp, area, warning_level, is_diabetic, redness_score, pus_score, tissue_analysis) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (patient_id, time.time(), wound_area, warning_level, int(is_diabetic_bool), redness, pus, tissue_results_json))
        conn.commit()
        conn.close()
        infection_warning = check_infection_proxy(patient_id)
        mask_image = Image.fromarray(predicted_mask_binary * 255, mode='L')
        buffered = io.BytesIO()
        mask_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return jsonify({
            'mask': img_str,
            'area': round(wound_area, 2),
            'tissue_analysis': tissue_results,
            'warning': warning_message,
            'redness_score': redness,
            'pus_score': pus,
            'infection_warning': infection_warning
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- (The /history and /patients functions are unchanged) ---
@app.route('/history', methods=['GET'])
def history():
    patient_id = request.args.get('patient_id')
    if not patient_id: return jsonify({'error': 'Patient ID is required'}), 400
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, area, redness_score, pus_score, tissue_analysis FROM measurements WHERE patient_id = ? ORDER BY timestamp ASC", (patient_id,))
    rows = cursor.fetchall()
    conn.close()
    history_data = []
    for row in rows:
        tissue_data = json.loads(row[4]) if row[4] else {}
        history_data.append({
            'timestamp': row[0], 'area': row[1], 'redness_score': row[2], 'pus_score': row[3],
            'healthy_tissue': tissue_data.get('Healthy Tissue', 0),
            'infected_tissue': tissue_data.get('Infected/Necrotic', 0)
        })
    return jsonify(history_data)

@app.route('/patients', methods=['GET'])
def get_all_patients():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    query = """
    SELECT m.patient_id, m.warning_level, m.is_diabetic
    FROM measurements m
    INNER JOIN (
        SELECT patient_id, MAX(timestamp) AS max_timestamp
        FROM measurements
        GROUP BY patient_id
    ) AS latest ON m.patient_id = latest.patient_id AND m.timestamp = latest.max_timestamp
    ORDER BY m.patient_id ASC;
    """
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    patient_data = [{'id': row[0], 'status': row[1], 'is_diabetic': row[2]} for row in rows]
    return jsonify(patient_data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
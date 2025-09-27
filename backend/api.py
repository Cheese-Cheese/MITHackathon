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

app = Flask(__name__)
CORS(app)

DB_FILE = 'wound_data.db'

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    # Defines the table with the warning_level column
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS measurements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            timestamp REAL NOT NULL,
            area REAL NOT NULL,
            warning_level INTEGER NOT NULL DEFAULT 0
        )
    ''')
    conn.commit()
    conn.close()
    print("Database ready.")

init_db()

# --- Early Warning System Logic ---
def check_healing_progress(patient_id):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT area FROM measurements WHERE patient_id = ? ORDER BY timestamp DESC LIMIT 2", (patient_id,))
    rows = cursor.fetchall()
    conn.close()
    
    if len(rows) < 2: return 0, None # No warning
        
    latest_area, previous_area = rows[0][0], rows[1][0]
    
    if latest_area > previous_area:
        message = f"Alert: Wound area has increased from {previous_area:.2f}% to {latest_area:.2f}%."
        return 2, message # Alert
        
    improvement_percentage = ((previous_area - latest_area) / previous_area) * 100 if previous_area > 0 else 0
    if improvement_percentage < 2:
        message = f"Warning: Healing has stalled. Improvement is only {improvement_percentage:.2f}%."
        return 1, message # Warning
        
    return 0, None # OK

# --- Load All Models ---
try:
    model_segmentation = tf.keras.models.load_model('wound_segmentation_model_v1.keras')
    model_classifier = tf.keras.models.load_model('tissue_classifier_model_v1.keras')
    class_names = ['Healthy Tissue', 'Infected/Necrotic'] 
    print("All models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    model_segmentation = None
    model_classifier = None

# --- API Endpoints ---
@app.route('/predict', methods=['POST'])
def predict():
    if model_segmentation is None or model_classifier is None:
        return jsonify({'error': 'One or more models are not loaded'}), 500
    
    file = request.files['file']
    patient_id = request.form.get('patient_id', 'default_patient')

    try:
        image = Image.open(file.stream).convert('RGB')
        image_resized = image.resize((224, 224))
        image_array = np.array(image_resized) / 255.0
        image_batch = np.expand_dims(image_array, axis=0)

        predicted_mask = model_segmentation.predict(image_batch)
        wound_area = np.sum((predicted_mask > 0.5)) / (224 * 224) * 100
        
        predictions = model_classifier.predict(image_batch)
        scores = tf.nn.softmax(predictions[0])
        tissue_results = {class_names[i]: float(scores[i]) * 100 for i in range(len(class_names))}
        
        warning_level, warning_message = check_healing_progress(patient_id)
        
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO measurements (patient_id, timestamp, area, warning_level) VALUES (?, ?, ?, ?)",
                       (patient_id, time.time(), wound_area, warning_level))
        conn.commit()
        conn.close()
        
        mask_image_binary = (predicted_mask > 0.5).astype(np.uint8)
        mask_image_display = mask_image_binary * 255
        mask_image = Image.fromarray(mask_image_display.squeeze(), mode='L')
        buffered = io.BytesIO()
        mask_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify({
            'mask': img_str,
            'area': round(wound_area, 2),
            'tissue_analysis': tissue_results,
            'warning': warning_message
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- THIS IS THE FUNCTION THAT WAS LIKELY MISSING OR INCORRECT ---
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

@app.route('/patients', methods=['GET'])
def get_all_patients():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    query = """
    SELECT m.patient_id, m.warning_level
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
    
    patient_data = [{'id': row[0], 'status': row[1]} for row in rows]
    return jsonify(patient_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
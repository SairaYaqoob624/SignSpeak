import time
import os
import numpy as np
import pandas as pd
import csv
import sqlite3
import threading
from datetime import datetime, timedelta
from flask import Flask, Response, render_template, jsonify, send_from_directory, request, session
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.models import load_model  # type: ignore
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from sklearn.model_selection import train_test_split
os.chdir(os.path.dirname(os.path.abspath(__file__)))
app = Flask(__name__, template_folder="../frontend")
app.secret_key = 'supersignsecret_key' 

live_gesture = ""          
last_spoken = ""
last_time = 0
COOLDOWN = 4.0             
model = load_model('hand_gesture_model.h5')
LIVE_LOGS = [] # Format: {word, user, time}
MAX_LIVE_LOGS = 15
is_recording = False
recording_label = ""
recorded_count = 0
MAX_RECORD_FRAMES = 100
_base = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(_base, '..', 'dataset', 'hand_landmarks.csv')

def load_label_encoder():
    data = pd.read_csv(csv_path)
    le = LabelEncoder()
    le.fit(data['label'].unique())
    return le

le = load_label_encoder()
from functools import wraps

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({"status": "error", "message": "Login required"}), 401
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get('role') != 'admin':
            return jsonify({"status": "error", "message": "Admin privileges required"}), 403
        return f(*args, **kwargs)
    return decorated_function

DB_PATH = 'signspeak.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  username TEXT UNIQUE, 
                  password_hash TEXT, 
                  role TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS gesture_logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  user_id INTEGER,
                  gesture TEXT, 
                  confidence REAL, 
                  timestamp DATETIME,
                  FOREIGN KEY(user_id) REFERENCES users(id))''')
    c.execute("SELECT * FROM users WHERE username = 'admin'")
    if not c.fetchone():
        hashed_pw = generate_password_hash('admin123')
        c.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                  ('admin', hashed_pw, 'admin'))
        
    conn.commit()
    conn.close()

init_db()

def log_gesture_async(user_id, gesture, confidence):
    def run():
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("INSERT INTO gesture_logs (user_id, gesture, confidence, timestamp) VALUES (?, ?, ?, ?)",
                      (user_id, gesture, float(confidence), datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error logging gesture: {e}")
    
    threading.Thread(target=run).start()

def normalize_landmarks(landmarks):
    if not landmarks or len(landmarks) != 63:
        return landmarks
    x0, y0, z0 = landmarks[0], landmarks[1], landmarks[2]
    shifted = []
    for i in range(21):
        shifted.append(landmarks[i*3] - x0)
        shifted.append(landmarks[i*3+1] - y0)
        shifted.append(landmarks[i*3+2] - z0)
    max_val = max(abs(val) for val in shifted)
    if max_val > 0:
        return [val / max_val for val in shifted]
    return shifted

# API Endpoint for Live Gesture Prediction (POST)
@app.route('/api/predict', methods=['POST'])
def predict_gesture():
    global live_gesture, last_spoken, last_time
    
    u_id = session.get('user_id')
    u_name = session.get('username')
    
    data = request.json
    landmarks = data.get('landmarks', [])
    
    if not landmarks or len(landmarks) != 63:
        live_gesture = ""
        return jsonify({"gesture": "", "confidence": 0.0})
        
    try:
        normalized = normalize_landmarks(landmarks)
        X = np.array(normalized).reshape(1, -1)
        pred = model.predict(X, verbose=0)
        
        gesture = le.inverse_transform([np.argmax(pred)])[0]
        confidence = float(np.max(pred))
        
        live_gesture = f"{gesture} ({confidence*100:.1f}%)"
        
        if u_id and u_name:
            now = time.time()
            if gesture != last_spoken and (now - last_time) >= COOLDOWN:
                last_spoken = gesture
                last_time = now
                log_gesture_async(u_id, gesture, confidence)
                LIVE_LOGS.insert(0, {
                    "word": gesture,
                    "user": u_name,
                    "time": datetime.now().strftime("%H:%M:%S")
                })
                if len(LIVE_LOGS) > MAX_LIVE_LOGS:
                    LIVE_LOGS.pop()
                    
        return jsonify({"gesture": gesture, "confidence": confidence})
    except Exception as e:
        print(f"Prediction API error: {e}")
        return jsonify({"error": str(e)}), 500


# API Endpoint for Custom Gesture Landmark Recording (POST)
@app.route('/api/record_landmark', methods=['POST'])
@login_required
def record_landmark():
    global is_recording, recorded_count, recording_label
    
    data = request.json
    landmarks = data.get('landmarks', [])
    label = data.get('label', '')
    
    if not landmarks or len(landmarks) != 63 or not label:
        return jsonify({"status": "error", "message": "Invalid landmarks or label"}), 400
        
    if is_recording and recorded_count < MAX_RECORD_FRAMES:
        try:
            normalized = normalize_landmarks(landmarks)
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(normalized + [label])
            recorded_count += 1
            if recorded_count >= MAX_RECORD_FRAMES:
                is_recording = False
            return jsonify({"status": "recorded", "count": recorded_count})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
            
    return jsonify({"status": "not_recording", "count": recorded_count})

@app.route('/')
def index():
    if 'user_id' not in session:
        return render_template('auth.html') # Serve auth if not logged in
    return render_template('index.html')

@app.route('/live_gesture')
def live():
    return jsonify({"gesture": live_gesture})
@app.route('/current_gesture')
def current_gesture():
    if live_gesture and " (" in live_gesture:
        parts = live_gesture.split(" (")
        word = parts[0]
        try:
            conf = float(parts[1].replace("%)", ""))
        except Exception:
            conf = 0.0
    else:
        word = live_gesture
        conf = 0.0
    return jsonify({"gesture": word, "confidence": conf})

@app.route('/style.css')
def style():
    return send_from_directory('../frontend', 'style.css')

@app.route('/script.js')
def script():
    return send_from_directory('../frontend', 'script.js')

@app.route('/auth.html')
def auth_page():
    return send_from_directory('../frontend', 'auth.html')

@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({"status": "error", "message": "Missing credentials"}), 400
        
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        hashed_pw = generate_password_hash(password)
        c.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                  (username, hashed_pw, 'user'))
        conn.commit()
        conn.close()
        return jsonify({"status": "success", "message": "User created! Please login."})
    except sqlite3.IntegrityError:
        return jsonify({"status": "error", "message": "Username already exists"}), 400

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, password_hash, role FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()
    
    if user and check_password_hash(user[1], password):
        session['user_id'] = user[0]
        session['username'] = username
        session['role'] = user[2]
        return jsonify({
            "status": "success", 
            "username": username, 
            "role": user[2]
        })
    
    return jsonify({"status": "error", "message": "Invalid username or password"}), 401

@app.route('/api/logout')
def logout():
    session.clear()
    return jsonify({"status": "success"})

@app.route('/api/me')
def get_me():
    if 'user_id' in session:
        return jsonify({
            "logged_in": True, 
            "username": session['username'], 
            "role": session['role']
        })
    return jsonify({"logged_in": False})

@app.route('/api/admin/users')
@admin_required
def admin_users():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            SELECT u.id, u.username, u.role, COUNT(l.id) as detections 
            FROM users u 
            LEFT JOIN gesture_logs l ON u.id = l.user_id 
            GROUP BY u.id
        """)
        users = [
            {"id": row[0], "username": row[1], "role": row[2], "detections": row[3]} 
            for row in c.fetchall()
        ]
        conn.close()
        return jsonify(users)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/delete_user', methods=['POST'])
@admin_required
def delete_user():
    user_id = request.json.get('user_id')
    if not user_id:
        return jsonify({"error": "No user ID provided"}), 400
        
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("DELETE FROM gesture_logs WHERE user_id = ?", (user_id,))
        c.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        conn.close()
        return jsonify({"status": "success", "message": "User and logs deleted."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/system_status')
@admin_required
def system_status():
    try:
        db_size = os.path.getsize(DB_PATH) / 1024 # KB
        model_size = os.path.getsize('hand_gesture_model.h5') / (1024 * 1024) # MB
        
        return jsonify({
            "model_ready": True,
            "model_name": "hand_gesture_model.h5",
            "model_size": f"{model_size:.2f} MB",
            "db_size": f"{db_size:.1f} KB",
            "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "python_version": "3.13.1",
            "status": "Healthy"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/live_feed')
@admin_required
def live_feed():
    return jsonify(LIVE_LOGS)

@app.route('/api/analytics/summary')
@admin_required
def analytics_summary():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT gesture, COUNT(*) as count FROM gesture_logs GROUP BY gesture ORDER BY count DESC LIMIT 5")
        top_gestures = [{"gesture": row[0], "count": row[1]} for row in c.fetchall()]
        c.execute("SELECT COUNT(*) FROM gesture_logs")
        total_count = c.fetchone()[0]
        c.execute("SELECT COUNT(DISTINCT gesture) FROM gesture_logs")
        unique_gestures = c.fetchone()[0]
        
        conn.close()
        return jsonify({
            "top_gestures": top_gestures,
            "total_detections": total_count,
            "unique_gestures": unique_gestures
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/analytics/stats')
@admin_required
def analytics_stats():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        yesterday = (datetime.now() - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
        c.execute("""
            SELECT strftime('%H:00', timestamp) as hour, COUNT(*) 
            FROM gesture_logs 
            WHERE timestamp >= ? 
            GROUP BY hour 
            ORDER BY timestamp ASC
        """, (yesterday,))
        
        stats = [{"hour": row[0], "count": row[1]} for row in c.fetchall()]
        conn.close()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_gestures')
def get_gestures():
    data = pd.read_csv(csv_path)
    labels = data['label'].unique().tolist()
    return jsonify({"gestures": labels})

@app.route('/start_recording', methods=['POST'])
def start_recording():
    global is_recording, recording_label, recorded_count
    label = request.json.get('label', 'unnamed')
    recording_label = label
    recorded_count = 0
    is_recording = True
    return jsonify({"status": "started", "label": label})

@app.route('/recording_status')
def recording_status():
    return jsonify({
        "is_recording": is_recording,
        "count": recorded_count,
        "max": MAX_RECORD_FRAMES
    })

@app.route('/train_model_async', methods=['POST'])
def retrain():
    global model, le
    try:
        df = pd.read_csv(csv_path)
        X = df.drop('label', axis=1)
        y = df['label']
        le = LabelEncoder()
        y_encoded = to_categorical(le.fit_transform(y))
        
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2)
        new_model = Sequential([
            Dense(128, activation='relu', input_shape=(X.shape[1],)),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(y_encoded.shape[1], activation='softmax')
        ])
        new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        new_model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)
        new_model.save('hand_gesture_model.h5')
        model = load_model('hand_gesture_model.h5')
        
        return jsonify({"status": "success", "message": "Model trained and reloaded!"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    print("SignSpeak Running at http://127.0.0.1:5000")
    app.run(debug=True, use_reloader=False)

import os
import cv2
import numpy as np
import pandas as pd
import face_recognition
from datetime import datetime
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import sqlite3
import pickle
import time
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import threading
import queue
import av

# Constants
KNOWN_FACES_DIR = 'known_faces'
ATTENDANCE_DB = 'attendance.db'
ENCODINGS_FILE = 'face_encodings.pkl'
TOLERANCE = 0.6
FRAME_THICKNESS = 2
FONT_THICKNESS = 1
MODEL = 'hog'  # or 'cnn' for better accuracy but slower performance

# Create necessary directories and files
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# Initialize database
def init_db():
    conn = sqlite3.connect(ATTENDANCE_DB)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS attendance
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT,
                  timestamp DATETIME,
                  date DATE,
                  status TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS registered_faces
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT UNIQUE,
                  registration_date DATETIME)''')
    conn.commit()
    conn.close()

init_db()

# Load or create face encodings
def load_face_encodings():
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, 'rb') as f:
            return pickle.load(f)
    return {'names': [], 'encodings': []}

def save_face_encodings(data):
    with open(ENCODINGS_FILE, 'wb') as f:
        pickle.dump(data, f)

face_data = load_face_encodings()

# Face recognition functions
def register_new_face(name, image_array):
    # Check if name already exists
    conn = sqlite3.connect(ATTENDANCE_DB)
    c = conn.cursor()
    c.execute("SELECT name FROM registered_faces WHERE name=?", (name,))
    if c.fetchone():
        conn.close()
        return False
    
    # Encode the face
    face_locations = face_recognition.face_locations(image_array, model=MODEL)
    if not face_locations:
        return False
    
    face_encoding = face_recognition.face_encodings(image_array, face_locations)[0]
    
    # Update face data
    face_data['names'].append(name)
    face_data['encodings'].append(face_encoding)
    save_face_encodings(face_data)
    
    # Save to database
    now = datetime.now()
    c.execute("INSERT INTO registered_faces (name, registration_date) VALUES (?, ?)", 
              (name, now))
    conn.commit()
    conn.close()
    
    # Save the image for reference
    cv2.imwrite(os.path.join(KNOWN_FACES_DIR, f"{name}.jpg"), cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
    
    return True

def recognize_face(frame):
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    
    face_locations = face_recognition.face_locations(rgb_small_frame, model=MODEL)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
    recognized_names = []
    
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(face_data['encodings'], face_encoding, TOLERANCE)
        name = "Unknown"
        
        if True in matches:
            first_match_index = matches.index(True)
            name = face_data['names'][first_match_index]
        
        recognized_names.append(name)
    
    # Scale back up face locations
    face_locations = [(top * 4, right * 4, bottom * 4, left * 4) 
                      for (top, right, bottom, left) in face_locations]
    
    return face_locations, recognized_names

def mark_attendance(name):
    conn = sqlite3.connect(ATTENDANCE_DB)
    c = conn.cursor()
    now = datetime.now()
    today = now.date()
    
    # Check if already marked today
    c.execute("SELECT * FROM attendance WHERE name=? AND date=?", (name, today))
    if not c.fetchone():
        c.execute("INSERT INTO attendance (name, timestamp, date, status) VALUES (?, ?, ?, ?)",
                  (name, now, today, "Present"))
        conn.commit()
    conn.close()

# Database functions
def get_attendance_data():
    conn = sqlite3.connect(ATTENDANCE_DB)
    df = pd.read_sql("SELECT * FROM attendance ORDER BY timestamp DESC", conn)
    conn.close()
    return df

def get_registered_faces():
    conn = sqlite3.connect(ATTENDANCE_DB)
    df = pd.read_sql("SELECT * FROM registered_faces ORDER BY name", conn)
    conn.close()
    return df

def get_attendance_stats():
    conn = sqlite3.connect(ATTENDANCE_DB)
    # Daily stats
    daily_stats = pd.read_sql('''SELECT date, COUNT(*) as count 
                                 FROM attendance 
                                 GROUP BY date 
                                 ORDER BY date''', conn)
    # Person stats
    person_stats = pd.read_sql('''SELECT name, COUNT(*) as count 
                                  FROM attendance 
                                  GROUP BY name 
                                  ORDER BY count DESC''', conn)
    conn.close()
    return daily_stats, person_stats

# Video transformer for real-time processing
class FaceRecognitionTransformer(VideoTransformerBase):
    def __init__(self):
        self.attendance_log = queue.Queue()
        self.last_detection_time = {}
        self.detection_interval = 10  # seconds between detections for same person
    
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Recognize faces
        face_locations, recognized_names = recognize_face(img)
        
        # Draw rectangles and names
        for (top, right, bottom, left), name in zip(face_locations, recognized_names):
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), FRAME_THICKNESS)
            cv2.putText(img, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 255, 255), FONT_THICKNESS)
            
            # Mark attendance with time interval check
            current_time = time.time()
            if name != "Unknown" and (name not in self.last_detection_time or 
                                     current_time - self.last_detection_time[name] > self.detection_interval):
                mark_attendance(name)
                self.attendance_log.put((name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                self.last_detection_time[name] = current_time
        
        return img

# Streamlit UI
st.set_page_config(page_title="Attendance System", layout="wide")
st.title("Advanced Attendance Management System with Face Recognition")

# Sidebar for navigation
menu = st.sidebar.selectbox("Menu", ["Home", "Register New Face", "Attendance Records", "Analytics", "Settings"])

if menu == "Home":
    st.header("Real-time Attendance")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Live Camera Feed")
        ctx = webrtc_streamer(
            key="example",
            video_transformer_factory=FaceRecognitionTransformer,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
        )
        
        if ctx.video_transformer:
            st.write("### Recent Attendance Logs")
            log_placeholder = st.empty()
            
            def update_log():
                while True:
                    try:
                        name, timestamp = ctx.video_transformer.attendance_log.get_nowait()
                        log_placeholder.write(f"{timestamp}: {name} marked present")
                    except queue.Empty:
                        time.sleep(0.1)
            
            threading.Thread(target=update_log, daemon=True).start()
    
    with col2:
        st.subheader("Quick Stats")
        daily_stats, person_stats = get_attendance_stats()
        
        st.metric("Total Registered Faces", len(face_data['names']))
        st.metric("Today's Attendance", 
                  len(get_attendance_data()[get_attendance_data()['date'] == datetime.now().date().isoformat()]))
        
        st.write("### Top Attendees")
        st.table(person_stats.head(5))

elif menu == "Register New Face":
    st.header("Register New Face")
    
    name = st.text_input("Enter person's name")
    uploaded_file = st.file_uploader("Upload a clear face image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None and name:
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Register Face"):
            if register_new_face(name, image_array):
                st.success(f"Face registered successfully for {name}!")
                face_data = load_face_encodings()  # Reload data
            else:
                st.error("Failed to detect a face in the image. Please try another photo.")

elif menu == "Attendance Records":
    st.header("Attendance Records")
    
    df = get_attendance_data()
    st.dataframe(df)
    
    # Filter options
    st.subheader("Filter Records")
    col1, col2 = st.columns(2)
    
    with col1:
        date_filter = st.date_input("Filter by date")
    with col2:
        name_filter = st.selectbox("Filter by name", ["All"] + list(df['name'].unique()))
    
    if st.button("Apply Filters"):
        if date_filter:
            df = df[df['date'] == date_filter.strftime("%Y-%m-%d")]
        if name_filter != "All":
            df = df[df['name'] == name_filter]
    
    st.dataframe(df)
    
    # Export options
    st.subheader("Export Data")
    export_format = st.selectbox("Select format", ["CSV", "Excel"])
    if st.button("Export"):
        if export_format == "CSV":
            csv = df.to_csv(index=False)
            st.download_button("Download CSV", csv, "attendance_records.csv", "text/csv")
        else:
            excel = df.to_excel(index=False)
            st.download_button("Download Excel", excel, "attendance_records.xlsx", "application/vnd.ms-excel")

elif menu == "Analytics":
    st.header("Attendance Analytics")
    
    daily_stats, person_stats = get_attendance_stats()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Daily Attendance Trend")
        if not daily_stats.empty:
            fig, ax = plt.subplots()
            ax.plot(pd.to_datetime(daily_stats['date']), daily_stats['count'])
            ax.set_xlabel("Date")
            ax.set_ylabel("Number of Attendees")
            st.pyplot(fig)
        else:
            st.warning("No attendance data available")
    
    with col2:
        st.subheader("Top Attendees")
        if not person_stats.empty:
            fig, ax = plt.subplots()
            sns.barplot(x='count', y='name', data=person_stats.head(10), ax=ax)
            ax.set_xlabel("Days Attended")
            ax.set_ylabel("Name")
            st.pyplot(fig)
        else:
            st.warning("No attendance data available")
    
    st.subheader("Attendance Heatmap")
    df = get_attendance_data()
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()
        
        heatmap_data = df.pivot_table(index='day_of_week', columns='hour', 
                                      values='name', aggfunc='count', fill_value=0)
        
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_data = heatmap_data.reindex(days_order)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(heatmap_data, cmap="YlGnBu", ax=ax)
        ax.set_title("Attendance by Day and Hour")
        st.pyplot(fig)
    else:
        st.warning("No attendance data available")

elif menu == "Settings":
    st.header("System Settings")
    
    st.subheader("Face Recognition Parameters")
    new_tolerance = st.slider("Recognition Tolerance (lower is stricter)", 
                              0.0, 1.0, TOLERANCE, 0.05)
    
    model_choice = st.radio("Face Detection Model", 
                            ["HOG (faster, less accurate)", "CNN (slower, more accurate)"], 
                            index=0 if MODEL == 'hog' else 1)
    
    if st.button("Save Settings"):
        global TOLERANCE, MODEL
        TOLERANCE = new_tolerance
        MODEL = 'hog' if model_choice.startswith("HOG") else 'cnn'
        st.success("Settings saved!")
    
    st.subheader("System Maintenance")
    if st.button("Rebuild Face Encodings"):
        # This would involve scanning the known_faces directory and recreating encodings
        st.warning("This feature would scan all images and rebuild encodings. Not implemented in this demo.")
    
    if st.button("Clear All Data (Danger!)"):
        st.error("This will delete all attendance records and face data. Proceed with caution!")
        if st.checkbox("I understand this cannot be undone"):
            if st.button("Confirm Delete"):
                conn = sqlite3.connect(ATTENDANCE_DB)
                c = conn.cursor()
                c.execute("DELETE FROM attendance")
                c.execute("DELETE FROM registered_faces")
                conn.commit()
                conn.close()
                
                os.remove(ENCODINGS_FILE)
                for file in os.listdir(KNOWN_FACES_DIR):
                    os.remove(os.path.join(KNOWN_FACES_DIR, file))
                
                face_data = {'names': [], 'encodings': []}
                st.warning("All data has been deleted!")

st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5em 1em;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stAlert {
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

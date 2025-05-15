import os
import cv2
import numpy as np
import pandas as pd
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
from deepface import DeepFace  # Alternative to face-recognition

# Constants
KNOWN_FACES_DIR = 'known_faces'
ATTENDANCE_DB = 'attendance.db'
ENCODINGS_FILE = 'face_encodings.pkl'
TOLERANCE = 0.6
FRAME_THICKNESS = 2
FONT_THICKNESS = 1

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

# Face recognition functions using DeepFace
def register_new_face(name, image_array):
    # Check if name already exists
    conn = sqlite3.connect(ATTENDANCE_DB)
    c = conn.cursor()
    c.execute("SELECT name FROM registered_faces WHERE name=?", (name,))
    if c.fetchone():
        conn.close()
        return False
    
    # Save the image temporarily
    temp_path = os.path.join(KNOWN_FACES_DIR, f"temp_{name}.jpg")
    cv2.imwrite(temp_path, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
    
    try:
        # Get face embedding using DeepFace
        embedding = DeepFace.represent(temp_path, model_name='Facenet')[0]["embedding"]
        
        # Save to database
        now = datetime.now()
        c.execute("INSERT INTO registered_faces (name, registration_date) VALUES (?, ?)", 
                  (name, now))
        conn.commit()
        
        # Save the image for reference
        final_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
        os.rename(temp_path, final_path)
        
        return True
    except Exception as e:
        st.error(f"Face registration failed: {str(e)}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False
    finally:
        conn.close()

def recognize_face(frame):
    temp_path = os.path.join(KNOWN_FACES_DIR, "temp_frame.jpg")
    cv2.imwrite(temp_path, frame)
    
    try:
        # Find similar faces in the database
        dfs = DeepFace.find(
            img_path=temp_path,
            db_path=KNOWN_FACES_DIR,
            model_name='Facenet',
            enforce_detection=False,
            silent=True
        )
        
        recognized_names = []
        face_locations = []
        
        if isinstance(dfs, list) and len(dfs) > 0:
            df = dfs[0]
            if len(df) > 0:
                for _, row in df.iterrows():
                    recognized_names.append(row['identity'].split('/')[-1].split('.')[0])
                    # Create dummy face locations (DeepFace doesn't provide exact locations)
                    face_locations.append((
                        int(row['source_y']),
                        int(row['source_x'] + row['source_w']),
                        int(row['source_y'] + row['source_h']),
                        int(row['source_x'])
                    ))
        
        return face_locations, recognized_names
    except Exception as e:
        st.error(f"Face recognition error: {str(e)}")
        return [], []
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

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

# Database functions (unchanged)
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
    daily_stats = pd.read_sql('''SELECT date, COUNT(*) as count 
                                 FROM attendance 
                                 GROUP BY date 
                                 ORDER BY date''', conn)
    person_stats = pd.read_sql('''SELECT name, COUNT(*) as count 
                                  FROM attendance 
                                  GROUP BY name 
                                  ORDER BY count DESC''', conn)
    conn.close()
    return daily_stats, person_stats

# Video transformer using DeepFace
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

# Streamlit UI (unchanged except for removed model selection)
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
        
        registered_faces = get_registered_faces()
        st.metric("Total Registered Faces", len(registered_faces))
        
        today_attendance = len(get_attendance_data()[get_attendance_data()['date'] == datetime.now().date().isoformat()])
        st.metric("Today's Attendance", today_attendance)
        
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
            else:
                st.error("Registration failed. Please try another photo.")

elif menu == "Attendance Records":
    st.header("Attendance Records")
    
    df = get_attendance_data()
    st.dataframe(df)
    
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
    
    if st.button("Save Settings"):
        global TOLERANCE
        TOLERANCE = new_tolerance
        st.success("Settings saved!")
    
    st.subheader("System Maintenance")
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
                
                for file in os.listdir(KNOWN_FACES_DIR):
                    if file.endswith('.jpg'):
                        os.remove(os.path.join(KNOWN_FACES_DIR, file))
                
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

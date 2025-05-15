import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import sqlite3
import time
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import threading
import queue
import av

# Constants
TOLERANCE = 0.6
KNOWN_FACES_DIR = 'known_faces'
ATTENDANCE_DB = 'attendance.db'
FRAME_THICKNESS = 2
FONT_THICKNESS = 1
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# Create directories
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# Initialize database with admin table
def init_db():
    conn = sqlite3.connect(ATTENDANCE_DB)
    c = conn.cursor()
    
    # Attendance tables
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
    
    # Admin table
    c.execute('''CREATE TABLE IF NOT EXISTS admin_users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE,
                  password TEXT,
                  is_superadmin BOOLEAN DEFAULT FALSE)''')
    
    # Add default admin if not exists
    c.execute("SELECT * FROM admin_users WHERE username='admin'")
    if not c.fetchone():
        c.execute("INSERT INTO admin_users (username, password, is_superadmin) VALUES (?, ?, ?)",
                 ('admin', 'admin123', True))
    
    conn.commit()
    conn.close()

init_db()

# Authentication
def authenticate(username, password):
    conn = sqlite3.connect(ATTENDANCE_DB)
    c = conn.cursor()
    c.execute("SELECT * FROM admin_users WHERE username=? AND password=?", (username, password))
    user = c.fetchone()
    conn.close()
    return user

# Face recognition functions
def register_new_face(name, image_array):
    conn = sqlite3.connect(ATTENDANCE_DB)
    c = conn.cursor()
    c.execute("SELECT name FROM registered_faces WHERE name=?", (name,))
    if c.fetchone():
        conn.close()
        return False
    
    try:
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        face_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
        cv2.imwrite(face_path, gray)
        
        now = datetime.now()
        c.execute("INSERT INTO registered_faces (name, registration_date) VALUES (?, ?)", 
                 (name, now))
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Registration failed: {str(e)}")
        return False
    finally:
        conn.close()

def recognize_face(image_array):
    recognized_names = []
    gray_input = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    current_tolerance = st.session_state.get('tolerance', TOLERANCE)
    
    for face_file in os.listdir(KNOWN_FACES_DIR):
        if face_file.endswith('.jpg'):
            name = os.path.splitext(face_file)[0]
            face_path = os.path.join(KNOWN_FACES_DIR, face_file)
            registered_face = cv2.imread(face_path, cv2.IMREAD_GRAYSCALE)
            
            if registered_face is None:
                continue
                
            try:
                registered_face = cv2.resize(registered_face, (gray_input.shape[1], gray_input.shape[0]))
                result = cv2.matchTemplate(gray_input, registered_face, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                
                if max_val > current_tolerance:
                    recognized_names.append(name)
            except:
                continue
    
    return recognized_names

def mark_attendance(name):
    conn = sqlite3.connect(ATTENDANCE_DB)
    c = conn.cursor()
    now = datetime.now()
    today = now.date()
    
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
    daily_stats = pd.read_sql('''SELECT date, COUNT(*) as count 
                               FROM attendance 
                               GROUP BY date 
                               ORDER BY date''', conn)
    person_stats = pd.read_sql('''SELECT name, COUNT(*) as count 
                                FROM attendance 
                                GROUP BY name 
                                ORDER BY count DESC''', conn)
    hourly_stats = pd.read_sql('''SELECT strftime('%H', timestamp) as hour, 
                                 COUNT(*) as count FROM attendance
                                 GROUP BY hour ORDER BY hour''', conn)
    conn.close()
    return daily_stats, person_stats, hourly_stats

# Video transformers
class FaceRegistrationTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame_queue = queue.Queue()
    
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_queue.put(img)
        return img

class FaceRecognitionTransformer(VideoTransformerBase):
    def __init__(self):
        self.attendance_log = queue.Queue()
        self.last_detection = {}
        self.detection_interval = 10
    
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        recognized_names = recognize_face(img)
        
        for name in recognized_names:
            current_time = time.time()
            if name not in self.last_detection or current_time - self.last_detection[name] > self.detection_interval:
                mark_attendance(name)
                self.attendance_log.put((name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                self.last_detection[name] = current_time
        
        return img

# Streamlit UI
st.set_page_config(page_title="Attendance System", layout="wide")

# Initialize session state
if 'tolerance' not in st.session_state:
    st.session_state.tolerance = TOLERANCE
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'current_user' not in st.session_state:
    st.session_state.current_user = None

# Login page
if not st.session_state.authenticated:
    st.title("Admin Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        user = authenticate(username, password)
        if user:
            st.session_state.authenticated = True
            st.session_state.current_user = user
            st.rerun()
        else:
            st.error("Invalid credentials")
    st.stop()

# Main App
st.title("Advanced Attendance Management System")

# Sidebar
with st.sidebar:
    st.write(f"Logged in as: **{st.session_state.current_user[1]}**")
    if st.session_state.current_user[3]:  # is_superadmin
        menu = st.selectbox("Menu", ["Dashboard", "Real-time Attendance", "Register Faces", 
                                   "Attendance Records", "Analytics", "User Management", "Settings"])
    else:
        menu = st.selectbox("Menu", ["Dashboard", "Real-time Attendance", "Register Faces", 
                                   "Attendance Records", "Analytics"])

    if st.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.current_user = None
        st.rerun()

# Dashboard
if menu == "Dashboard":
    st.header("Admin Dashboard")
    
    daily_stats, person_stats, hourly_stats = get_attendance_stats()
    registered_faces = get_registered_faces()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Registered", len(registered_faces))
    with col2:
        today = datetime.now().date().isoformat()
        today_count = len(daily_stats[daily_stats['date'] == today]) if not daily_stats.empty else 0
        st.metric("Today's Attendance", today_count)
    with col3:
        st.metric("System Uptime", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"))
    
    st.subheader("Recent Activity")
    st.dataframe(get_attendance_data().head(10))
    
    st.subheader("Hourly Attendance Pattern")
    if not hourly_stats.empty:
        fig, ax = plt.subplots()
        ax.bar(hourly_stats['hour'], hourly_stats['count'])
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Attendance Count")
        st.pyplot(fig)
    else:
        st.warning("No attendance data available")

# Real-time Attendance
elif menu == "Real-time Attendance":
    st.header("Real-time Face Recognition Attendance")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        ctx = webrtc_streamer(
            key="recognition",
            video_transformer_factory=FaceRecognitionTransformer,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
        )
        
        if ctx.video_transformer:
            st.write("### Recent Recognitions")
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
        daily_stats, person_stats, _ = get_attendance_stats()
        
        st.metric("Total Registered", len(get_registered_faces()))
        today_count = len(daily_stats[daily_stats['date'] == datetime.now().date().isoformat()]) if not daily_stats.empty else 0
        st.metric("Today's Attendance", today_count)
        
        st.write("### Top Attendees")
        st.table(person_stats.head(5))

# Face Registration
elif menu == "Register Faces":
    st.header("Register New Faces")
    
    tab1, tab2 = st.tabs(["Webcam Registration", "Image Upload"])
    
    with tab1:
        st.write("Use your webcam to register a new face")
        name = st.text_input("Enter person's name (Webcam)")
        
        ctx = webrtc_streamer(
            key="registration",
            video_transformer_factory=FaceRegistrationTransformer,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
        )
        
        if ctx.video_transformer and name:
            if st.button("Capture and Register"):
                try:
                    frame = ctx.video_transformer.frame_queue.get()
                    if register_new_face(name, frame):
                        st.success(f"Successfully registered {name}!")
                    else:
                        st.error("Registration failed")
                except queue.Empty:
                    st.warning("No frame captured yet")
    
    with tab2:
        st.write("Upload an image to register a new face")
        name = st.text_input("Enter person's name (Upload)")
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file and name:
            image = Image.open(uploaded_file)
            image_array = np.array(image)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Register Face"):
                if register_new_face(name, image_array):
                    st.success(f"Successfully registered {name}!")
                else:
                    st.error("Registration failed")

# Attendance Records
elif menu == "Attendance Records":
    st.header("Attendance Records")
    
    df = get_attendance_data()
    
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
    
    st.download_button(
        "Export to CSV",
        df.to_csv(index=False),
        "attendance_records.csv",
        "text/csv"
    )

# Analytics
elif menu == "Analytics":
    st.header("Attendance Analytics")
    
    daily_stats, person_stats, hourly_stats = get_attendance_stats()
    
    tab1, tab2, tab3 = st.tabs(["Trends", "Individuals", "Hourly Patterns"])
    
    with tab1:
        st.subheader("Daily Attendance Trend")
        if not daily_stats.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(pd.to_datetime(daily_stats['date']), daily_stats['count'])
            ax.set_xlabel("Date")
            ax.set_ylabel("Count")
            st.pyplot(fig)
        else:
            st.warning("No data available")
    
    with tab2:
        st.subheader("Individual Attendance")
        if not person_stats.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(y='name', x='count', data=person_stats.head(20), ax=ax)
            ax.set_ylabel("Name")
            ax.set_xlabel("Days Attended")
            st.pyplot(fig)
        else:
            st.warning("No data available")
    
    with tab3:
        st.subheader("Hourly Patterns")
        if not hourly_stats.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(hourly_stats['hour'], hourly_stats['count'])
            ax.set_xlabel("Hour of Day")
            ax.set_ylabel("Attendance Count")
            st.pyplot(fig)
        else:
            st.warning("No data available")

# User Management (Admin only)
elif menu == "User Management" and st.session_state.current_user[3]:
    st.header("User Management")
    
    tab1, tab2 = st.tabs(["View Users", "Add New User"])
    
    with tab1:
        conn = sqlite3.connect(ATTENDANCE_DB)
        users = pd.read_sql("SELECT id, username, is_superadmin FROM admin_users", conn)
        conn.close()
        st.dataframe(users)
        
        # Delete user
        user_to_delete = st.selectbox("Select user to delete", 
                                    users[users['username'] != 'admin']['username'].tolist())
        if st.button("Delete User"):
            conn = sqlite3.connect(ATTENDANCE_DB)
            c = conn.cursor()
            c.execute("DELETE FROM admin_users WHERE username=?", (user_to_delete,))
            conn.commit()
            conn.close()
            st.success(f"User {user_to_delete} deleted")
            st.rerun()
    
    with tab2:
        with st.form("add_user_form"):
            new_username = st.text_input("Username")
            new_password = st.text_input("Password", type="password")
            is_admin = st.checkbox("Admin Privileges")
            
            if st.form_submit_button("Add User"):
                conn = sqlite3.connect(ATTENDANCE_DB)
                c = conn.cursor()
                try:
                    c.execute("INSERT INTO admin_users (username, password, is_superadmin) VALUES (?, ?, ?)",
                             (new_username, new_password, is_admin))
                    conn.commit()
                    st.success("User added successfully")
                except sqlite3.IntegrityError:
                    st.error("Username already exists")
                finally:
                    conn.close()

# Settings
elif menu == "Settings":
    st.header("System Settings")
    
    st.subheader("Recognition Settings")
    new_tolerance = st.slider("Recognition Threshold", 0.0, 1.0, st.session_state.tolerance, 0.05)
    if st.button("Save Recognition Settings"):
        st.session_state.tolerance = new_tolerance
        st.success("Settings saved")
    
    st.subheader("System Maintenance")
    if st.button("Clear Attendance Records"):
        conn = sqlite3.connect(ATTENDANCE_DB)
        c = conn.cursor()
        c.execute("DELETE FROM attendance")
        conn.commit()
        conn.close()
        st.success("Attendance records cleared")
    
    if st.button("Clear All Data (DANGER)"):
        st.warning("This will delete ALL data including registered faces!")
        if st.checkbox("I understand this cannot be undone"):
            conn = sqlite3.connect(ATTENDANCE_DB)
            c = conn.cursor()
            c.execute("DELETE FROM attendance")
            c.execute("DELETE FROM registered_faces")
            conn.commit()
            conn.close()
            
            for file in os.listdir(KNOWN_FACES_DIR):
                if file.endswith('.jpg'):
                    os.remove(os.path.join(KNOWN_FACES_DIR, file))
            
            st.success("All data has been reset")

# Custom CSS
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
    .st-emotion-cache-1v0mbdj {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

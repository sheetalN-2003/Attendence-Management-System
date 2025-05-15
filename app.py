import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import streamlit as st
import sqlite3
import time
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Constants (using uppercase naming convention)
TOLERANCE = 0.6  # Default value that can be modified via session state
KNOWN_FACES_DIR = 'known_faces'
ATTENDANCE_DB = 'attendance.db'
FRAME_THICKNESS = 2
FONT_THICKNESS = 1

# Create necessary directories
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

# Face recognition functions using OpenCV
def register_new_face(name, image_array):
    """Register a new face using basic OpenCV features"""
    conn = sqlite3.connect(ATTENDANCE_DB)
    c = conn.cursor()
    c.execute("SELECT name FROM registered_faces WHERE name=?", (name,))
    if c.fetchone():
        conn.close()
        return False
    
    try:
        # Convert to grayscale (simplest possible feature extraction)
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        
        # Save the image
        face_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
        cv2.imwrite(face_path, gray)
        
        # Save to database
        now = datetime.now()
        c.execute("INSERT INTO registered_faces (name, registration_date) VALUES (?, ?)", 
                  (name, now))
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Face registration failed: {str(e)}")
        return False
    finally:
        conn.close()

def recognize_face(image_array):
    """Simplified face recognition using template matching"""
    recognized_names = []
    
    # Convert input to grayscale
    gray_input = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    
    # Get current tolerance from session state or use default
    current_tolerance = st.session_state.get('tolerance', TOLERANCE)
    
    # Compare with all registered faces
    for face_file in os.listdir(KNOWN_FACES_DIR):
        if face_file.endswith('.jpg'):
            name = os.path.splitext(face_file)[0]
            face_path = os.path.join(KNOWN_FACES_DIR, face_file)
            registered_face = cv2.imread(face_path, cv2.IMREAD_GRAYSCALE)
            
            if registered_face is None:
                continue
                
            # Resize to same dimensions
            try:
                registered_face = cv2.resize(registered_face, (gray_input.shape[1], gray_input.shape[0]))
                
                # Simple template matching
                result = cv2.matchTemplate(gray_input, registered_face, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                
                if max_val > current_tolerance:
                    recognized_names.append(name)
            except:
                continue
    
    return recognized_names

def mark_attendance(name):
    """Mark attendance in database"""
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
    conn.close()
    return daily_stats, person_stats

# Streamlit UI
st.set_page_config(page_title="Attendance System", layout="wide")
st.title("Attendance Management System")

# Initialize session state for tolerance
if 'tolerance' not in st.session_state:
    st.session_state.tolerance = TOLERANCE

# Sidebar for navigation
menu = st.sidebar.selectbox("Menu", ["Home", "Register New Face", "Attendance Records", "Analytics", "Settings"])

if menu == "Home":
    st.header("Attendance System")
    
    uploaded_file = st.file_uploader("Upload an image to recognize faces", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Recognize Faces"):
            recognized_names = recognize_face(image_array)
            if recognized_names:
                st.success("Recognized faces: " + ", ".join(recognized_names))
                for name in recognized_names:
                    mark_attendance(name)
            else:
                st.warning("No recognized faces found")

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

elif menu == "Settings":
    st.header("System Settings")
    
    st.subheader("Recognition Parameters")
    new_tolerance = st.slider("Recognition Threshold (higher is stricter)", 
                            0.0, 1.0, st.session_state.tolerance, 0.05)
    
    if st.button("Save Settings"):
        st.session_state.tolerance = new_tolerance
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

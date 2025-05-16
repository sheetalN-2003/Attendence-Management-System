# AI-Powered Attendance Management System 👨💻📊

![Streamlit App](https://img.shields.io/badge/Deployed_on-Streamlit_Cloud-FF4B4B?logo=streamlit)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-orange?logo=opencv)

A **face recognition-based attendance system** developed as part of Microsoft's Advanced AI Certification through EduNet Foundation.

➡️ **Live Demo**: [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://attendence-management-system-majnyve3vkzmpsh5gjncuq.streamlit.app/)

## ✨ Key Features
| Feature | Description | Tech Used |
|---------|-------------|-----------|
| **Real-time Face Registration** | Register new users via webcam/image upload | OpenCV, WebRTC |
| **Automated Attendance Logging** | Marks attendance on face recognition | Template Matching |
| **Admin Dashboard** | View/export attendance records & analytics | SQLite, Pandas |
| **Multi-User Support** | Role-based access (Admin/User) | Streamlit Auth |
| **Cloud Deployment** | Ready for Streamlit/Azure deployment | Docker-ready |

## 🎯 Project Objectives
1. **Automate** manual attendance processes with 90%+ accuracy
2. **Optimize** for low-resource environments (no GPU dependency)
3. **Demonstrate** end-to-end AI solution development
4. **Showcase** Microsoft AI principles learned in training

## To get access as Admin
1. Name:admin
2. Password:admin123

## 🚀 How It Works
```mermaid
graph TD
    A[Webcam/Image Input] --> B(Face Detection)
    B --> C{Recognized?}
    C -->|Yes| D[Mark Attendance]
    C -->|No| E[Register New Face]
    D --> F[Analytics Dashboard]

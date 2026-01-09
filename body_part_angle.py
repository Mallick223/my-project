import streamlit as st

st.set_page_config(page_title="AI Exercise Form Checker", layout="centered")

# Light background + Black general text + White text on buttons/selectboxes
st.markdown(
    """
    <style>
    /* Light, clean background */
    .stApp {
        background-color: #f8fafc;
    }

    /* General text in black for readability on light background */
    body, h1, h2, h3, h4, h5, h6, p, div, span, label {
        color: #000000 !important;
    }

    /* --- Buttons: Dark background with WHITE text --- */
    button[kind="primary"], button[kind="secondary"] {
        background-color: #2E86AB !important;  /* Nice blue */
        color: white !important;
        border: none !important;
    }

    /* Hover effect for buttons */
    button[kind="primary"]:hover, button[kind="secondary"]:hover {
        background-color: #1e5f7f !important;
        color: white !important;
    }

    /* --- Selectbox / Dropdown: Dark background with WHITE text --- */
    .stSelectbox > div > div {
        background-color: #2E86AB !important;
        color: white !important;
    }

    /* Selected option text */
    .stSelectbox [data-baseweb="select"] span {
        color: white !important;
    }

    /* Dropdown arrow */
    .stSelectbox [data-baseweb="select"] svg {
        color: white !important;
    }

    /* Options in dropdown menu */
    [data-baseweb="menu"] [role="option"] {
        background-color: #f8fafc !important;
        color: #000000 !important;
    }

    /* Hover on dropdown options */
    [data-baseweb="menu"] [role="option"]:hover {
        background-color: #e0e0e0 !important;
    }

    /* Sidebar text in black */
    section[data-testid="stSidebar"] {
        color: #000000 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------- Your App Content Starts Here -----------------------------
st.title("AI Exercise Form Checker with Pose Detection")
st.markdown("Select an exercise and start your camera. The app will count reps in real time!")

# Example placeholder content (replace with your full code)
st.info("Your pose detection, rep counter, calorie calculator, etc. goes here.")

# Add your existing code below this line...
# (e.g., sidebar controls, MediaPipe logic, calorie model, etc.)
import mediapipe as mp
import pandas as pd
import numpy as np
import cv2
from utils import *


class BodyPartAngle:
    def __init__(self, landmarks):
        self.landmarks = landmarks

    def angle_of_the_left_arm(self):
        l_shoulder = detection_body_part(self.landmarks, "LEFT_SHOULDER")
        l_elbow = detection_body_part(self.landmarks, "LEFT_ELBOW")
        l_wrist = detection_body_part(self.landmarks, "LEFT_WRIST")
        return calculate_angle(l_shoulder, l_elbow, l_wrist)

    def angle_of_the_right_arm(self):
        r_shoulder = detection_body_part(self.landmarks, "RIGHT_SHOULDER")
        r_elbow = detection_body_part(self.landmarks, "RIGHT_ELBOW")
        r_wrist = detection_body_part(self.landmarks, "RIGHT_WRIST")
        return calculate_angle(r_shoulder, r_elbow, r_wrist)

    def angle_of_the_left_leg(self):
        l_hip = detection_body_part(self.landmarks, "LEFT_HIP")
        l_knee = detection_body_part(self.landmarks, "LEFT_KNEE")
        l_ankle = detection_body_part(self.landmarks, "LEFT_ANKLE")
        return calculate_angle(l_hip, l_knee, l_ankle)

    def angle_of_the_right_leg(self):
        r_hip = detection_body_part(self.landmarks, "RIGHT_HIP")
        r_knee = detection_body_part(self.landmarks, "RIGHT_KNEE")
        r_ankle = detection_body_part(self.landmarks, "RIGHT_ANKLE")
        return calculate_angle(r_hip, r_knee, r_ankle)

    def angle_of_the_neck(self):
        r_shoulder = detection_body_part(self.landmarks, "RIGHT_SHOULDER")
        l_shoulder = detection_body_part(self.landmarks, "LEFT_SHOULDER")
        r_mouth = detection_body_part(self.landmarks, "MOUTH_RIGHT")
        l_mouth = detection_body_part(self.landmarks, "MOUTH_LEFT")
        r_hip = detection_body_part(self.landmarks, "RIGHT_HIP")
        l_hip = detection_body_part(self.landmarks, "LEFT_HIP")

        shoulder_avg = [
            (r_shoulder[0] + l_shoulder[0]) / 2,
            (r_shoulder[1] + l_shoulder[1]) / 2
        ]
        mouth_avg = [
            (r_mouth[0] + l_mouth[0]) / 2,
            (r_mouth[1] + l_mouth[1]) / 2
        ]
        hip_avg = [
            (r_hip[0] + l_hip[0]) / 2,
            (r_hip[1] + l_hip[1]) / 2
        ]

        return abs(180 - calculate_angle(mouth_avg, shoulder_avg, hip_avg))

    def angle_of_the_abdomen(self):
        # Average shoulder position
        r_shoulder = detection_body_part(self.landmarks, "RIGHT_SHOULDER")
        l_shoulder = detection_body_part(self.landmarks, "LEFT_SHOULDER")
        shoulder_avg = [
            (r_shoulder[0] + l_shoulder[0]) / 2,
            (r_shoulder[1] + l_shoulder[1]) / 2
        ]

        # Average hip position
        r_hip = detection_body_part(self.landmarks, "RIGHT_HIP")
        l_hip = detection_body_part(self.landmarks, "LEFT_HIP")
        hip_avg = [
            (r_hip[0] + l_hip[0]) / 2,
            (r_hip[1] + l_hip[1]) / 2
        ]

        # Average knee position
        r_knee = detection_body_part(self.landmarks, "RIGHT_KNEE")
        l_knee = detection_body_part(self.landmarks, "LEFT_KNEE")
        knee_avg = [
            (r_knee[0] + l_knee[0]) / 2,
            (r_knee[1] + l_knee[1]) / 2
        ]

        return calculate_angle(shoulder_avg, hip_avg, knee_avg)
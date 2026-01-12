import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from utils import score_table
from types_of_exercise import TypeOfExercise
import joblib
import pandas as pd

# ----------------------------- Page Config -----------------------------
st.set_page_config(page_title="AI Fitness Coach Pro", layout="wide")
st.title("💪 AI Fitness Coach Pro")
st.markdown("**Real-time form correction + rep counting + calorie prediction & personalized plans**")

# ----------------------------- Load Calorie Model -----------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("calorie_model.pkl")
        return model
    except Exception as e:
        st.error(f"Failed to load calorie model: {e}")
        return None

model = load_model()

# ----------------------------- MediaPipe Setup -----------------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ----------------------------- Tabs -----------------------------
tab1, tab2 = st.tabs(["📹 Real-Time Form Checker", "🔥 Calorie & Plan Calculator"])

# ----------------------------- Tab 1: Pose Detection & Rep Counting -----------------------------
with tab1:
    st.subheader("Real-Time Exercise Form Checker")

    # Sidebar Controls (shared)
    st.sidebar.header("Pose Detection Settings")

    exercise_options = {
        "push-up": "pushup",
        "squat": "squat",
        "pull-up": "pullup",
        "sit-up": "sit-up",
        # Add more as needed
    }

    exercise_name = st.sidebar.selectbox(
        "Choose exercise",
        options=list(exercise_options.keys()),
        format_func=lambda x: x.replace("-", " ").title()
    )
    exercise_type = exercise_options[exercise_name]

    video_source = st.sidebar.radio(
        "Video source",
        options=["Webcam (0)", "Upload video file"],
    )

    if video_source == "Upload video file":
        uploaded_file = st.sidebar.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            import tempfile
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            video_path = tfile.name
        else:
            video_path = None
    else:
        video_path = 0  # webcam

    start_button = st.sidebar.button("Start Detection")
    stop_button = st.sidebar.button("Stop Detection")

    # Session State
    if "running" not in st.session_state:
        st.session_state.running = False

    if stop_button:
        st.session_state.running = False

    frame_placeholder = st.empty()
    counter_placeholder = st.empty()
    status_placeholder = st.empty()

    if start_button or st.session_state.running:
        st.session_state.running = True

        if video_path is None and video_source == "Upload video file":
            st.warning("Please upload a video first.")
        else:
            cap = cv2.VideoCapture(video_path if video_path != 0 else 0)

            if not cap.isOpened():
                st.error("Cannot open video source. Check your webcam or file.")
            else:
                width, height = 800, 480
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

                counter = 0
                status = True

                with mp_pose.Pose(min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5) as pose:

                    st.info("Detection started! Press **Stop Detection** to end.")
                    while st.session_state.running and cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            st.warning("End of video or camera disconnected.")
                            break

                        frame = cv2.resize(frame, (width, height))
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        rgb_frame.flags.writeable = False
                        results = pose.process(rgb_frame)
                        rgb_frame.flags.writeable = True
                        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

                        try:
                            landmarks = results.pose_landmarks.landmark
                            counter, status = TypeOfExercise(landmarks).calculate_exercise(
                                exercise_type, counter, status
                            )
                        except Exception:
                            pass

                        frame = score_table(exercise_type, frame, counter, status)

                        if results.pose_landmarks:
                            mp_drawing.draw_landmarks(
                                frame,
                                results.pose_landmarks,
                                mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                                mp_drawing.DrawingSpec(color=(174, 139, 45), thickness=2, circle_radius=2),
                            )

                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                        counter_placeholder.metric(label=f"{exercise_name.replace('-', ' ').title()} Count", value=counter)
                        status_text = "Up" if status else "Down"
                        status_placeholder.write(f"**Current Position:** {status_text}")

                    cap.release()
                    st.session_state.running = False
                    st.success("Detection stopped.")

    else:
        st.info("👈 Choose exercise and click **Start Detection** to begin.")
        st.image("https://via.placeholder.com/800x480/333333/FFFFFF?text=Camera+feed+will+appear+here", 
                 caption="Preview area")

# ----------------------------- Tab 2: Calorie Calculator & Plan -----------------------------
with tab2:
    st.subheader("Calorie Burn Predictor + Personalized Workout & Diet Plan")

    if model is None:
        st.error("Calorie prediction model failed to load. Please check 'calorie_model.pkl'.")
    else:
        st.markdown("Enter your details to get estimated calories burned and a tailored plan.")

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age", min_value=10, max_value=100, value=30)

        with col2:
            weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)

        with col3:
            duration = st.number_input("Duration (minutes)", min_value=5, max_value=180, value=30)

        goal = st.selectbox(
            "Your Goal",
            options=["weight_loss", "muscle_gain", "general_fitness"],
            format_func=lambda x: x.replace("_", " ").title()
        )

        if st.button("Calculate Calories & Get Plan"):
            if age and weight and duration:
                calories = model.predict([[age, weight, duration]])[0]

                if goal == "weight_loss":
                    workout = "Cardio + Yoga"
                    diet = "Low Carb, High Protein"
                elif goal == "muscle_gain":
                    workout = "Strength Training"
                    diet = "High Protein, High Calories"
                else:
                    workout = "Mixed Workout"
                    diet = "Balanced Diet"

                st.success(f"**Estimated Calories Burned:** {round(calories, 2)} kcal")

                st.markdown("### Recommended Plan")
                st.info(f"**Workout:** {workout}")
                st.info(f"**Diet:** {diet}")
            else:
                st.warning("Please fill in all fields.")
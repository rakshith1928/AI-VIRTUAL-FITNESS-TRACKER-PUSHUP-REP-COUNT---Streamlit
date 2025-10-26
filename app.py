import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

st.set_page_config(page_title="💪 Push-Up Counter", layout="wide")
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>💪 Push-Up Counter</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Track your push-ups in real-time!</p>", unsafe_allow_html=True)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# --- Session State ---
if 'running' not in st.session_state:
    st.session_state.running = False
if 'counter' not in st.session_state:
    st.session_state.counter = 0
if 'stage' not in st.session_state:
    st.session_state.stage = None

FRAME_WINDOW = st.image([])
reps_placeholder = st.empty()

# --- Centered Buttons ---
button_col1, button_col2, button_col3 = st.columns([1,2,1])
with button_col2:
    start_btn = st.button("Start")
    stop_btn = st.button("Stop")

# --- Start Push-Up Counter ---
if start_btn:
    st.session_state.running = True
    st.session_state.counter = 0
    st.session_state.stage = None

if stop_btn:
    st.session_state.running = False
    reps_placeholder.markdown(
        f"<h2 style='text-align:center; color:#00FF00;'>✅ Total Push-Ups: {st.session_state.counter}</h2>",
        unsafe_allow_html=True
    )

if st.session_state.running:
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                st.warning("Webcam not detected!")
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark

                # Left side
                l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                # Right side
                r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                # Angles
                left_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
                right_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
                avg_angle = (left_angle + right_angle) / 2

                # Push-up logic
                if avg_angle < 70:
                    st.session_state.stage = "down"
                if avg_angle > 160 and st.session_state.stage == "down":
                    st.session_state.stage = "up"
                    st.session_state.counter += 1

                # Overlay
                cv2.rectangle(image, (0, 0), (300, 90), (0, 123, 255), -1)
                cv2.putText(image, f'REPS: {st.session_state.counter}', (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                cv2.putText(image, f'STAGE: {st.session_state.stage}', (10, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            except:
                pass

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            FRAME_WINDOW.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    cap.release()
    cv2.destroyAllWindows()







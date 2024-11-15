import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
from PIL import Image

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def process_video(video_file):
    # Create a temporary file to store the uploaded video
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    
    # Open the video file
    cap = cv2.VideoCapture(tfile.name)
    
    feedback = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert the BGR image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe Pose
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            # Draw pose landmarks
            mp_drawing.draw_landmarks(
                frame_rgb,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )
            
            # Analyze form (simplified version)
            if frame_count % 10 == 0:  # Analyze every 10th frame
                feedback.extend(analyze_form(results.pose_landmarks))
        
        frame_count += 1
        
        # Display the frame
        st.image(frame_rgb, channels="RGB", use_column_width=True)
        
    cap.release()
    return feedback

def analyze_form(landmarks):
    feedback = []
    
    # Get key landmarks
    hip_l = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    hip_r = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    knee_l = landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
    knee_r = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
    ankle_l = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
    ankle_r = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
    shoulder_l = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    shoulder_r = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    
    # Basic form checks
    
    # Check knee alignment
    knee_hip_dist_l = abs(knee_l.x - hip_l.x)
    knee_hip_dist_r = abs(knee_r.x - hip_r.x)
    if knee_hip_dist_l > 0.2 or knee_hip_dist_r > 0.2:
        feedback.append("Keep knees in line with hips")
    
    # Check squat depth
    hip_height = (hip_l.y + hip_r.y) / 2
    knee_height = (knee_l.y + knee_r.y) / 2
    if hip_height < knee_height:
        feedback.append("Increase squat depth")
    
    # Check back angle
    back_angle = calculate_angle(
        (shoulder_l.x, shoulder_l.y),
        ((hip_l.x + hip_r.x)/2, (hip_l.y + hip_r.y)/2),
        (hip_l.x, hip_l.y + 0.5)
    )
    if back_angle < 45:
        feedback.append("Keep chest up")
    
    return list(set(feedback))  # Remove duplicates

def calculate_angle(p1, p2, p3):
    """Calculate the angle between three points"""
    a = np.array([p1[0], p1[1]])
    b = np.array([p2[0], p2[1]])
    c = np.array([p3[0], p3[1]])
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)

# Streamlit UI
st.set_page_config(page_title="CrossFit Movement Analyzer", layout="wide")

st.title("CrossFit Movement Analyzer")
st.write("Upload a video of your CrossFit movement for real-time analysis!")

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    st.write("Analyzing your movement...")
    feedback = process_video(uploaded_file)
    
    if feedback:
        st.subheader("Form Feedback:")
        for item in feedback:
            st.write(f"• {item}")
    else:
        st.write("No major form issues detected!")

from flask import Flask, render_template, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

class MovementMetrics:
    """Class to store and analyze movement metrics."""
    def __init__(self):
        self.angles = {}  # Store joint angles
        self.velocities = {}  # Store movement velocities
        self.positions = {}  # Store key positions
        self.timing = {}  # Store phase timing
        self.symmetry = {}  # Store symmetry metrics
        self.power_metrics = {}  # Store power/explosiveness metrics

def analyze_movement(video_path):
    """Analyze the CrossFit movement from video."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    pose_landmarks = []
    metrics = MovementMetrics()
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    prev_landmarks = None
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and detect poses
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            pose_landmarks.append(results.pose_landmarks)
            
            # Calculate metrics if we have previous landmarks
            if prev_landmarks:
                calculate_frame_metrics(prev_landmarks, results.pose_landmarks, metrics, 1/fps)
            
            prev_landmarks = results.pose_landmarks
            frame_count += 1
            
        frames.append(frame)
    
    cap.release()
    
    # Analyze form based on pose landmarks and metrics
    feedback = analyze_form_with_metrics(pose_landmarks, metrics)
    return feedback, metrics

def calculate_frame_metrics(prev_landmarks, curr_landmarks, metrics, time_delta):
    """Calculate metrics between consecutive frames."""
    # Calculate joint angles
    metrics.angles['hips'] = calculate_hip_angle(curr_landmarks)
    metrics.angles['knees'] = calculate_knee_angle(curr_landmarks)
    metrics.angles['ankles'] = calculate_ankle_angle(curr_landmarks)
    metrics.angles['shoulders'] = calculate_shoulder_angle(curr_landmarks)
    
    # Calculate velocities
    hip_velocity = calculate_joint_velocity(
        prev_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP],
        curr_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP],
        time_delta
    )
    metrics.velocities['hip'] = hip_velocity
    
    # Calculate power metrics (rate of change in height)
    metrics.power_metrics['hip_power'] = calculate_power_metric(
        prev_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP],
        curr_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP],
        time_delta
    )
    
    # Calculate symmetry
    metrics.symmetry['knee_symmetry'] = calculate_knee_symmetry(curr_landmarks)
    metrics.symmetry['hip_symmetry'] = calculate_hip_symmetry(curr_landmarks)

def analyze_form_with_metrics(pose_landmarks, metrics):
    """Analyze the form based on pose landmarks and calculated metrics."""
    feedback = []
    
    if not pose_landmarks:
        return ["No pose detected in the video"]
    
    # Detect movement type based on pose patterns
    movement_type = detect_movement_type(pose_landmarks)
    
    if movement_type == "clean_and_jerk":
        feedback.extend(analyze_clean_and_jerk_with_metrics(pose_landmarks, metrics))
    elif movement_type == "squat_clean":
        feedback.extend(analyze_squat_clean_with_metrics(pose_landmarks, metrics))
    elif movement_type == "squat":
        feedback.extend(analyze_squat_with_metrics(pose_landmarks, metrics))
    
    return list(set(feedback))

def analyze_squat_clean_with_metrics(pose_landmarks, metrics):
    """Enhanced squat clean analysis with detailed metrics."""
    feedback = []
    
    # First Pull Analysis
    first_pull_feedback, first_pull_metrics = analyze_first_pull_metrics(pose_landmarks[:len(pose_landmarks)//3], metrics)
    feedback.extend(first_pull_feedback)
    
    # Add specific metrics to feedback
    if 'max_back_angle' in first_pull_metrics:
        feedback.append(f"Maximum back angle during first pull: {first_pull_metrics['max_back_angle']:.1f}° "
                       f"({'good' if first_pull_metrics['max_back_angle'] > 45 else 'needs improvement'})")
    
    if 'bar_path_deviation' in first_pull_metrics:
        feedback.append(f"Bar path deviation: {first_pull_metrics['bar_path_deviation']:.2f}cm "
                       f"({'good' if first_pull_metrics['bar_path_deviation'] < 5 else 'needs improvement'})")
    
    # Transition Phase Analysis
    transition_feedback, transition_metrics = analyze_transition_metrics(
        pose_landmarks[len(pose_landmarks)//3:2*len(pose_landmarks)//3],
        metrics
    )
    feedback.extend(transition_feedback)
    
    if 'hip_power' in transition_metrics:
        feedback.append(f"Hip drive power: {transition_metrics['hip_power']:.1f} m/s² "
                       f"({'explosive' if transition_metrics['hip_power'] > 2.0 else 'needs more explosion'})")
    
    # Receiving Position Analysis
    receiving_feedback, receiving_metrics = analyze_receiving_metrics(
        pose_landmarks[2*len(pose_landmarks)//3:],
        metrics
    )
    feedback.extend(receiving_feedback)
    
    if 'squat_depth' in receiving_metrics:
        feedback.append(f"Squat depth: {receiving_metrics['squat_depth']:.1f}° "
                       f"({'good' if receiving_metrics['squat_depth'] > 90 else 'needs more depth'})")
    
    return feedback

def analyze_first_pull_metrics(pose_landmarks, metrics):
    """Detailed analysis of first pull with metrics."""
    feedback = []
    pull_metrics = {}
    
    max_back_angle = -float('inf')
    max_bar_path_deviation = -float('inf')
    
    for landmarks in pose_landmarks:
        # Back angle analysis
        back_angle = calculate_back_angle(
            landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER],
            landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER],
            landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP],
            landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        )
        max_back_angle = max(max_back_angle, back_angle)
        
        # Bar path analysis
        deviation = calculate_bar_path_deviation(landmarks)
        max_bar_path_deviation = max(max_bar_path_deviation, deviation)
    
    pull_metrics['max_back_angle'] = max_back_angle
    pull_metrics['bar_path_deviation'] = max_bar_path_deviation
    
    # Generate feedback based on metrics
    if max_back_angle < 45:
        feedback.append(f"Back angle too horizontal ({max_back_angle:.1f}°). Aim for >45°")
    
    if max_bar_path_deviation > 5:  # 5cm threshold
        feedback.append(f"Bar path deviating {max_bar_path_deviation:.1f}cm from vertical. Keep bar close")
    
    return feedback, pull_metrics

def analyze_transition_metrics(pose_landmarks, metrics):
    """Detailed analysis of transition phase with metrics."""
    feedback = []
    transition_metrics = {}
    
    max_hip_power = -float('inf')
    min_arm_angle = float('inf')
    
    for i in range(1, len(pose_landmarks)):
        # Calculate hip power
        hip_power = calculate_hip_power(pose_landmarks[i-1], pose_landmarks[i])
        max_hip_power = max(max_hip_power, hip_power)
        
        # Calculate arm angle
        arm_angle = calculate_arm_angle(pose_landmarks[i])
        min_arm_angle = min(min_arm_angle, arm_angle)
    
    transition_metrics['hip_power'] = max_hip_power
    transition_metrics['min_arm_angle'] = min_arm_angle
    
    # Generate feedback based on metrics
    if max_hip_power < 2.0:  # Threshold for explosive movement
        feedback.append(f"Hip drive power: {max_hip_power:.1f} m/s². More explosion needed")
    
    if min_arm_angle < 160:  # Arms should stay straight
        feedback.append(f"Arms bending too early ({min_arm_angle:.1f}°). Keep arms straight longer")
    
    return feedback, transition_metrics

def analyze_receiving_metrics(pose_landmarks, metrics):
    """Detailed analysis of receiving position with metrics."""
    feedback = []
    receiving_metrics = {}
    
    # Find the lowest position
    lowest_pos = find_lowest_squat_position(pose_landmarks)
    
    if lowest_pos:
        # Calculate squat depth
        squat_depth = calculate_squat_depth(lowest_pos)
        receiving_metrics['squat_depth'] = squat_depth
        
        # Calculate front rack position metrics
        rack_angle = calculate_front_rack_angle(lowest_pos)
        receiving_metrics['rack_angle'] = rack_angle
        
        # Calculate knee tracking
        knee_tracking = calculate_knee_tracking(lowest_pos)
        receiving_metrics['knee_tracking'] = knee_tracking
        
        # Generate feedback based on metrics
        if squat_depth < 90:  # Threshold for proper depth
            feedback.append(f"Squat depth: {squat_depth:.1f}°. Need more depth (aim for >90°)")
        
        if rack_angle < 80:  # Threshold for proper rack position
            feedback.append(f"Front rack position: {rack_angle:.1f}°. Elbows need to be higher")
        
        if abs(knee_tracking) > 0.1:  # Threshold for knee alignment
            feedback.append(f"Knee tracking off by {abs(knee_tracking)*100:.1f}%. Keep knees in line with toes")
    
    return feedback, receiving_metrics

def calculate_hip_power(prev_landmarks, curr_landmarks):
    """Calculate hip power (velocity * force approximation)."""
    prev_hip = prev_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    curr_hip = curr_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    
    # Calculate vertical velocity
    velocity = (curr_hip.y - prev_hip.y) * 100  # Convert to cm/s
    
    # Approximate force using displacement
    force = abs(velocity)  # Simple approximation
    
    return velocity * force

def calculate_arm_angle(landmarks):
    """Calculate angle of arms relative to vertical."""
    shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    elbow = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
    wrist = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    
    return calculate_angle(shoulder, elbow, wrist)

def calculate_squat_depth(landmarks):
    """Calculate squat depth angle."""
    hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    knee = landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
    ankle = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
    
    return calculate_angle(hip, knee, ankle)

def calculate_front_rack_angle(landmarks):
    """Calculate front rack position angle."""
    shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    elbow = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
    wrist = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    
    return calculate_angle(shoulder, elbow, wrist)

def calculate_knee_tracking(landmarks):
    """Calculate knee tracking relative to toe alignment."""
    knee = landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
    ankle = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
    hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    
    # Calculate deviation from vertical line
    return (knee.x - ankle.x) - (hip.x - ankle.x)

def calculate_bar_path_deviation(landmarks):
    """Calculate deviation of bar path from vertical line."""
    wrist_l = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    wrist_r = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    ankle_l = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
    ankle_r = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
    
    # Calculate horizontal distance from mid-ankle
    wrist_mid_x = (wrist_l.x + wrist_r.x) / 2
    ankle_mid_x = (ankle_l.x + ankle_r.x) / 2
    
    return abs(wrist_mid_x - ankle_mid_x) * 100  # Convert to cm

def calculate_joint_velocity(prev_joint, curr_joint, time_delta):
    """Calculate velocity of a joint between frames."""
    displacement = np.sqrt(
        (curr_joint.x - prev_joint.x)**2 +
        (curr_joint.y - prev_joint.y)**2
    )
    return displacement / time_delta

def calculate_power_metric(prev_joint, curr_joint, time_delta):
    """Calculate power metric based on joint movement."""
    velocity = calculate_joint_velocity(prev_joint, curr_joint, time_delta)
    # Approximate force using velocity (simplified model)
    force = abs(velocity)
    return velocity * force

def calculate_knee_symmetry(landmarks):
    """Calculate symmetry between left and right knees."""
    knee_l = landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
    knee_r = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
    hip_l = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    hip_r = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    
    left_angle = calculate_angle(hip_l, knee_l, landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE])
    right_angle = calculate_angle(hip_r, knee_r, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE])
    
    return abs(left_angle - right_angle)

def calculate_hip_symmetry(landmarks):
    """Calculate symmetry between left and right hips."""
    hip_l = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    hip_r = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    
    return abs(hip_l.y - hip_r.y)

def detect_movement_type(pose_landmarks):
    """Detect the type of CrossFit movement based on pose patterns."""
    # Analyze the first few frames to determine movement type
    initial_poses = pose_landmarks[:10]  # Look at first 10 frames
    
    # Check for characteristic starting positions
    if is_clean_and_jerk_position(initial_poses):
        # Further differentiate between clean and jerk and squat clean
        if is_squat_clean_pattern(pose_landmarks):
            return "squat_clean"
        return "clean_and_jerk"
    
    # Default to squat if no other movement is detected
    return "squat"

def is_clean_and_jerk_position(poses):
    """Check if the poses match clean and jerk starting position."""
    if not poses:
        return False
        
    # Check first frame for characteristic starting position
    first_pose = poses[0]
    
    # Get relevant landmarks
    hip_l = first_pose.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    hip_r = first_pose.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    knee_l = first_pose.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
    knee_r = first_pose.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
    ankle_l = first_pose.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
    ankle_r = first_pose.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
    
    # Check if in starting position (hips above knees, hands near shins)
    hips_above_knees = (hip_l.y < knee_l.y) and (hip_r.y < knee_r.y)
    knees_above_ankles = (knee_l.y < ankle_l.y) and (knee_r.y < ankle_r.y)
    
    return hips_above_knees and knees_above_ankles

def is_squat_clean_pattern(pose_landmarks):
    """Detect if the movement pattern matches a squat clean."""
    if len(pose_landmarks) < 20:  # Need enough frames to detect pattern
        return False
        
    # Look for characteristic deep squat receiving position
    mid_movement = pose_landmarks[len(pose_landmarks)//2]  # Check middle of movement
    
    hip_l = mid_movement.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    hip_r = mid_movement.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    knee_l = mid_movement.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
    knee_r = mid_movement.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
    ankle_l = mid_movement.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
    ankle_r = mid_movement.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
    
    # Check for deep squat position (hips below knees)
    hips_below_knees = (hip_l.y > knee_l.y) and (hip_r.y > knee_r.y)
    knees_forward = (knee_l.x > ankle_l.x) and (knee_r.x > ankle_r.x)
    
    return hips_below_knees and knees_forward

def calculate_back_angle(shoulder_l, shoulder_r, hip_l, hip_r):
    """Calculate the angle of the back relative to horizontal."""
    # Calculate midpoints
    shoulder_mid_y = (shoulder_l.y + shoulder_r.y) / 2
    shoulder_mid_x = (shoulder_l.x + shoulder_r.x) / 2
    hip_mid_y = (hip_l.y + hip_r.y) / 2
    hip_mid_x = (hip_l.x + hip_r.x) / 2
    
    # Calculate angle using arctangent
    return abs(np.degrees(np.arctan2(shoulder_mid_y - hip_mid_y, shoulder_mid_x - hip_mid_x)))

def is_knees_caving(knee_l, knee_r, hip_l, hip_r):
    """Check if knees are caving inward."""
    knee_width = abs(knee_l.x - knee_r.x)
    hip_width = abs(hip_l.x - hip_r.x)
    return knee_width < hip_width * 0.7  # Knees should track at least 70% of hip width

def is_full_extension(ankle_l, ankle_r, hip_l, hip_r):
    """Check for full extension in the second pull."""
    left_extension = hip_l.y - ankle_l.y
    right_extension = hip_r.y - ankle_r.y
    return left_extension < 0.1 and right_extension < 0.1  # Small value indicates full extension

def is_elbows_up(elbow_l, elbow_r, shoulder_l, shoulder_r):
    """Check if elbows are positioned correctly in the catch."""
    return (elbow_l.y < shoulder_l.y) and (elbow_r.y < shoulder_r.y)

def is_overhead_lockout(shoulder_l, shoulder_r, wrist_l, wrist_r):
    """Check for proper overhead lockout position."""
    left_lockout = abs(wrist_l.x - shoulder_l.x) < 0.1
    right_lockout = abs(wrist_r.x - shoulder_r.x) < 0.1
    return left_lockout and right_lockout

def analyze_clean_and_jerk(pose_landmarks):
    """Analyze clean and jerk form and provide feedback."""
    feedback = []
    
    # Analyze different phases of clean and jerk
    first_pull_feedback = analyze_first_pull(pose_landmarks)
    transition_feedback = analyze_transition(pose_landmarks)
    second_pull_feedback = analyze_second_pull(pose_landmarks)
    catch_feedback = analyze_catch(pose_landmarks)
    jerk_feedback = analyze_jerk(pose_landmarks)
    
    feedback.extend(first_pull_feedback)
    feedback.extend(transition_feedback)
    feedback.extend(second_pull_feedback)
    feedback.extend(catch_feedback)
    feedback.extend(jerk_feedback)
    
    return feedback

def analyze_first_pull(pose_landmarks):
    """Analyze the first pull phase of clean and jerk."""
    feedback = []
    
    for i, landmarks in enumerate(pose_landmarks):
        # Get relevant landmarks
        shoulder_l = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        shoulder_r = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        hip_l = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        hip_r = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        knee_l = landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
        knee_r = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
        
        # Check back angle
        back_angle = calculate_back_angle(shoulder_l, shoulder_r, hip_l, hip_r)
        if back_angle < 45:  # If back is too horizontal
            feedback.append("Keep your chest up during the first pull")
            
        # Check if knees are caving in
        if is_knees_caving(knee_l, knee_r, hip_l, hip_r):
            feedback.append("Keep your knees tracking over your toes")
            
    return feedback

def analyze_transition(pose_landmarks):
    """Analyze the transition phase (bar passing knees)."""
    feedback = []
    
    for landmarks in pose_landmarks:
        hip_l = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        hip_r = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        knee_l = landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
        knee_r = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
        
        # Check if hips rise too early
        if (hip_l.y < knee_l.y - 0.1) and (hip_r.y < knee_r.y - 0.1):
            feedback.append("Keep your hips down during the transition")
            
    return feedback

def analyze_second_pull(pose_landmarks):
    """Analyze the second pull (explosion) phase."""
    feedback = []
    
    for landmarks in pose_landmarks:
        ankle_l = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        ankle_r = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
        hip_l = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        hip_r = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        
        # Check for full extension
        if not is_full_extension(ankle_l, ankle_r, hip_l, hip_r):
            feedback.append("Achieve full extension in your hips during the second pull")
            
    return feedback

def analyze_catch(pose_landmarks):
    """Analyze the catch phase."""
    feedback = []
    
    for landmarks in pose_landmarks:
        elbow_l = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        elbow_r = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        shoulder_l = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        shoulder_r = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        # Check elbow position in catch
        if not is_elbows_up(elbow_l, elbow_r, shoulder_l, shoulder_r):
            feedback.append("Keep your elbows up in the catch position")
            
    return feedback

def analyze_jerk(pose_landmarks):
    """Analyze the jerk portion of the movement."""
    feedback = []
    
    for landmarks in pose_landmarks:
        shoulder_l = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        shoulder_r = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        wrist_l = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        wrist_r = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        
        # Check for lockout
        if not is_overhead_lockout(shoulder_l, shoulder_r, wrist_l, wrist_r):
            feedback.append("Achieve full lockout overhead in the jerk")
            
    return feedback

def analyze_squat(pose_landmarks):
    """Analyze squat form."""
    feedback = []
    
    for frame_landmarks in pose_landmarks:
        hip_l = frame_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        hip_r = frame_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        knee_l = frame_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
        knee_r = frame_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
        
        # Check if hips are level
        if abs(hip_l.y - hip_r.y) > 0.1:
            feedback.append("Keep your hips level throughout the squat")
            
        # Check knee alignment
        if abs(knee_l.x - hip_l.x) > 0.1 or abs(knee_r.x - hip_r.x) > 0.1:
            feedback.append("Keep your knees aligned with your hips")
            
    return feedback

def analyze_squat_clean(pose_landmarks):
    """Analyze squat clean form and provide feedback."""
    feedback = []
    
    # Analyze different phases of squat clean
    first_pull_feedback = analyze_squat_clean_first_pull(pose_landmarks)
    transition_feedback = analyze_squat_clean_transition(pose_landmarks)
    receiving_feedback = analyze_squat_clean_receiving(pose_landmarks)
    recovery_feedback = analyze_squat_clean_recovery(pose_landmarks)
    
    feedback.extend(first_pull_feedback)
    feedback.extend(transition_feedback)
    feedback.extend(receiving_feedback)
    feedback.extend(recovery_feedback)
    
    return feedback

def analyze_squat_clean_first_pull(pose_landmarks):
    """Analyze the first pull phase of squat clean."""
    feedback = []
    
    for landmarks in pose_landmarks[:len(pose_landmarks)//3]:  # First third of movement
        shoulder_l = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        shoulder_r = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        hip_l = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        hip_r = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        knee_l = landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
        knee_r = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
        ankle_l = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        ankle_r = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
        
        # Check starting position
        if not is_proper_start_position(shoulder_l, shoulder_r, hip_l, hip_r, knee_l, knee_r, ankle_l, ankle_r):
            feedback.append("Start with shoulders slightly ahead of the bar, hips higher than knees")
        
        # Check back angle
        back_angle = calculate_back_angle(shoulder_l, shoulder_r, hip_l, hip_r)
        if back_angle < 45:
            feedback.append("Maintain a strong back angle during the first pull")
        
        # Check bar path (approximated by wrist position)
        if not is_vertical_bar_path(landmarks):
            feedback.append("Keep the bar close to your body during the pull")
    
    return feedback

def analyze_squat_clean_transition(pose_landmarks):
    """Analyze the transition phase of squat clean."""
    feedback = []
    
    mid_point = len(pose_landmarks)//2
    transition_phase = pose_landmarks[mid_point-5:mid_point+5]  # Analyze around the middle of the movement
    
    for landmarks in transition_phase:
        hip_l = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        hip_r = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        knee_l = landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
        knee_r = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
        
        # Check for early arm bend
        if is_early_arm_bend(landmarks):
            feedback.append("Keep arms straight until the second pull")
        
        # Check hip position
        if not is_proper_hip_position(hip_l, hip_r, knee_l, knee_r):
            feedback.append("Stay over the bar longer before transitioning to the squat")
    
    return feedback

def analyze_squat_clean_receiving(pose_landmarks):
    """Analyze the receiving position in squat clean."""
    feedback = []
    
    # Find the lowest position in the movement
    lowest_position = find_lowest_squat_position(pose_landmarks)
    
    if lowest_position:
        shoulder_l = lowest_position.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        shoulder_r = lowest_position.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        hip_l = lowest_position.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        hip_r = lowest_position.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        knee_l = lowest_position.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
        knee_r = lowest_position.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
        ankle_l = lowest_position.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        ankle_r = lowest_position.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
        
        # Check depth
        if not is_full_squat_depth(hip_l, hip_r, knee_l, knee_r):
            feedback.append("Achieve full depth in the receiving position")
        
        # Check torso angle
        if not is_upright_torso(shoulder_l, shoulder_r, hip_l, hip_r):
            feedback.append("Stay more upright in the receiving position")
        
        # Check front rack position
        if not is_proper_front_rack(lowest_position):
            feedback.append("Keep elbows high in the front rack position")
    
    return feedback

def analyze_squat_clean_recovery(pose_landmarks):
    """Analyze the recovery phase of squat clean."""
    feedback = []
    
    # Analyze the final third of the movement
    recovery_phase = pose_landmarks[-(len(pose_landmarks)//3):]
    
    for landmarks in recovery_phase:
        hip_l = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        hip_r = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        knee_l = landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
        knee_r = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
        
        # Check if knees are caving during stand up
        if is_knees_caving(knee_l, knee_r, hip_l, hip_r):
            feedback.append("Keep knees out during the stand up")
        
        # Check if hips rise faster than shoulders
        if is_hips_rising_fast(landmarks):
            feedback.append("Lead with the chest during the recovery")
    
    return feedback

def is_proper_start_position(shoulder_l, shoulder_r, hip_l, hip_r, knee_l, knee_r, ankle_l, ankle_r):
    """Check if the starting position is correct for squat clean."""
    shoulders_over_bar = (shoulder_l.x > ankle_l.x) and (shoulder_r.x > ankle_r.x)
    hips_above_knees = (hip_l.y < knee_l.y) and (hip_r.y < knee_r.y)
    return shoulders_over_bar and hips_above_knees

def is_vertical_bar_path(landmarks):
    """Check if the bar path (approximated by wrists) is vertical."""
    wrist_l = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    wrist_r = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    shoulder_l = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    shoulder_r = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    
    wrist_distance = abs((wrist_l.x + wrist_r.x)/2 - (shoulder_l.x + shoulder_r.x)/2)
    return wrist_distance < 0.15  # Threshold for vertical path

def is_early_arm_bend(landmarks):
    """Check for early arm bend in the pull."""
    elbow_l = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
    elbow_r = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
    shoulder_l = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    shoulder_r = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    wrist_l = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    wrist_r = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    
    # Check if elbows are bent by measuring the angle
    left_arm_angle = calculate_angle(shoulder_l, elbow_l, wrist_l)
    right_arm_angle = calculate_angle(shoulder_r, elbow_r, wrist_r)
    
    return left_arm_angle < 160 or right_arm_angle < 160  # Arms should be nearly straight

def is_proper_hip_position(hip_l, hip_r, knee_l, knee_r):
    """Check if hips are in proper position during transition."""
    hip_height = (hip_l.y + hip_r.y) / 2
    knee_height = (knee_l.y + knee_r.y) / 2
    return hip_height > knee_height - 0.1  # Hips shouldn't rise too early

def find_lowest_squat_position(pose_landmarks):
    """Find the frame with the lowest hip position."""
    lowest_hip_y = -float('inf')
    lowest_position = None
    
    for landmarks in pose_landmarks:
        hip_l = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        hip_r = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        current_hip_y = (hip_l.y + hip_r.y) / 2
        
        if current_hip_y > lowest_hip_y:
            lowest_hip_y = current_hip_y
            lowest_position = landmarks
    
    return lowest_position

def is_full_squat_depth(hip_l, hip_r, knee_l, knee_r):
    """Check if full squat depth is achieved."""
    hip_height = (hip_l.y + hip_r.y) / 2
    knee_height = (knee_l.y + knee_r.y) / 2
    return hip_height > knee_height  # Hips should be below knees

def is_upright_torso(shoulder_l, shoulder_r, hip_l, hip_r):
    """Check if torso is upright in receiving position."""
    torso_angle = calculate_back_angle(shoulder_l, shoulder_r, hip_l, hip_r)
    return torso_angle > 75  # Torso should be nearly vertical

def is_proper_front_rack(landmarks):
    """Check if front rack position is correct."""
    elbow_l = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
    elbow_r = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
    shoulder_l = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    shoulder_r = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    
    elbows_high = (elbow_l.y < shoulder_l.y) and (elbow_r.y < shoulder_r.y)
    elbows_forward = (elbow_l.x > shoulder_l.x) and (elbow_r.x > shoulder_r.x)
    
    return elbows_high and elbows_forward

def is_hips_rising_fast(landmarks):
    """Check if hips are rising faster than shoulders during recovery."""
    hip_l = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    hip_r = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    shoulder_l = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    shoulder_r = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    
    hip_rise_rate = (hip_l.y + hip_r.y) / 2
    shoulder_rise_rate = (shoulder_l.y + shoulder_r.y) / 2
    
    return hip_rise_rate < shoulder_rise_rate - 0.1  # Hips shouldn't rise much faster than shoulders

def calculate_angle(p1, p2, p3):
    """Calculate angle between three points."""
    # Convert to numpy arrays for easier calculation
    p1_arr = np.array([p1.x, p1.y])
    p2_arr = np.array([p2.x, p2.y])
    p3_arr = np.array([p3.x, p3.y])
    
    # Calculate vectors
    v1 = p1_arr - p2_arr
    v2 = p3_arr - p2_arr
    
    # Calculate angle in degrees
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    
    return angle

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400
        
    video = request.files['video']
    if video.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if video:
        # Save to temporary file
        temp_dir = tempfile.gettempdir()
        video_path = os.path.join(temp_dir, secure_filename(video.filename))
        video.save(video_path)
        
        # Analyze the video
        feedback, metrics = analyze_movement(video_path)
        
        # Clean up
        os.remove(video_path)
        
        return jsonify({
            'feedback': feedback,
            'metrics': metrics.__dict__
        })

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)

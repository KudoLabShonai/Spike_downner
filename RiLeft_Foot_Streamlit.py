import cv2
import mediapipe as mp
import os
import streamlit as st
from tempfile import NamedTemporaryFile
from PIL import Image
import subprocess

# Streamlit UI
st.title("Volleyball Spike Analysis")
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

def convert_to_mp4(input_path, output_path):
    """動画形式をMP4に変換する"""
    command = [
        'ffmpeg', '-i', input_path, '-vcodec', 'libx264', '-acodec', 'aac', '-strict', 'experimental', output_path
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

if uploaded_file is not None:
    tfile = NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    input_video_path = tfile.name
    
    output_video_path = "output_video.mp4"
    output_image_dir = "output_images"
    os.makedirs(output_image_dir, exist_ok=True)

    # MediaPipe Pose setup
    mp_pose = mp.solutions.pose
    
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        st.error("Failed to load video.")
        st.stop()
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    min_y_value = float('inf')
    min_y_frame = 0

    with mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                left_heel_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL].y * height
                right_heel_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL].y * height
                avg_y = (left_heel_y + right_heel_y) / 2

                if avg_y < min_y_value:
                    min_y_value = avg_y
                    min_y_frame = frame_count
    
    start_frame = max(0, min_y_frame)
    frame_count_add = int(fps / 1.5)
    end_frame = min(total_frames - 1, min_y_frame + frame_count_add)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    with mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or int(cap.get(cv2.CAP_PROP_POS_FRAMES)) > end_frame:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                for landmark in [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
                                 mp_pose.PoseLandmark.LEFT_HEEL, mp_pose.PoseLandmark.RIGHT_HEEL]:
                    lm = results.pose_landmarks.landmark[landmark]
                    x, y = int(lm.x * width), int(lm.y * height)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            out.write(frame)
    
    cap.release()
    out.release()

    # Step detection and image saving
    cap = cv2.VideoCapture(output_video_path)
    prev_left_ankle_y = prev_left_knee_y = None
    prev_right_ankle_y = prev_right_knee_y = None
    STEP_THRESHOLD = 0.01 
    saved_steps = 0
    target_steps = 3
    step_images = []
    
    with mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False) as pose:
        while cap.isOpened() and saved_steps < target_steps:
            ret, frame = cap.read()
            if not ret:
                break
            
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            if results.pose_landmarks:
                left_ankle_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y
                left_knee_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y
                right_ankle_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y
                right_knee_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y
                
                step_detected = False
                step_type = ""
                
                if prev_left_ankle_y is not None and (left_ankle_y - prev_left_ankle_y) > STEP_THRESHOLD and (left_knee_y - prev_left_knee_y) > STEP_THRESHOLD:
                    step_detected = True
                    step_type = "left"
                
                if prev_right_ankle_y is not None and (right_ankle_y - prev_right_ankle_y) > STEP_THRESHOLD and (right_knee_y - prev_right_knee_y) > STEP_THRESHOLD:
                    step_detected = True
                    step_type = "right"
                
                if step_detected:
                    image_path = os.path.join(output_image_dir, f"{step_type}_step_{saved_steps + 1}.png")
                    cv2.imwrite(image_path, frame)
                    step_images.append(image_path)
                    saved_steps += 1
                
                prev_left_ankle_y, prev_left_knee_y = left_ankle_y, left_knee_y
                prev_right_ankle_y, prev_right_knee_y = right_ankle_y, right_knee_y
    
    st.success("Processing complete!")

    with open(output_video_path, "rb") as file:
        st.download_button("Download Processed Video", file, "processed_video.mp4", "video/mp4")
    
    for img in step_images:
        image = Image.open(img)
        st.image(image, caption=os.path.basename(img))
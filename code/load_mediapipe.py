import torch
import mediapipe as mp
import cv2
import numpy as np

# -----------------------
# 2. MediaPipe Pose setup
# -----------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# image_path = "powerlifting/train/images/deadlift_47_jpg.rf.497d03700d37bca33004bc112b2970e7.jpg"

# image = cv2.imread(image_path)
# rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# results = pose.process(rgb)

def extract_pose_features(image_path):
    """Extracts normalized pose landmarks from an image."""
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    
    if not results.pose_landmarks:
        return None

    landmarks = np.array(
        [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
    )

    # Normalize coordinates: make them invariant to image position & scale
    center = landmarks[mp_pose.PoseLandmark.NOSE.value]  # e.g., nose as origin
    landmarks -= center

    # Scale normalization (divide by body size)
    shoulder_dist = np.linalg.norm(
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value] -
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    )
    if shoulder_dist > 0:
        landmarks /= shoulder_dist

    return landmarks.flatten()

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import mediapipe as mp
import cv2

# -----------------------
# 2. MediaPipe Pose setup
# -----------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

image_path = "powerlifting/train/images/deadlift_47_jpg.rf.497d03700d37bca33004bc112b2970e7.jpg"

image = cv2.imread(image_path)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = pose.process(rgb)



# mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
# cv2.imshow("Pose", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



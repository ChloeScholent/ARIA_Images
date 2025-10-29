import torch
import mediapipe as mp
import cv2
import numpy as np
import os 
import pandas as pd 
from torch.utils.data import Dataset

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


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


class Landmark_Dataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
    
        # Map class names to numeric labels
        self.class_map = {"bench": 0, "squat": 1, "deadlift": 2}
        self.data["label"] = self.data["class"].map(self.class_map)
        
        # Drop the text class column
        self.features = self.data.drop(["class", "label"], axis=1).values.astype("float32")
        self.labels = self.data["label"].values.astype("int64")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx])
        y = torch.tensor(self.labels[idx])
        return x, y

image_dir = "data/dataset/"
data = []

for filename in os.listdir(image_dir):
    # if not filename.endswith("jpg"):
    #     continue
    
    class_name = None
    for name in ["bench", "squat", "deadlift"]:
        if name in filename.lower():
            class_name = name
            break
    if class_name is None:
        continue

    image_path = os.path.join(image_dir, filename)
    landmarks = extract_pose_features(image_path)

    if landmarks is not None:
        data.append([class_name] + landmarks.tolist())

columns = ["class"]
for i in range(33):
    columns += [f"x{i}", f"y{i}", f"z{i}"]

df = pd.DataFrame(data, columns=columns)
df.to_csv("pose_dataset.csv", index=False)
print("✅ Saved pose_dataset.csv with", len(df), "samples")







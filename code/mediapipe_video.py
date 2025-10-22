# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 12:18:15 2025

@author: rouxm
"""

import cv2
import mediapipe as mp
import pandas 
import os

mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(r"C://Users//rouxm//Desktop//2025-2026 - ARIA//Image//Porjet//mediapipe//squat_clean_clement.mp4")

output_csv = r"C://Users//rouxm//Desktop//2025-2026 - ARIA//Image//Porjet//mediapipe//squat_clean_clement_landmarks.csv"
output_xlsx = r"C://Users//rouxm//Desktop//2025-2026 - ARIA//Image//Porjet//mediapipe//squat_landmarks.xlsx"
save_as_excel=False

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can use 'XVID' if mp4v doesn’t work
out = cv2.VideoWriter(r"C://Users//rouxm//Desktop//2025-2026 - ARIA//Image//Porjet//mediapipe//squat_clean_clement.avi", fourcc, fps, (width, height))

all_data=[]
frame_idx=0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = mp_pose.process(frame)
    frame_data = {'frame': frame_idx}
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, 
            results.pose_landmarks, 
            mp.solutions.pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=4, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=6, circle_radius=2)
            )
    
    
        for i, lm in enumerate(results.pose_landmarks.landmark):
            frame_data[f'{i}_x'] = lm.x
            frame_data[f'{i}_y'] = lm.y
            frame_data[f'{i}_z'] = lm.z
            frame_data[f'{i}_visibility'] = lm.visibility
        else :
            for i in range(33):
                frame_data[f'{i}_x'] = None
                frame_data[f'{i}_y'] = None
                frame_data[f'{i}_z'] = None
                frame_data[f'{i}_visibility'] = None

    all_data.append(frame_data)
    cv2.imshow("Frame", frame)
    out.write(frame)
    frame_idx+=1
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

df=pandas.DataFrame(all_data)
if save_as_excel:
    df.to_excel(output_xlsx, index=False)
else:
    df.to_csv(output_csv, index=False)

print(f"Saved pose data to: {'Excel' if save_as_excel else 'CSV'}")


import math
import cv2
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt

mp_pose = mp.solutions.pose

pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

mp_drawing = mp.solutions.drawing_utils

def detectPose(image, pose, display=True):
    output_image = image.copy()

    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = pose.process(imageRGB)

    height, width, _ = image.shape

    landmarks = []

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS)

        for landmark in results.pose_landmarks.landmark:
            landmarks.append((int(landmark.x * width), int(landmark.y * height), (landmark.z * width)))
    
    if display:
        plt.figure(figsize=[22,22])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis("off")
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis("off")

        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
    else:
        return output_image, landmarks


# Inicializar pela webcam
video = cv2.VideoCapture(0)

# inicializar por um vídeo
#video = cv2.VideoCapture('Caminho_do_video')

time1 = 0

while video.isOpened():
    ok, frame = video.read()

    if not ok:
        break

    frame = cv2.flip(frame, 1)

    frame_height, frame_widht, _ = frame.shape

    frame = cv2.resize(frame, (int(frame_widht * (640/frame_height)), 640))

    frame, _ = detectPose(frame, pose_video, display=False)

    time2 = time()

    if (time2 - time1) > 0:
        frames_per_second = 1.0 / (time2 - time1)

        cv2.putText(frame, 'FPS: {}'.format(int(frames_per_second)), (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255 ,0), 3)

    time1 = time2

    cv2.imshow('Pose Detection', frame)

    k = cv2.waitKey(1) & 0xFF

    if(k == 27):
        break

video.release()

cv2.destroyAllWindows()
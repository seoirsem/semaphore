import pandas as pd
import glob
from matplotlib import pyplot as plt
import torch
import numpy as np
import math
import time
from os.path import exists
import cv2

from train_network import Model
from pose import Pose, semaphore_letters

def return_processed_frame(frame, pose, landmarks : bool, bounding_box : bool, show_background):
    """
    This function processes the frame and returns it adorned with the requested box or landmarks
    """
    landmarks_identified = pose.compute_frame(frame)
    if not landmarks_identified:
        if not show_background:
            return np.zeros_like(frame), pose
        else:
            return frame, pose
    
    if landmarks:
        frame = pose.show_pose_landmarks(show_background)
    
    kp = torch.tensor(pose.return_keypoint_vector()[1])
    model_output = model(kp)
    idx = torch.argmax(model_output).item()
    conf = torch.max(model_output).item()
    if bounding_box:
        frame = pose.draw_labelled_box(frame, "{} conf: {:.1f}%".format(semaphore_letters[idx], 100*conf))
    else:
        frame = pose.draw_label(frame, "{} conf: {:.1f}%".format(semaphore_letters[idx], 100*conf))

    return frame, pose

pose = Pose()
model = Model(38, 60, 27)
model_path = "train_semaphore.pt"

testfile = "data/25/1.png"
testfile = "data/flag_guy.png"

if exists(model_path):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Loaded model at ' + model_path)
    model.eval()
else:
    ValueError("No classifier model was found at {}".format(model_path))

camera = cv2.VideoCapture(0)

video_path = "data/navy_morse_video.mp4"
#camera = cv2.VideoCapture(video_path)

landmarks = True
bounding_box= False
show_background = False

wait = 0
while camera.isOpened():
    # if wait < 10:
    #     wait += 1
    #     continue
    success, frame = camera.read()
    if success:
        frame, pose = return_processed_frame(frame, pose, landmarks, bounding_box, show_background)
        cv2.imshow("pose landmarks", frame)
        #time.sleep(0.5)
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

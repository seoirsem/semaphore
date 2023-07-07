import cv2
import mediapipe as mp
import numpy as np
import time


class Pose():
    
    def __init__(self):

        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose()



mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_drawing_styles = mp.solutions.drawing_styles


pose = mp_pose.Pose(
    static_image_mode = False,
    model_complexity = 1, #either 0,1,2, 2 is best and we chooses it for a static image (no framerate issues)
    enable_segmentation = True,#to see where the person is
    min_detection_confidence = 0.5)

cm = cv2.VideoCapture(0)

while (cv2.waitKey(1) == -1):
    #img = cv2.imread('images/jack.png')
    success, img = cm.read()
    if not success:
        continue

    results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
        continue

    ann = img.copy()
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.3 # when do we consider it part of the person?
    bg_image = np.zeros(img.shape, dtype=np.uint8)
    bg_image[:] = (192,192,192)
    ann = np.where(condition, ann, bg_image)

    canny = cv2.Canny(cv2.cvtColor(ann,cv2.COLOR_RGB2GRAY), 100, 250)
    canny = np.stack([canny]*3,axis = -1)
    mp_drawing.draw_landmarks(
        ann,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    #ann = cv2.resize(ann,(1280,960))
    can_img = np.where(condition,canny,img)
    cv2.imshow('annotated',can_img)


cm.release()
import cv2
import mediapipe as mp
import numpy as np
import time

VISIBILITY_THRESHOLD = 0.5
PRESENCE_THRESHOLD = 0.5
pose_indices = {
    "nose": 0,
    "left_eye_inner": 1,
    "left_eye": 2,
    "left_eye_outer": 3,
    "right_eye_inner": 4,
    "right_eye": 5,
    "right_eye_outer": 6,
    "left_ear": 7,
    "right_ear": 8,
    "mouth_left": 9,
    "mouth_right": 10,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_pinky": 17,
    "right_pinky": 18,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 21,
    "right_thumb": 22,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}

semaphore_letters = {
    0: " ",
    1: "A",
    2: "B",
    3: "C",
    4: "D",
    5: "E",
    6: "F",
    7: "G",
    8: "H",
    9: "I",
    10: "J",
    11: "K",
    12: "L",
    13: "M",
    14: "N",
    15: "O",
    16: "P",
    17: "Q",
    18: "R",
    19: "S",
    20: "T",
    21: "U",
    22: "V",
    23: "W",
    24: "X",
    25: "Y",
    26: "Z"
}
semaphore_numbers = {
    "_" : 0,
    "A" : 1,
    "B" : 2,
    "C" : 3,
    "D" : 4,
    "E" : 5,
    "F" : 6,
    "G" : 7,
    "H" : 8,
    "I" : 9,
    "J" : 10,
    "K" : 11,
    "L" : 12,
    "M" : 13,
    "N" : 14,
    "O" : 15,
    "P" : 16,
    "Q" : 17,
    "R" : 18,
    "S" : 19,
    "T" : 20,
    "U" : 21,
    "V" : 22,
    "W" : 23,
    "X" : 24,
    "Y" : 25,
    "Z" : 26
}


class Pose():
    """
    A class to process and store each frame's pose information
    This class will not have any responsability for the temporal aspects of the
    video feed 
     """
    def __init__(self):
        # Create utils:
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.pose = self.mp_pose.Pose(
            static_image_mode = False,
            model_complexity = 1, #either 0,1,2, 2 is best and we chooses it for a static image (no framerate issues)
            enable_segmentation = True,#to see where the person is
            min_detection_confidence = 0.5)
        
        # Create data variables
        self._results = None
        self._pose_landmarks = None
        self._frame = None
        self._key_points = {}
        self._success = False

    def compute_frame(self,frame):
        """ 
        This performs all the frame processing work 
        
        """
        self._frame = frame
        self._shape = frame.shape
        
        self._results = self.pose.process(frame)#cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        self._pose_landmarks = self._results.pose_landmarks
        if self._results.pose_landmarks is None:
            self._success = False
        else:
            self._success = True
            self._key_points = self.get_keypoints(self._results)

        return self._success

    def get_keypoints(self, results):
        """
        Extracts the keypoint data from the obtained results. Applies thresholds to 
        when we use those keypoints. From docs at 
        https://github.com/google/mediapipe/blob/master/docs/solutions/pose.md#output
        
        x and y: Landmark coordinates normalized to [0.0, 1.0] by the image width and height respectively.
        
        z: Represents the landmark depth with the depth at the midpoint of hips being the origin, and the 
        smaller the value the closer the landmark is to the camera. The magnitude of z uses roughly the same scale as x.
        
        visibility: A value in [0.0, 1.0] indicating the likelihood of the landmark being visible (present and not occluded) in the image.
        """
        key_points = {}
        for idx, kp in enumerate(results.pose_landmarks.landmark):
            if ((kp.HasField('visibility') and kp.visibility < VISIBILITY_THRESHOLD) or
                (kp.HasField('presence') and kp.presence < PRESENCE_THRESHOLD)):
                continue
            key_points[idx] = kp
        return key_points

    def kp_in_keypoints(self, str_point : str):
        """ to access keypints by string names """
        return pose_indices[str_point] in self._key_points

    def kp_i_in_keypoints(self, idx : int):
        """ to access keypoints by index """
        return idx in self._key_points

    def return_kp_from_str(self, str_point : str):
        """ access the keypoint by a string name """
        return self._key_points[pose_indices[str_point]]

    def show_pose_landmarks(self, show_background : bool):
        """ 
        plot the pose landmarks on the original frame and pause here
        you can choose whether to show the original image with show_background
        """
        fr = self._frame
        if not show_background:
            fr = np.zeros_like(fr)
        self.mp_draw.draw_landmarks(
         fr,
         self._pose_landmarks,
         self.mp_pose.POSE_CONNECTIONS,
         landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
        return fr
    def draw_labelled_box(self,fr, label : str):
        """
        This function takes the input image WHICH IS ASSUMED TO BE A MODIFIED VERSION OF 
        FRAME and draws a bounding box on the person. It also adds text to the image 
        which is probably the predicted pose position. The modified frame is returned 
        """
        x1,x2,y1,y2 = self.person_bounding_box()
        #print(x1,y1,x2,y2)
        fr = cv2.rectangle(fr, (x1, y1), (x2,y2), (36,255,12), 1)
        cv2.putText(fr, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        return fr
    
    def draw_label(self, frame, label : str):
        """
        This adds text of the classifier to the top of the image
        """
        size_y = self._shape[1]
        size_x = self._shape[0]
        cv2.putText(frame, label, (size_x // 2 + 30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        return frame

    def return_keypoint_vector(self):
        """
        This function outputs the key datapoints for the use of the semaphore 
        recognition model in a vector format
        """
        points_to_use = [24,23,         # lower torso 
                        11,13,15,19,17, # left arm (person's perspective)
                        12,14,16,20,18, # right arm
                        0,10,9,5,2,8,7] # subset of face keypoints

        ret_vector = []
        missing_pts = 0
        # find 0,0:
        x_mean = 0
        y_mean = 0
        for i in points_to_use:
            if self.kp_i_in_keypoints(i):
                pt = self._key_points[i]
                x_mean += pt.x
                y_mean += pt.y
        x_mean = x_mean/19.0
        y_mean = y_mean/19.0
        for i in points_to_use:
            if self.kp_i_in_keypoints(i):
                pt = self._key_points[i]
                ret_vector.append(pt.x - x_mean)
                ret_vector.append(pt.y - y_mean)
            else: # the point is off screen
                ret_vector.append(0)
                ret_vector.append(0)
                missing_pts += 1
        return missing_pts, ret_vector


    def return_results(self):
        """ If you want to read the blaze results in an external function"""
        return self._results

    def person_area(self):
        """ 
        Returns the area of the detected person. This may be useful
        for drawing a bounding box on the person in the applet
        """
        if self._results:
            return np.stack((self._results.segmentation_mask,) * 3, axis=-1) > 0.2
        else:
            return None
        
    def person_bounding_box(self):
        """
        calculates the bounding box from the points we will use
        """
        points_to_use = [24,23,         # lower torso 
                        11,13,15,19,17, # left arm (person's perspective)
                        12,14,16,20,18, # right arm
                        0,10,9,5,2,8,7] # subset of face keypoints

        xmin = 100
        xmax = -100
        ymin = 100
        ymax = -100

        for i in points_to_use:
            if self.kp_i_in_keypoints(i):
                pt = self._key_points[i]
                xmin = min(xmin,pt.x)
                xmax = max(xmax,pt.x)
                ymin = min(ymin,pt.y)
                ymax = max(ymax,pt.y)

        xmin = int(xmin*self._shape[1])
        xmax = int(xmax*self._shape[1])
        ymin = int(ymin*self._shape[0])
        ymax = int(ymax*self._shape[0])

        return xmin, xmax, ymin, ymax

    def return_person_canny(self):
        """ This is just for fun - and puts a canny filter on the person in the frame """
        canny = cv2.Canny(cv2.cvtColor(self._frame,cv2.COLOR_RGB2GRAY), 100, 250)
        canny = np.stack([canny]*3,axis = -1)
        condition =  self.pose.person_area()
        can_img = np.where(condition,canny,self._frame)
        return can_img
        #cv2.imshow('Canny',pose.return_person_canny())
        #cv2.waitKey(0)


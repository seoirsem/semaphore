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

    def compute_frame(self,frame):
        """ 
        This performs all the frame processing work 
        
        """
        self._frame = frame
        self._results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self._pose_landmarks = self._results.pose_landmarks
        self._key_points = self.get_keypoints(self._results)

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

    def show_pose_landmarks(self):
        """ plot the pose landmarks on the original frame and pause here """
        fr = self._frame
        self.mp_draw.draw_landmarks(
         fr,
         self._pose_landmarks,
         self.mp_pose.POSE_CONNECTIONS,
         landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
        cv2.imshow("frame",fr)
        cv2.waitKey(0)
    
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
        for i in points_to_use:
            if self.kp_i_in_keypoints(i):
                pt = self._key_points[i]
                ret_vector.append(pt.x)
                ret_vector.append(pt.y)
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

    def return_person_canny(self):
        """ This is just for fun - and puts a canny filter on the person in the frame """
        canny = cv2.Canny(cv2.cvtColor(self._frame,cv2.COLOR_RGB2GRAY), 100, 250)
        canny = np.stack([canny]*3,axis = -1)
        condition =  self.pose.person_area()
        can_img = np.where(condition,canny,self._frame)
        return can_img
        #cv2.imshow('Canny',pose.return_person_canny())
        #cv2.waitKey(0)


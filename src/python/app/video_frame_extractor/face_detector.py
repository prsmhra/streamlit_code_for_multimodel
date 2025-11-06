"""
File: face_detetctor.py
Description:
Created on: 21/11/24
Author: 
E-mail:
"""
import time
import cv2
import math
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from src.python.app.constants.constants import Constants


mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=Constants.ONE, circle_radius=Constants.ONE)

class FaceMeshDetector:

    def __init__(self):
        with open(Constants.MP_TASK_FILE, mode=Constants.READ_BINARY) as f:
            f_buffer = f.read()
        base_options = mp_python.BaseOptions(model_asset_buffer=f_buffer)
        options = mp_python.vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            num_faces=1,
            result_callback=self.mp_callback)
        self.model = mp_python.vision.FaceLandmarker.create_from_options(
            options)

        self.landmarks = None
        self.blendshapes = None
        self.category_names = None
        self.latest_time_ms = Constants.ZERO

    def mp_callback(self, mp_result, output_image, timestamp_ms: int):
        if len(mp_result.face_landmarks) >= Constants.ONE and len(
                mp_result.face_blendshapes) >= Constants.ONE:
            self.landmarks = mp_result.face_landmarks[Constants.ZERO]
            self.blendshapes = [b.score for b in mp_result.face_blendshapes[Constants.ZERO]]
            self.category_names = [b.category_name for b in mp_result.face_blendshapes[Constants.ZERO]]
        else:

            self.landmarks = None
            self.blendshapes = None
            self.category_names = None

    def update(self, frame):
        t_ms = int(time.time() * Constants.THOUSAND)
        if t_ms <= self.latest_time_ms:
            return

        frame_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        self.model.detect_async(frame_mp, t_ms)
        self.latest_time_ms = t_ms

    def get_results(self):
        return self.landmarks, self.blendshapes, self.category_names
    

    def get_blendshape_dict(self,blendshapes, category_names):
        """
        This method return blendshape dict
        :param blendshapes: blendshape value list
        :param category_names: blendshape name list
        :return: blendshape dict
        """
        blendshapes_dict = dict()
        for key, val in enumerate(blendshapes):
            blendshapes_dict[category_names[key]] = val
        return blendshapes_dict
    
    def get_points(self,landmarks, frame):
        # make points
        img_h, img_w = frame.shape[:Constants.TWO]
        mesh_points = [(int(p.x*img_w), int(p.y*img_h)) for p in landmarks]
        mesh_array = np.array(mesh_points)
        return mesh_array
    
    def compute_head_angles(self, landmarks):
        nose_tip = landmarks[Constants.NOSE_POINT]  # Nose tip
        chin = landmarks[Constants.CHIN_POINT]  # Chin
        left_eye_outer = landmarks[Constants.EYE_POINT_L]  # Left eye outer corner
        right_eye_outer = landmarks[Constants.EYE_POINT_R]  # Right eye outer corner

        dx = right_eye_outer.x - left_eye_outer.x
        dz = right_eye_outer.z - left_eye_outer.z
        yaw = np.arctan2(dz, dx)

        dy = chin.y - nose_tip.y
        dz = chin.z - nose_tip.z
        pitch = Constants.NINTY-np.arctan2(dy, dz) * (Constants.ONE_EIGHTY / np.pi)

        dx_roll = left_eye_outer.x - right_eye_outer.x
        dy_roll = left_eye_outer.y - right_eye_outer.y
        roll = 180+(np.arctan2(dy_roll, dx_roll) * (Constants.ONE_EIGHTY / np.pi))
        return [pitch, yaw, roll]
    


    def generate_blendshapes(self, frame):
        """
        This method generate blendshape, landmarks
        :param frame: frame
        :return: landmarks, blendshape, blendshape dict
        """
        self.update(frame)
        landmarks, blendshapes, category_names = self.get_results()

        if (landmarks is None) or (blendshapes is None):
            return [], [], {}, []
        
        rotation_angles = self.compute_head_angles(landmarks)
        landmarks = self.get_points(landmarks, frame)
        if len(rotation_angles) < Constants.THREE:
            rotation_angles = [Constants.ZERO, Constants.ZERO, Constants.ZERO]

        blendshapes_dict = self.get_blendshape_dict(blendshapes, category_names)
        return landmarks, blendshapes, blendshapes_dict, rotation_angles

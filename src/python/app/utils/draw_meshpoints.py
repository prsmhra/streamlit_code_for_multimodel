import cv2
import numpy as np
from src.python.app.constants.constants import Constants

def draw_mesh_points(mesh_points, frame):
        """
        This method draw the landmarks on blank frame
        :param landmarks: landmark dictionary
        :param frame: input frame.
        :return mesh_frame: frame with face landmarks marked
        """
        
        mesh_frame = np.zeros(frame.shape, dtype=np.uint8)
        if len(mesh_points)>Constants.ZERO:
            for pnt in mesh_points:
                cv2.circle(mesh_frame, (pnt[0],pnt[1]), color=Constants.GREEN, radius=Constants.ONE, thickness=-Constants.ONE)

            cv2.polylines(mesh_frame, [mesh_points[Constants.LEFT_IRIS]], True, Constants.RED, Constants.ONE, cv2.LINE_AA)
            cv2.polylines(mesh_frame, [mesh_points[Constants.RIGHT_IRIS]], True, Constants.RED, Constants.ONE, cv2.LINE_AA)
            cv2.polylines(mesh_frame, [mesh_points[Constants.LEFT_EYE]], True, Constants.YELLOW, Constants.ONE, cv2.LINE_AA)
            cv2.polylines(mesh_frame, [mesh_points[Constants.RIGHT_EYE]], True, Constants.YELLOW, Constants.ONE, cv2.LINE_AA)
        return mesh_frame
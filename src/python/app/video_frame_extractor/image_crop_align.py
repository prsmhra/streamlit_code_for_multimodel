import cv2
import numpy as np
import logging
import mediapipe as mp
import time
from src.python.app.constants.constants import Constants

mp_pose = mp.solutions.pose
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
FACE_MODEL = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.1)

BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path='assets/efficientdet_lite0_8.tflite'),
    max_results=Constants.FIVE,
    running_mode=VisionRunningMode.IMAGE)
PERSON_DETECTOR = ObjectDetector.create_from_options(options)

class ImageCropAlign:
    def __init__(self):
        self.previos_frame = None
        self.previos_bbox = None
        self.prev_face = None


    def calculate_iou_with_pre_frame(self, previous_box, current_box):
        iou_list = []
        for current_boxes in current_box:
            x1 = max(previous_box[Constants.ZERO], current_boxes[Constants.ZERO])
            y1 = max(previous_box[Constants.ONE], current_boxes[Constants.ONE])
            x2 = min(previous_box[Constants.TWO], current_boxes[Constants.TWO])
            y2 = min(previous_box[Constants.THREE], current_boxes[Constants.THREE])

            intersection_area = max(Constants.ZERO, x2 - x1 + Constants.ONE) * max(Constants.ZERO,
                                                                                   y2 - y1 + Constants.ONE)
            previous_box_area = (previous_box[Constants.TWO] - previous_box[Constants.ZERO] + Constants.ONE) * (
                    previous_box[Constants.THREE] - previous_box[Constants.ONE] + Constants.ONE)
            current_boxes_area = (current_boxes[Constants.TWO] - current_boxes[Constants.ZERO] + Constants.ONE) * (
                    current_boxes[Constants.THREE] - current_boxes[
                Constants.ONE] + Constants.ONE)

            iou = intersection_area / float(
                previous_box_area + current_boxes_area - intersection_area + Constants.EPSILON)
            iou_list.append(iou)
        return iou_list

    def person_detector(self, image):
        """
        This method crops and aligns the image based on the face detection results.
        """
        person_bbox_list = []
        try:
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            results = PERSON_DETECTOR.detect(img)
            boxes = []
            if results.detections:
                for detection in results.detections:
                    for cat in detection.categories:
                        if cat.category_name == "person":
                            box = detection.bounding_box
                            boxes.append(box)
            for box in boxes:
                h = box.height
                orgX = box.origin_x
                orgY = box.origin_y
                w = box.width
                person_bbox_list.append([orgX, orgY, w, h])
            if len(person_bbox_list) > Constants.ONE and self.previos_bbox is not None:
                iou_list = self.calculate_iou_with_pre_frame(self.previos_bbox[Constants.ZERO],
                                                             person_bbox_list)
                seg_bbox_index = iou_list.index(max(iou_list))
                self.previos_bbox = [person_bbox_list[seg_bbox_index]]
            elif len(person_bbox_list) == Constants.ONE:
                self.previos_bbox = person_bbox_list
        except Exception as e:
            logging.info(
                f"Cannot create silhouette || Error: {str(e)}")
            
        self.previos_frame = image

    def crop_and_align(self, image):
        """
        This method crops and aligns the image based on the face detection results.
        """
        person_face = None
        frame_height, frame_width = image.shape[:Constants.TWO]
        self.person_detector(image)
        try:
            if self.previos_bbox is not None:
                x1, y1, w, h = self.previos_bbox[Constants.ZERO]
                x2 = x1 + w
                y2 = y1 + h
                y1 = int(
                    y1 - Constants.POINT_TWO * h) if y1 - Constants.POINT_TWO * h > Constants.ZERO else Constants.ZERO
                y2 = int(
                    y2 + Constants.POINT_TWO * h) if y2 + Constants.POINT_TWO * h < frame_height else frame_height
                x1 = int(
                    x1 - Constants.POINT_TWO * w) if x1 - Constants.POINT_TWO * w > Constants.ZERO else Constants.ZERO
                x2 = int(
                    x2 + Constants.POINT_TWO * w) if x2 + Constants.POINT_TWO * w < frame_width else frame_width
                cropped_roi = image[y1:y2, x1:x2]
            else:
                cropped_roi = image.copy()
                x1 = Constants.ZERO
                y1 = Constants.ZERO
                x2 = frame_width
                y2 = frame_height
                roi_width = frame_width
                roi_height = frame_height

            # detections on cropped roi
            frame_rgb = cv2.cvtColor(cropped_roi, cv2.COLOR_BGR2RGB)

            ch, cw, _ = frame_rgb.shape
            face_left, face_top, face_right, face_bottom = Constants.ZERO, Constants.ZERO, Constants.ZERO, Constants.ZERO
            results = FACE_MODEL.process(frame_rgb)
            # Draw the face detection annotations on the image.
            frame_rgb.flags.writeable = True
            image.flags.writeable = True
            # person_cropped = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            if results.detections:
                current_bbox = []
                for res in results.detections:
                    xMin = int(res.location_data.relative_bounding_box.xmin * cw)
                    yMin = int(res.location_data.relative_bounding_box.ymin * ch)
                    width = int(res.location_data.relative_bounding_box.width * cw)
                    height = int(res.location_data.relative_bounding_box.height * ch)

                    current_bbox.append([xMin+x1-10, yMin+y1-10, xMin+x1+width+10, yMin+y1+height+10])

                if len(results.detections) > Constants.ONE:
                    iou_list = self.calculate_iou_with_pre_frame(self.prev_face[Constants.ZERO],
                                                                    current_bbox)
                    seg_bbox_index = iou_list.index(max(iou_list))

                    self.prev_face = [current_bbox[seg_bbox_index]]
                elif len(current_bbox) == Constants.ONE:
                    self.prev_face = current_bbox
                face_left, face_top, face_right, face_bottom = self.prev_face[Constants.ZERO]
                person_face = image[face_top:face_bottom, face_left:face_right, :]
            return cv2.cvtColor(person_face, cv2.COLOR_BGR2RGB), cropped_roi
        except Exception as e:
            return None, None

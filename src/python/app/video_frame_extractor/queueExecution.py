import os
import csv
import numpy as np
from src.python.app.constants.constants import Constants
from src.python.app.video_frame_extractor.au_detection import AUIntensity
from src.python.app.video_frame_extractor.process_blendshape_detection import DetectBlendshapeAction
from src.python.app.video_frame_extractor.get_crafted_features import FeatureCAlculator


class QueueExecution:

    def __init__(self):
        self.extreme_feature_values = None
        self.rest_face_features = None
        self.detect_blendshapes = DetectBlendshapeAction()
        self.au_intensity = AUIntensity()
        self.featureCalculator = FeatureCAlculator()
        self.inv_blend_divisors = list()
        self.rest_mouth_features = [Constants.ZERO] * 21
        self.rest_nose_features = [Constants.ZERO] * 12
        self.rest_eye_features = [Constants.ZERO] * 13
        self.mouth_extreme_features = [Constants.ZERO] * 18
        self.eye_extreme_features = [Constants.ZERO] * 16
        self.csv_writer = None
        self.csv_file = None
        self.reset_queue()


    def reset_queue(self, queue_length=Constants.QUEUE_LENGTH):
        self.org_frame_queue = list()
        self.face_frame_queue = list()
        self.mesh_frame_queue = list()
        self.blendshape_dict_queue = list()
        self.landmarks_queue = list()
        self.crafted_features_queue = list()
        self.rotation_queue = list()
        self.frame_count = Constants.ZERO
        self.blendshape_queue = [[] for _ in range(len(Constants.BLENDSHAPE_HEADERS_KEY[Constants.TWO:]))]
        self.running_avg_queue = [[] for _ in range(len(Constants.BLENDSHAPE_HEADERS_KEY[Constants.TWO:]))]

        self.results = [[Constants.STOP, Constants.ZERO, Constants.ZERO, Constants.ZERO, Constants.STOP, Constants.TEN_POW_SEVEN, None, None] for _ in
                        range(len(Constants.BLENDSHAPE_HEADERS_KEY[Constants.TWO:]))]
        self.blendshape_detection = [Constants.ZERO for _ in range(len(Constants.BLENDSHAPE_HEADERS_KEY[Constants.TWO:]))]

    def scientific_notation(self, x):

        # Initialize n
        n = Constants.ZERO
        # Adjust x to fall within the range 0.01 < x < 0.1
        if x >= Constants.POINT_ONE:
            while x >= Constants.POINT_ONE:
                x /= Constants.TEN
                n += Constants.ONE
        elif x < Constants.POINT_ZERO_ONE and x != Constants.ZERO:
            while x < Constants.POINT_ZERO_ONE:
                x *= Constants.TEN
                n -= Constants.ONE

        return n

    def check_elements(self, blendshapes, indices_list):
        # Check if all elements at specified indices are less than the threshold
        return all(blendshapes[idx] == Constants.ZERO for idx in indices_list)

    def popQueues(self, queue_):
        if len(queue_) == Constants.QUEUE_LENGTH:
            queue_.pop(Constants.ZERO)

    def processQueue(self, frame, mesh_frame, face_frame, landmarks, modelblendshapes, blendshape_dict, rotations):
        # take deepcopy
        blendshapes = modelblendshapes.copy()
        frame_list = []
        # initialize vals
        au_dict = Constants.AU_DICT
        frame_emotion = dict()
        au_time_dict = Constants.EMOTION_GRAPH

        crafted_features = self.featureCalculator.get_crafted_features(landmarks)

        # fill queues
        self.crafted_features_queue.append(crafted_features)
        self.landmarks_queue.append(landmarks)
        self.mesh_frame_queue.append(mesh_frame)
        self.org_frame_queue.append(frame)
        self.face_frame_queue.append(face_frame)
        self.blendshape_dict_queue.append(blendshape_dict)
        self.rotation_queue.append(rotations)

        # add inverse values - only for mediapipe
        for idx in Constants.SMALL_VAL_BLENDSHAPES:
            div = self.scientific_notation(Constants.ONE / blendshapes[idx + Constants.ONE])
            inv_val = Constants.ONE / (blendshapes[idx + Constants.ONE] * Constants.TEN ** (div))

            blendshapes.append(inv_val)

        # blend shapes queue update
        for key, val in enumerate(blendshapes):
            if key == Constants.ZERO:
                continue
            self.blendshape_queue[key - Constants.ONE].append(val)

        # calculate running averages
        if len(self.blendshape_queue[Constants.ZERO]) == Constants.QUEUE_LENGTH:

            # update running avg queue
            for key in range(len(blendshapes)):
                if key == Constants.ZERO:
                    continue
                self.running_avg_queue[key - Constants.ONE].append(
                    sum(self.blendshape_queue[key - Constants.ONE]) / len(self.blendshape_queue[key - Constants.ONE]))
                self.blendshape_queue[key - Constants.ONE].pop(Constants.ZERO)

        blendshape_indices = np.arange(Constants.ZERO, len(blendshapes) - Constants.ONE, Constants.ONE)
        # au intensity calculation
        if len(self.running_avg_queue[Constants.ZERO]) == Constants.QUEUE_LENGTH:

            self.results = list(map(lambda key: self.detect_blendshapes.process_action_state(key, self.frame_count,
                                                                                             self.running_avg_queue[key],
                                                                                             self.results[key]),
                                    blendshape_indices))
            self.blendshape_detection = [Constants.ACTION_STATE.get(x[Constants.ZERO], Constants.ZERO) for x in self.results]

            mouth_motion, nose_motion, eye_motion = Constants.ZERO, Constants.ZERO, Constants.ZERO

            # resting mouth
            if self.check_elements(self.blendshape_detection, Constants.MOUTH_BLENDSHAPES_CHECK):
                self.rest_mouth_features = self.crafted_features_queue[Constants.ZERO][
                                           :Constants.NOSE_FEATURES_START_IDX]
                self.mouth_extreme_features = self.featureCalculator.get_mouth_extreme_values(self.rest_mouth_features)
            else:
                mouth_motion = Constants.ONE

            # resting nose
            if self.check_elements(self.blendshape_detection, Constants.NOSE_BLENDSHAPES_CHECK):
                self.rest_nose_features = self.crafted_features_queue[Constants.ZERO][
                                          Constants.NOSE_FEATURES_START_IDX:Constants.EYE_FEATURES_START_IDX]
            else:
                nose_motion = Constants.ONE

            # resting eyes
            if self.check_elements(self.blendshape_detection, Constants.EYE_BLENDSHAPES_CHECK):
                self.rest_eye_features = self.crafted_features_queue[Constants.ZERO][
                                         Constants.EYE_FEATURES_START_IDX:]
                self.eye_extreme_features = self.featureCalculator.get_eye_extreme_values(self.rest_eye_features)
            else:
                eye_motion = Constants.ONE

            self.rest_face_features = self.rest_mouth_features + self.rest_nose_features + self.rest_eye_features
            self.extreme_feature_values = self.mouth_extreme_features + self.eye_extreme_features

            # calculate au scores, emotion_scores
            au_dict, frame_emotion, au_time_dict, frame_list = self.au_intensity.inference(self.blendshape_detection,
                                                                               self.blendshape_dict_queue[Constants.ZERO],
                                                                               self.crafted_features_queue.pop(Constants.ZERO),
                                                                               self.rest_face_features,
                                                                               self.extreme_feature_values,
                                                                               mouth_motion,
                                                                               nose_motion,
                                                                               eye_motion)


            # pop running average
            for idx in range(len(self.running_avg_queue)):
                self.running_avg_queue[idx].pop(Constants.ZERO)

        self.frame_count += Constants.ONE

        ret_vals = {
            Constants.DECISON_FRAME: self.org_frame_queue[Constants.ZERO],
            Constants.DECISOON_FACE_FRAME: self.face_frame_queue[Constants.ZERO],
            Constants.DECISION_FRAME_MESH: self.mesh_frame_queue[Constants.ZERO],
            Constants.BLENDSHAPE_VALS: self.blendshape_dict_queue[Constants.ZERO],
            Constants.ROTATION_VALS: self.rotation_queue[Constants.ZERO],
            Constants.AU_INTENSITY: au_dict,
            Constants.EMOTION_DETECTION: frame_emotion,
            Constants.EMOTION_GRAPH_DATA: au_time_dict,
            Constants.FRAME_COUNTS : frame_list

        }

        self.popQueues(self.crafted_features_queue)
        self.popQueues(self.landmarks_queue)
        self.popQueues(self.mesh_frame_queue)
        self.popQueues(self.org_frame_queue)
        self.popQueues(self.face_frame_queue)
        self.popQueues(self.blendshape_dict_queue)
        self.popQueues(self.rotation_queue)

        return ret_vals

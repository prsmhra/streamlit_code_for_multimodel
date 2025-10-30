"""
File: au_detection.py
Description:
Created on: 26/09/24
Author: 
E-mail:
"""
import numpy as np
from collections import deque, Counter
from src.python.app.video_frame_extractor.face_detector import FaceMeshDetector
from src.python.app.video_frame_extractor.detect_emotion import DetectEmotion
from src.python.app.video_frame_extractor.detect_pain import DetectPain
from src.python.app.constants.constants import Constants

class AUIntensity:
    def __init__(self):
        self.eye_close = deque()
        self.au_dict = Constants.AU_DICT
        self.facemesh_detector = FaceMeshDetector()
        self.emotion_detector = DetectEmotion()
        self.pain_detector = DetectPain()
        self.emotion_time_dict = dict()
        self.blendshape_detection = None
        self.blendshapes_dict = None
        self.extreme_vals = list()
        self.count = Constants.ZERO
        self.frame_list = deque(maxlen=Constants.CHART_MAX_FRAME)

    def get_emotion_time_dict(self, frame_emotion):
        """
        This method return blendshape dict
        :param blendshapes: blendshape value list
        :param category_names: blendshape name list
        :return: blendshape dict
        """

        for key, val in frame_emotion.items():
            
            if key not in self.emotion_time_dict:
                self.emotion_time_dict[key] = [val]
            else:   
                if len(self.emotion_time_dict[key]) < Constants.CHART_MAX_FRAME:
                    self.emotion_time_dict[key].append(val)
                else:
                    self.emotion_time_dict[key].pop(Constants.ZERO)
                    self.emotion_time_dict[key].append(val)

    def map_to_intensity(self,min_val, max_val, current_val,inverted=False):
        # Ensure current_val is within the bounds [min_val, max_val]
        current_val = max(min_val, min(current_val, max_val))
        
        # Map the current_val to the range 0 to 5
        if inverted:
            intensity = Constants.FIVE * (max_val - current_val) / (max_val - min_val+Constants.EPSILON)
        else:
            intensity = Constants.FIVE * (current_val - min_val) / (max_val - min_val+Constants.EPSILON)

        intensity = min(Constants.FIVE, intensity)
        return round(intensity)


    def is_au_43_45_46(self):
        """
        This method calculate values for AU07, AU43, AU45, AU46
        :param blendshapes_dict: blendshape dictionary
        :return: au7, au43, au45, au46 values
        """
        au43, au45, au46 = Constants.ZERO, Constants.ZERO, Constants.ZERO

        if self.blendshape_detection[Constants.EYE_BLINK_LEFT_IDX]==Constants.ONE\
        or self.blendshape_detection[Constants.EYE_BLINK_RIGHT_IDX]==Constants.ONE:
            left_blink = self.map_to_intensity(Constants.ZERO, self.rest_face_features[Constants.LEFT_EYE_INNER_DST_IDX],
                                                    self.crafted_features[Constants.LEFT_EYE_INNER_DST_IDX], inverted=True)
            
            right_blink = self.map_to_intensity(Constants.ZERO, self.rest_face_features[Constants.RIGHT_EYE_INNER_DST_IDX],
                                                self.crafted_features[Constants.RIGHT_EYE_INNER_DST_IDX], inverted=True)
 
        
            if self.blendshape_detection[Constants.EYE_BLINK_LEFT_IDX]==Constants.ONE\
                and self.blendshape_detection[Constants.EYE_BLINK_RIGHT_IDX]==Constants.ONE:
                
                au45 = max(left_blink, right_blink)

            elif self.blendshape_detection[Constants.EYE_BLINK_LEFT_IDX]==Constants.ONE\
            or self.blendshape_detection[Constants.EYE_BLINK_RIGHT_IDX]==Constants.ONE:
                au46 = max(left_blink, right_blink)

            if au45 or au46:
                if len(self.eye_close) < Constants.MAX_QUEUE_LEN:
                    self.eye_close.append(Constants.ONE)
                else:
                    self.eye_close.popleft()
                    self.eye_close.append(Constants.ONE)
            else:
                if len(self.eye_close) < Constants.MAX_QUEUE_LEN:
                    self.eye_close.append(Constants.ZERO)
                else:
                    self.eye_close.popleft()
                    self.eye_close.append(Constants.ZERO)
            blink_close = Counter(np.array(self.eye_close))

            if blink_close[Constants.ONE] >= Constants.CLOSE_FRAME_THRESH:
                au43 =  max(left_blink, right_blink)
                au45 = Constants.ZERO

    
        return au43, au45, au46

    

    def is_au_44(self):
        """
        This method calculate values for AU44
        :param blendshapes_dict: blendshape dictionary
        :return: au44 values
        """
        au44 = Constants.ZERO
        if self.blendshape_detection[Constants.EYE_SQUINT_LEFT_IDX]==Constants.ONE\
                or self.blendshape_detection[Constants.EYE_SQUINT_RIGHT_IDX]==Constants.ONE:
            left_squint_intensity = self.map_to_intensity(self.extreme_vals[Constants.LEFT_EYES_QUINT_IDX],
                                                       self.rest_face_features[Constants.LEFT_EYE_INNER_DST_IDX],
                                                   self.crafted_features[Constants.LEFT_EYE_INNER_DST_IDX],
                                                   inverted=True)
            
            right_squint_intenisty = self.map_to_intensity(self.extreme_vals[Constants.RIGHT_EYES_QUINT_IDX],
                                                   self.rest_face_features[Constants.RIGHT_EYE_INNER_DST_IDX],
                                                   self.crafted_features[Constants.RIGHT_EYE_INNER_DST_IDX],
                                                   inverted=True)
            
            au44 = max(left_squint_intensity, right_squint_intenisty)
        return au44

    def is_au_05_07(self):
        """
        This method calculate values for AU05
        :param blendshapes_dict: blendshape dictionary
        :return: au5 values
        """
        au5, au7 = Constants.ZERO, Constants.ZERO
        if self.blendshape_detection[Constants.EYE_WIDE_LEFT_IDX]==Constants.ONE\
            or self.blendshape_detection[Constants.EYE_WIDE_RIGHT_IDX]==Constants.ONE:
            left_intensity = self.map_to_intensity(Constants.ZERO,
                                                   self.rest_face_features[Constants.LEFT_EYE_WIDE_IDX],
                                                   self.crafted_features[Constants.LEFT_EYE_WIDE_IDX],
                                                   inverted=True)  # should it be True
            
            right_intensity = self.map_to_intensity(Constants.ZERO,
                                                   self.rest_face_features[Constants.RIGHT_EYE_WIDE_IDX],
                                                   self.crafted_features[Constants.LEFT_EYE_WIDE_IDX],
                                                   inverted=True) # should it be True
            au5 = max(left_intensity, right_intensity)


        if self.blendshape_detection[Constants.EYE_SQUINT_LEFT_IDX] == Constants.ONE \
            or self.blendshape_detection[Constants.EYE_SQUINT_RIGHT_IDX] == Constants.ONE:
            left_lid_intensity = self.map_to_intensity(self.extreme_vals[Constants.LEFT_LID_TIGHT_MAX_IDX],
                                                       self.rest_face_features[Constants.LEFT_EYE_INNER_DST_IDX],
                                                   self.crafted_features[Constants.LEFT_EYE_INNER_DST_IDX],
                                                   inverted=True)
            
            right_lid_intenisty = self.map_to_intensity(self.extreme_vals[Constants.RIGHT_LID_TIGHT_MAX_IDX],
                                                   self.rest_face_features[Constants.RIGHT_EYE_INNER_DST_IDX],
                                                   self.crafted_features[Constants.RIGHT_EYE_INNER_DST_IDX],
                                                   inverted=True)
            au7 = max(left_lid_intensity, right_lid_intenisty)

        return au5, au7

    def is_au_41_42(self):
        """
        This method calculate values for AU41, AU42
        :param blendshapes_dict: blendshape dictionary
        :return: au41, au42 values
        """
        au41, au42 = Constants.ZERO, Constants.ZERO
        if self.blendshape_detection[Constants.EYE_BLINK_LEFT_IDX]==Constants.ONE\
            or self.blendshape_detection[Constants.EYE_BLINK_RIGHT_IDX]==Constants.ONE\
            or self.blendshape_detection[Constants.EYE_SQUINT_LEFT_IDX]==Constants.ONE\
            or self.blendshape_detection[Constants.EYE_SQUINT_RIGHT_IDX]==Constants.ONE:
            left_eye_inensiy = self.map_to_intensity(self.extreme_vals[Constants.LEFT_EYE_DROOP_IDX],
                                                    self.rest_face_features[Constants.LEFT_EYE_INNER_DST_IDX],
                                                    self.crafted_features[Constants.LEFT_EYE_INNER_DST_IDX], inverted=True)
            right_eye_inensiy = self.map_to_intensity(self.extreme_vals[Constants.RIGHT_EYE_DROOP_IDX],
                                                    self.rest_face_features[Constants.RIGHT_EYE_INNER_DST_IDX],
                                                    self.crafted_features[Constants.RIGHT_EYE_INNER_DST_IDX], inverted=True)
            au41 = max(left_eye_inensiy, right_eye_inensiy)

        if  self.blendshape_detection[Constants.EYE_BLINK_LEFT_IDX]==Constants.ONE\
            or self.blendshape_detection[Constants.EYE_BLINK_LEFT_IDX]==Constants.ONE\
            or self.blendshape_detection[Constants.EYE_SQUINT_LEFT_IDX]==Constants.ONE\
            or self.blendshape_detection[Constants.EYE_SQUINT_RIGHT_IDX]==Constants.ONE:
                left_eye_inensiy = self.map_to_intensity(self.extreme_vals[Constants.LEFT_EYE_SLIT_IDX],
                                                    self.rest_face_features[Constants.LEFT_EYE_INNER_DST_IDX],
                                                    self.crafted_features[Constants.LEFT_EYE_INNER_DST_IDX], inverted=True)
                right_eye_inensiy = self.map_to_intensity(self.extreme_vals[Constants.RIGHT_EYE_SLIT_IDX],
                                                        self.rest_face_features[Constants.RIGHT_EYE_INNER_DST_IDX],
                                                        self.crafted_features[Constants.RIGHT_EYE_INNER_DST_IDX], inverted=True)
                au42 = max(left_eye_inensiy, right_eye_inensiy)
        return au41, au42

    def is_au_16(self):
        """
        This method calculate values for AU16
        :param blendshapes_dict: blendshape dictionary
        :return: au16 values
        """
        au16_intensity = Constants.ZERO
        if self.blendshape_detection[Constants.MOUTH_LOWER_DOWN_LEFT_IDX]==Constants.ONE\
                or self.blendshape_detection[Constants.MOUTH_LOWER_DOWN_RIGHT_IDX]==Constants.ONE:
                if self.extreme_vals[Constants.LIPS_PART_IDX]<self.crafted_features[Constants.LIP_INNER_V_DST_IDX]:
                    au16_intensity = Constants.FIVE
                else:
                    if abs((self.rest_face_features[Constants.LIP_U_OUT_NOSE_TIP_DST_IDX]\
                    -self.crafted_features[Constants.LIP_U_OUT_NOSE_TIP_DST_IDX])/self.rest_face_features[Constants.LIP_U_OUT_NOSE_TIP_DST_IDX])<=Constants.POINT_TWO:
                        
                        au16_intensity = self.map_to_intensity(Constants.ZERO,
                                                            self.extreme_vals[Constants.LIPS_PART_IDX],
                                                            self.crafted_features[Constants.LIP_INNER_V_DST_IDX])
        return au16_intensity

    def is_au_18_22(self):
        """
        This method calculate values for AU18, AU22
        :param blendshapes_dict: blendshape dictionary
        :return: au18, au22 values
        """
        au18_intensity, au22_intensity = Constants.ZERO, Constants.ZERO
        if self.blendshape_detection[Constants.MOUTH_FUNNEL_IDX]==Constants.ONE\
            and self.au25_intensity>=2:
            au22_intensity_h = self.map_to_intensity((Constants.TWO_THIRD)*self.rest_face_features[Constants.LIP_OUTER_H_DST_IDX],
                                                    self.rest_face_features[Constants.LIP_OUTER_H_DST_IDX],
                                                    self.crafted_features[Constants.LIP_OUTER_H_DST_IDX],
                                                    inverted=True)
            
            au22_intensity_v = self.map_to_intensity(self.rest_face_features[Constants.LIP_OUTER_V_DST_IDX],
                                                   Constants.TWO * self.rest_face_features[Constants.LIP_OUTER_V_DST_IDX],
                                                   self.crafted_features[Constants.LIP_OUTER_V_DST_IDX])
            au22_intensity = max(au22_intensity_h,au22_intensity_v)

   
        if self.blendshape_detection[Constants.MOUTH_PUCKER_IDX]==Constants.ONE\
            and self.au25_intensity<2:
            au18_intensity_h = self.map_to_intensity(Constants.HALF*self.rest_face_features[Constants.LIP_OUTER_H_DST_IDX],
                                                    self.rest_face_features[Constants.LIP_OUTER_H_DST_IDX],
                                                    self.crafted_features[Constants.LIP_OUTER_H_DST_IDX],
                                                    inverted=True)
            au18_intensity_v = self.map_to_intensity(self.rest_face_features[Constants.LIP_OUTER_V_DST_IDX],
                                                   Constants.ONE_TWO_FIVE * self.rest_face_features[Constants.LIP_OUTER_V_DST_IDX],
                                                   self.crafted_features[Constants.LIP_OUTER_V_DST_IDX])
            au18_intensity = max(au18_intensity_h,au18_intensity_v)
        return au18_intensity, au22_intensity

    def is_au_61_62_63_64(self):
        """
        This method calculate values for AU61, AU62, AU63, AU64
        :param blendshapes_dict: blendshape dictionary
        :return: au61, au62, au63, au64 values
        """
        au61, au62, au63, au64 = Constants.ZERO, Constants.ZERO, Constants.ZERO, Constants.ZERO
        if (self.blendshape_detection[Constants.EYE_LOOK_IN_LEFT_IDX] == Constants.ONE \
                and self.blendshape_detection[Constants.EYE_LOOK_OUT_RIGHT_IDX] == Constants.ONE)\
            or (self.blendshape_detection[Constants.EYE_LOOK_IN_RIGHT_IDX] == Constants.ONE\
                and self.blendshape_detection[Constants.EYE_LOOK_OUT_LEFT_IDX] == Constants.ONE)\
            or (self.blendshape_detection[Constants.EYE_LOOK_DOWN_LEFT_IDX] == Constants.ONE \
                and self.blendshape_detection[Constants.EYE_LOOK_DOWN_RIGHT_IDX] == Constants.ONE)\
            or (self.blendshape_detection[Constants.EYE_LOOK_UP_LEFT_IDX] == Constants.ONE \
                and self.blendshape_detection[Constants.EYE_LOOK_UP_RIGHT_IDX] == Constants.ONE):

            if self.blendshapes_dict.get(Constants.EYE_LOOK_IN_LEFT_KEY) >= Constants.LOOK_RIGHT_THRESH \
                    and self.blendshapes_dict.get(Constants.EYE_LOOK_OUT_RIGHT_KEY) >= Constants.LOOK_RIGHT_THRESH:
                
                au62 = self.map_to_intensity(Constants.LOOK_RIGHT_THRESH, Constants.ONE,
                                                max(self.blendshapes_dict.get(Constants.EYE_LOOK_IN_LEFT_KEY), self.blendshapes_dict.get(Constants.EYE_LOOK_OUT_RIGHT_KEY)))
            elif self.blendshapes_dict.get(Constants.EYE_LOOK_IN_RIGHT_KEY) >= Constants.LOOK_LEFT_THRESH \
                    and self. blendshapes_dict.get(Constants.EYE_LOOK_OUT_LEFT_KEY) >= Constants.LOOK_LEFT_THRESH:
                
                au61 = self.map_to_intensity(Constants.LOOK_LEFT_THRESH, Constants.ONE,
                                                max(self.blendshapes_dict.get(Constants.EYE_LOOK_IN_RIGHT_KEY), self.blendshapes_dict.get(Constants.EYE_LOOK_OUT_LEFT_KEY)))

            if self.blendshapes_dict.get(Constants.EYE_LOOK_DOWN_LEFT_KEY) >= Constants.LOOK_DOWN_THRESH \
                    and self.blendshapes_dict.get(Constants.EYE_LOOK_DOWN_RIGHT_KEY) >= Constants.LOOK_DOWN_THRESH:
                au64 = self.map_to_intensity(Constants.LOOK_DOWN_THRESH, Constants.ONE,
                                                max(self.blendshapes_dict.get(Constants.EYE_LOOK_DOWN_LEFT_KEY), self.blendshapes_dict.get(Constants.EYE_LOOK_DOWN_RIGHT_KEY)))
            elif self.blendshapes_dict.get(Constants.EYE_LOOK_UP_LEFT_KEY) >= Constants.LOOK_UP_THRESH \
                    and self.blendshapes_dict.get(Constants.EYE_LOOK_UP_RIGHT_KEY) >= Constants.LOOK_UP_THRESH:
                au63 = self.map_to_intensity(Constants.LOOK_UP_THRESH, Constants.POINT_THREE, # max value will be 1
                                                max(self.blendshapes_dict.get(Constants.EYE_LOOK_UP_LEFT_KEY), self.blendshapes_dict.get(Constants.EYE_LOOK_UP_RIGHT_KEY)))
            
        return au61, au62, au63, au64

    

    def is_au_01_04(self):
        """
        This method calculate values for AU04
        :param blendshapes_dict: blendshape dictionary
        :return: au4 values
        """
        au1, au4 = Constants.ZERO, Constants.ZERO
        
        if self.blendshape_detection[Constants.BROW_INNER_UP_IDX]==Constants.ONE:

            
            
            
            left_intensity = self.map_to_intensity(self.rest_face_features[Constants.LEFT_EYE_WIDE_IDX], #self.rest_face_features[Constants.LEFT_BROW_INNER_RAISER],
                                                   Constants.POINT_SEVEN_FIVE*self.extreme_vals[Constants.LEFT_INNER_BROW_MAX_IDX], # changed from 1
                                                   self.crafted_features[Constants.LEFT_BROW_INNER_RAISER])
            
            right_intensity = self.map_to_intensity(self.rest_face_features[Constants.RIGHT_EYE_WIDE_IDX], #self.rest_face_features[Constants.RIGHT_BROW_INNER_RAISER],
                                                   Constants.POINT_SEVEN_FIVE*self.extreme_vals[Constants.RIGHT_INNER_BROW_MAX_IDX], # changed from 1
                                                   self.crafted_features[Constants.RIGHT_BROW_INNER_RAISER])
            
            au1 = max(left_intensity, right_intensity)
        if self.blendshape_detection[Constants.BROW_DOWN_LEFT_IDX]==Constants.ONE\
        or self.blendshape_detection[Constants.BROW_DOWN_RIGHT_IDX]==Constants.ONE:
            left_brow_down_intensity = self.map_to_intensity(Constants.POINT_EIGHT*self.rest_face_features[Constants.LEFT_BROW_LOWER_IDX],
                                                             self.extreme_vals[Constants.LEFT_BROW_LOWER_MAX_IDX],
                                                             self.crafted_features[Constants.LEFT_BROW_LOWER_IDX],
                                                             inverted=True)
            
            right_brow_down_intensity = self.map_to_intensity(Constants.POINT_EIGHT*self.rest_face_features[Constants.RIGHT_BROW_LOWER_IDX],
                                                             self.extreme_vals[Constants.RIGHT_BROW_LOWER_MAX_IDX],
                                                             self.crafted_features[Constants.RIGHT_BROW_LOWER_IDX],
                                                             inverted=True)
            
            # added
            left_brow_right_intensity = self.map_to_intensity(Constants.HALF * self.rest_face_features[Constants.LEFT_BROW_CEN_DST_IDX], # changed from Constants.POINT_SEVEN_FIVE
                                                             self.rest_face_features[Constants.LEFT_BROW_CEN_DST_IDX],
                                                             self.crafted_features[Constants.LEFT_BROW_CEN_DST_IDX],
                                                             inverted=True)
            right_brow_right_intensity = self.map_to_intensity(Constants.HALF * self.rest_face_features[Constants.RIGHT_BROW_CEN_DST_IDX], # changed from Constants.POINT_SEVEN_FIVE
                                                             self.rest_face_features[Constants.RIGHT_BROW_CEN_DST_IDX],
                                                             self.crafted_features[Constants.RIGHT_BROW_CEN_DST_IDX],
                                                             inverted=True)
            au4 = max(left_brow_down_intensity, right_brow_down_intensity, left_brow_right_intensity, right_brow_right_intensity)

        return au1, au4

    def is_au_28(self):
        """
        This method calculate values for AU28
        :param blendshapes_dict: blendshape dictionary
        :return: au28 values
        """
        au28_intensity = Constants.ZERO
        if self.blendshape_detection[Constants.MOUTH_ROLL_LOWER_IDX]==Constants.ONE\
                and self.blendshape_detection[Constants.MOUTH_ROLL_UPPER_IDX]==Constants.ONE:
            au28_intensity = self.map_to_intensity(Constants.ZERO,
                                                   self.rest_face_features[Constants.LIP_ROLL_HEIGHT_IDX],
                                                   self.crafted_features[Constants.LIP_ROLL_HEIGHT_IDX],
                                                   inverted=True)

        return au28_intensity

    def is_au_02(self):
        """
        This method calculate values for AU02
        :param blendshapes_dict: blendshape dictionary
        :return: au2 values
        """
        au2 = Constants.ZERO
        if (self.blendshape_detection[Constants.BROW_OUTER_UP_LEFT_IDX]==Constants.ONE\
            or self.blendshape_detection[Constants.BROW_OUTER_UP_RIGHT_IDX]==Constants.ONE):
            left_au2_intensity = self.map_to_intensity(self.rest_face_features[Constants.LEFT_BROW_OUTER_RAISER],
                                    Constants.POINT_EIGHT*self.extreme_vals[Constants.LEFT_OUTER_BROW_MAX_IDX], # changed from 1
                                    self.crafted_features[Constants.LEFT_BROW_OUTER_RAISER]
            )

            right_au2_intensity = self.map_to_intensity(self.rest_face_features[Constants.RIGHT_BROW_OUTER_RAISER],
                                    Constants.POINT_EIGHT*self.extreme_vals[Constants.RIGHT_OUTER_BROW_MAX_IDX], # changed from 1
                                    self.crafted_features[Constants.RIGHT_BROW_OUTER_RAISER]
            )
            au2 = max(left_au2_intensity, right_au2_intensity)
 
        return au2

    def is_au_13_14_24(self):
        """
        This method calculate values for AU13, AU14, AU24
        :param blendshapes_dict: blendshape dictionary
        :return: au13, au14, au24 values
        """
        au13_intensity, au14_intensity, au24_intensity = Constants.ZERO, Constants.ZERO, Constants.ZERO

        if self.blendshape_detection[Constants.MOUTH_DIMPLE_LEFT_IDX]==Constants.ONE\
                or self.blendshape_detection[Constants.MOUTH_DIMPLE_RIGHT_IDX]==Constants.ONE:
            
            au14_intensity_h = self.map_to_intensity(self.rest_face_features[Constants.LIP_OUTER_H_DST_IDX],
                                                   self.extreme_vals[Constants.LIP_H_STRETCH_IDX],
                                                   self.crafted_features[Constants.LIP_OUTER_H_DST_IDX])
            au14_intensity_v = Constants.ZERO
            au14_intensity = max(au14_intensity_h,au14_intensity_v)
            
            
        if self.blendshape_detection[Constants.MOUTH_PRESS_LEFT_IDX]==Constants.ONE \
                or self.blendshape_detection[Constants.MOUTH_PRESS_RIGHT_IDX]==Constants.ONE:
           
            au24_intensity = self.map_to_intensity(Constants.ZERO,
                                                   self.rest_face_features[Constants.LIP_ROLL_HEIGHT_IDX],
                                                   self.crafted_features[Constants.LIP_ROLL_HEIGHT_IDX],
                                                   inverted=True)
            
        if self.blendshape_detection[Constants.CHEEK_PUFF_IDX]==Constants.ONE:
            au13_intensity = max(au14_intensity, self.au20_intensity, self.au12_intensity)

        return au13_intensity, au14_intensity, au24_intensity

    def is_au_12_20_26_27(self, au25Intensity):
        """
        This method calculate values for AU20, AU26, AU27
        :param blendshapes_dict: blendshape dictionary
        :return: au20, au26, au27 values
        """
        self.au12_intensity, self.au20_intensity, au26_intensity, au27_intensity = Constants.ZERO, Constants.ZERO, Constants.ZERO, Constants.ZERO
        # jaw open
        if self.blendshape_detection[Constants.JAW_OPEN_IDX]==Constants.ONE:
            
            if self.extreme_vals[Constants.JAW_DROP_IDX]<self.crafted_features[Constants.LIP_INNER_V_DST_IDX]:
                au26_intensity = Constants.FIVE
            else:
                au26_intensity = self.map_to_intensity(self.extreme_vals[Constants.LIPS_PART_IDX],
                                                       self.extreme_vals[Constants.JAW_DROP_IDX],
                                                       self.crafted_features[Constants.LIP_INNER_V_DST_IDX])
                
                au26_intensity_nose_chin = Constants.ZERO
                if au25Intensity<=Constants.ONE:
                
                    au26_intensity_nose_chin = self.map_to_intensity(self.rest_face_features[Constants.NOSE_TIP_CHIN_DST_IDX],
                                                                    self.extreme_vals[Constants.MOUTH_CLOSE_JAW_DROP_MAX_IDX],
                                                                    self.crafted_features[Constants.NOSE_TIP_CHIN_DST_IDX])
                
                au26_intensity = max(au26_intensity, au26_intensity_nose_chin)
                

        # lip corner pull
        if self.blendshape_detection[Constants.MOUTH_SMILE_LEFT_IDX]==Constants.ONE\
            or self.blendshape_detection[Constants.MOUTH_SMILE_RIGHT_IDX]==Constants.ONE:
            au12_intensity_h = self.map_to_intensity(self.rest_face_features[Constants.LIP_OUTER_H_DST_IDX],
                                                   self.extreme_vals[Constants.LIP_H_STRETCH_IDX],
                                                   self.crafted_features[Constants.LIP_OUTER_H_DST_IDX])
            if self.crafted_features[Constants.LEFT_LIP_V_IDX]<=Constants.ZERO and self.crafted_features[Constants.RIGHT_LIP_V_IDX]<=Constants.ZERO:
                au12_intensity_v = Constants.FIVE
            else:
                au12_intensity_v = self.map_to_intensity(Constants.ZERO,
                                                    self.extreme_vals[Constants.LIP_V_STRETCH_IDX],
                                                    max(self.crafted_features[Constants.LEFT_LIP_V_IDX],self.crafted_features[Constants.RIGHT_LIP_V_IDX]),
                                                    inverted=True)

            self.au12_intensity = max(au12_intensity_h,au12_intensity_v)

        # lip strech
        if self.blendshape_detection[Constants.MOUTH_STRETCH_LEFT_IDX]==Constants.ONE\
                or self.blendshape_detection[Constants.MOUTH_STRETCH_RIGHT_IDX]==Constants.ONE:

            self.au20_intensity = self.map_to_intensity(self.rest_face_features[Constants.LIP_OUTER_H_DST_IDX],
                                                   self.extreme_vals[Constants.LIP_H_STRETCH_IDX],
                                                   self.crafted_features[Constants.LIP_OUTER_H_DST_IDX])

        if self.blendshape_detection[Constants.JAW_OPEN_IDX]==Constants.ONE:
            
            if self.extreme_vals[Constants.MOUTH_STRETCH_V_IDX]<self.crafted_features[Constants.LIP_INNER_V_DST_IDX]:
                au27_intensity = Constants.FIVE
            else:
                
                au27_intensity = self.map_to_intensity(self.extreme_vals[Constants.LIPS_PART_IDX],
                                                       self.extreme_vals[Constants.MOUTH_STRETCH_V_IDX],
                                                       self.crafted_features[Constants.LIP_INNER_V_DST_IDX])
        
        return self.au12_intensity, self.au20_intensity, au26_intensity, au27_intensity

    def is_au_10_15(self):
        """
        This method calculate values for AU10, AU12, AU15
        :param blendshapes_dict: blendshape dictionary
        :return: au10, au12, au15 values
        """
        au10_intensity, au15_intensity = Constants.ZERO, Constants.ZERO
        if  self.blendshape_detection[Constants.MOUTH_FROWN_LEFT_IDX]==Constants.ONE\
            or self.blendshape_detection[Constants.MOUTH_FROWN_RIGHT_IDX]==Constants.ONE:

            au15_intensity_h = self.map_to_intensity(self.rest_face_features[Constants.LIP_OUTER_H_DST_IDX],
                                                   self.extreme_vals[Constants.LIP_H_STRETCH_IDX],
                                                   self.crafted_features[Constants.LIP_OUTER_H_DST_IDX])
            
            au15_intensity_v = self.map_to_intensity(Constants.ZERO,
                                                   self.rest_face_features[Constants.LIP_DOWN_V_DST_IDX],
                                                   self.crafted_features[Constants.LIP_DOWN_V_DST_IDX],
                                                   inverted=True)
            au15_intensity = max(au15_intensity_h,au15_intensity_v)


        if (self.blendshape_detection[Constants.MOUTH_SHRUG_UPPER_IDX]==Constants.ONE\
            or self.blendshape_detection[Constants.MOUTH_UPPER_UP_LEFT_IDX]==Constants.ONE\
            or self.blendshape_detection[Constants.MOUTH_UPPER_UP_RIGHT_IDX]==Constants.ONE):
            
            au10_intensity = self.map_to_intensity(self.extreme_vals[Constants.UPPER_LIP_RAISE_IDX],
                                                   self.rest_face_features[Constants.UPPER_LIP_NOSE_TIP_DST_IDX],
                                                   self.crafted_features[Constants.UPPER_LIP_NOSE_TIP_DST_IDX],
                                                   inverted=True)
        return au10_intensity, au15_intensity
    
    def is_au17(self):
        """
        This method calculate the values for AU17
        :param blendshapes_dict: blenshape dictinoary
        :return: au17
        """
        au17_intensity = Constants.ZERO
        if (self.blendshape_detection[Constants.MOUTH_ROLL_LOWER_IDX]==Constants.ONE\
            or self.blendshape_detection[Constants.MOUTH_SHRUG_LOWER_IDX]==Constants.ONE)\
            and self.blendshape_detection[Constants.JAW_OPEN_IDX]!=Constants.ONE:
            au17_intensity = self.map_to_intensity(Constants.POINT_SEVEN_FIVE*self.rest_face_features[Constants.NOSE_TIP_CHIN_DST_IDX],
                                                                 self.rest_face_features[Constants.NOSE_TIP_CHIN_DST_IDX],
                                                                 self.crafted_features[Constants.NOSE_TIP_CHIN_DST_IDX],
                                                                 inverted=True)
        return au17_intensity
    
    def is_au09(self):
        """
        This method calculate the value for AU09
        :param blendshapes_dict: blendshape dict
        :param brow_lower: AU4 value
        :return:
        """
        au9_intensity = Constants.ZERO
        if (self.blendshape_detection[Constants.BROW_DOWN_LEFT_IDX]==Constants.ONE and self.blendshape_detection[Constants.INV_NOSE_SNEER_LEFT_IDX]==Constants.ONE)\
        or (self.blendshape_detection[Constants.BROW_DOWN_RIGHT_IDX]==Constants.ONE and self.blendshape_detection[Constants.INV_NOSE_SNEER_RIGHT_IDX]==Constants.ONE): # add other


            au9_intensity_l = self.map_to_intensity(self.rest_face_features[Constants.LEFT_NOSE_BRIDGE_TIP_MIN_DST_IDX],
                                                     self.rest_face_features[Constants.LEFT_NOSE_BRIDGE_TIP_MAX_DST_IDX],
                                                     self.crafted_features[Constants.LEFT_NOSE_BRIDGE_TIP_MAX_DST_IDX],
                                                     inverted=True)
            au9_intensity_r = self.map_to_intensity(self.rest_face_features[Constants.RIGHT_NOSE_BRIDGE_TIP_MIN_DST_IDX],
                                                     self.rest_face_features[Constants.RIGHT_NOSE_BRIDGE_TIP_MAX_DST_IDX],
                                                     self.crafted_features[Constants.RIGHT_NOSE_BRIDGE_TIP_MAX_DST_IDX],
                                                     inverted=True)
            au9_intensity = max(au9_intensity_l, au9_intensity_r)
        return au9_intensity         

    
    def is_au06(self):
        """
        This method calculate the value for AU06
        :param blendshapes_dict: blendhape dict
        :param au7: au7 value
        :return: au6
        """
        au6_intensity = Constants.ZERO
        if (self.blendshape_detection[Constants.INV_CHEEK_SQUINT_LEFT_IDX]==Constants.ONE or self.blendshape_detection[Constants.EYE_SQUINT_LEFT_IDX]==Constants.ONE)\
            or (self.blendshape_detection[Constants.INV_CHEEK_SQUINT_RIGHT_IDX]==Constants.ONE or self.blendshape_detection[Constants.EYE_SQUINT_RIGHT_IDX]==Constants.ONE):
            
            au6_intensity_l = self.map_to_intensity(self.rest_face_features[Constants.LEFT_CHEEK_RAISE_UP_DST_IDX],
                                                   self.rest_face_features[Constants.LEFT_CHEEK_RAISE_DST_IDX],
                                                   self.crafted_features[Constants.LEFT_CHEEK_RAISE_DST_IDX],
                                                   inverted=True)
            au6_intensity_r = self.map_to_intensity(self.rest_face_features[Constants.RIGHT_CHEEK_RAISE_UP_DST_IDX],
                                                   self.rest_face_features[Constants.RIGHT_CHEEK_RAISE_DST_IDX],
                                                   self.crafted_features[Constants.RIGHT_CHEEK_RAISE_DST_IDX],
                                                   inverted=True)
            au6_intensity = max(au6_intensity_l, au6_intensity_r)
        return au6_intensity
    
    def is_au11(self):
        """
        This method calculate the value for AU06
        :param blendshapes_dict: blendhape dict
        :param au7: au7 value
        :return: au6
        """
        au11_intensity= Constants.ZERO
        

        if (self.blendshape_detection[Constants.BROW_DOWN_LEFT_IDX]==Constants.ONE and self.blendshape_detection[Constants.INV_NOSE_SNEER_LEFT_IDX]==Constants.ONE)\
        or (self.blendshape_detection[Constants.BROW_DOWN_RIGHT_IDX]==Constants.ONE and self.blendshape_detection[Constants.INV_NOSE_SNEER_RIGHT_IDX]==Constants.ONE): # add other
  
            au11_intensity_l = self.map_to_intensity(self.rest_face_features[Constants.LEFT_NASOLABIAL_MIN_DST_IDX],
                                                   self.rest_face_features[Constants.LEFT_NASOLABIAL_MAX_DST_IDX],
                                                   self.crafted_features[Constants.LEFT_NASOLABIAL_MIN_DST_IDX])
            au11_intensity_r = self.map_to_intensity(self.rest_face_features[Constants.RIGHT_NASOLABIAL_MIN_DST_IDX],
                                                   self.rest_face_features[Constants.RIGHT_NASOLABIAL_MAX_DST_IDX],
                                                   self.crafted_features[Constants.RIGHT_NASOLABIAL_MIN_DST_IDX])
            au11_intensity = max(au11_intensity_l, au11_intensity_r)
        return au11_intensity

    def is_au25(self):
        """
        This method calculate the value AU25
        :param blendshapes_dict: blendshape dict
        :return: jaw
        """
        self.au25_intensity = Constants.ZERO

        if  self.blendshape_detection[Constants.MOUTH_LOWER_DOWN_LEFT_IDX]==Constants.ONE\
            or self.blendshape_detection[Constants.MOUTH_LOWER_DOWN_RIGHT_IDX]==Constants.ONE\
            or (self.blendshape_detection[Constants.MOUTH_UPPER_UP_LEFT_IDX]==Constants.ONE or self.blendshape_detection[Constants.MOUTH_UPPER_UP_RIGHT_IDX]==Constants.ONE)\
            or self.blendshape_detection[Constants.JAW_OPEN_IDX]==Constants.ONE:
            
            if self.extreme_vals[Constants.LIPS_PART_IDX]<self.crafted_features[Constants.LIP_INNER_V_DST_IDX]:
                self.au25_intensity = Constants.FIVE
            else:
                
                self.au25_intensity = self.map_to_intensity(Constants.ZERO,
                                                       self.extreme_vals[Constants.LIPS_PART_IDX],
                                                       self.crafted_features[Constants.LIP_INNER_V_DST_IDX])
        return self.au25_intensity


    def is_au23(self):
        """
        This method to calculate the value of AU23
        :param blendshapes_dict: blendshape dict
        :return: au23
        """
        au23_intensity = Constants.ZERO
        if self.blendshape_detection[Constants.MOUTH_ROLL_LOWER_IDX]==Constants.ONE\
           or self.blendshape_detection[Constants.MOUTH_ROLL_UPPER_IDX]==Constants.ONE:
                # dst based
                au23_intensity_1 = self.map_to_intensity(Constants.ZERO, self.rest_face_features[Constants.LIP_TIGHT_D1_IDX],
                                                    self.crafted_features[Constants.LIP_TIGHT_D1_IDX],
                                                    inverted=True)
                au23_intensity_2 = self.map_to_intensity(Constants.ZERO, self.rest_face_features[Constants.LIP_TIGHT_D2_IDX],
                                                    self.crafted_features[Constants.LIP_TIGHT_D2_IDX],
                                                    inverted=True)
                au23_intensity_3 = self.map_to_intensity(Constants.POINT_TWO_FIVE * self.rest_face_features[Constants.LIP_TIGHT_D3_IDX],
                                                    self.rest_face_features[Constants.LIP_TIGHT_D3_IDX],
                                                    self.crafted_features[Constants.LIP_TIGHT_D3_IDX],
                                                    inverted=True)
                au23_intensity_4 = self.map_to_intensity(Constants.POINT_TWO_FIVE * self.rest_face_features[Constants.LIP_TIGHT_D4_IDX],
                                                    self.rest_face_features[Constants.LIP_TIGHT_D4_IDX],
                                                    self.crafted_features[Constants.LIP_TIGHT_D4_IDX],
                                                    inverted=True)
                au23_intensity = max(au23_intensity_1,au23_intensity_2, au23_intensity_3, au23_intensity_4)
        return au23_intensity
    

    def nose_cheek_aus_intensity_calc(self,au_dict):
        au_dict[Constants.AU6_KEY] = self.is_au06()
        au_dict[Constants.AU11_KEY] = self.is_au11()
        au_dict[Constants.AU9_KEY] = self.is_au09()
        return au_dict
    

    def eye_aus_intensity_calc(self,au_dict):
        au_dict[Constants.AU43_KEY], au_dict[Constants.AU45_KEY], au_dict[
            Constants.AU46_KEY] = self.is_au_43_45_46()
        au_dict[Constants.AU61_KEY], au_dict[Constants.AU62_KEY], au_dict[Constants.AU63_KEY], au_dict[
            Constants.AU64_KEY] = self.is_au_61_62_63_64()
        au_dict[Constants.AU2_KEY] = self.is_au_02()
        au_dict[Constants.AU1_KEY], au_dict[Constants.AU4_KEY] = self.is_au_01_04()
        au_dict[Constants.AU5_KEY], au_dict[Constants.AU7_KEY] = self.is_au_05_07()
        au_dict[Constants.AU41_KEY], au_dict[Constants.AU42_KEY] = self.is_au_41_42()
        au_dict[Constants.AU44_KEY] = self.is_au_44()
        return au_dict

    def mouth_aus_intensity_calc(self, au_dict):
        au_dict[Constants.AU16_KEY] = self.is_au_16()
        au_dict[Constants.AU17_KEY] = self.is_au17()       
        au_dict[Constants.AU25_KEY] = self.is_au25()
        au_dict[Constants.AU12_KEY], au_dict[Constants.AU20_KEY], au_dict[Constants.AU26_KEY], au_dict[Constants.AU27_KEY] = self.is_au_12_20_26_27(au_dict[Constants.AU25_KEY])
        au_dict[Constants.AU13_KEY], au_dict[Constants.AU14_KEY], au_dict[Constants.AU24_KEY] = self.is_au_13_14_24()
        au_dict[Constants.AU23_KEY] = self.is_au23()
        au_dict[Constants.AU18_KEY], au_dict[Constants.AU22_KEY] = self.is_au_18_22()
        au_dict[Constants.AU10_KEY], au_dict[Constants.AU15_KEY] = self.is_au_10_15()
        au_dict[Constants.AU28_KEY] = self.is_au_28()
        return au_dict
    

    def inference(self, blendshape_detection,blendshapes_dict,crafted_features, rest_face_features, extreme_vals,
                  mouth_motion, nose_motion, eye_motion):
        """
        This method call generate the blendshapes and call the necessary methods to calculate the au values
        :param frame: input frame
        :return:
            au_dict: AU values
            mesh_frame: black frame with face landmarks marked
            blendshape_dict: contains blendshape values
        """
        self.blendshape_detection = blendshape_detection
        self.blendshapes_dict = blendshapes_dict
        self.crafted_features = crafted_features
        self.rest_face_features = rest_face_features
        self.extreme_vals = extreme_vals

        au_dict = Constants.AU_DICT

        if mouth_motion:
            au_dict = self.mouth_aus_intensity_calc(au_dict)
        if nose_motion:
            au_dict = self.nose_cheek_aus_intensity_calc(au_dict)
        if eye_motion:
            au_dict = self.eye_aus_intensity_calc(au_dict)

        sadness, surprise, fear, anger, disgust, contempt, joy = self.emotion_detector.detect_emotion(au_dict, blendshape_detection)

        frame_emotion = {Constants.HAPPY_KEY: joy, Constants.SAD_KEY: sadness, Constants.SURPRISE_KEY: surprise, Constants.FEAR_KEY: fear,
                         Constants.ANGER_KEY: anger, Constants.DISGUST_KEY: disgust, Constants.CONTEMPT_KEY: contempt}
        self.frame_list.append(self.count)
        self.count += Constants.ONE
        self.get_emotion_time_dict(frame_emotion)
        self.pain_detector.detect_pain(au_dict, blendshape_detection)

        return au_dict, frame_emotion, self.emotion_time_dict, self.frame_list





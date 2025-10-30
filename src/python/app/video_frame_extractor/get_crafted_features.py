import math
from src.python.app.constants.constants import Constants


class FeatureCAlculator:

    def __init__(self):
        self.face_width = Constants.ZERO
        self.face_height = Constants.ZERO
        self.face_diagonal = Constants.ZERO
        self.diagonal_angle = Constants.ZERO

    # Function to calculate Euclidean distance between two points (x, y)
    def calculate_distance(self, p1, p2,pri=None):
        normalized_distance = self.normalize_distance_with_diagonal(p1, p2,pri)
        return normalized_distance

    def normalize_distance_with_diagonal(self, point1, point2,pri=None):
        """
        Normalize the Euclidean distance between two points using the face diagonal's projected distance.
        
        Args:
            face_width (float): Width of the face.
            face_height (float): Height of the face.
            face_diagonal (float): Diagonal length of the face.
            diagonal_angle (float): Angle of the face diagonal with the horizontal (in radians).
            point1 (tuple): Coordinates of the first point (x1, y1).
            point2 (tuple): Coordinates of the second point (x2, y2).
        
        Returns:
            dict: Normalized distance and intermediate values.
        """
        x1, y1 = point1
        x2, y2 = point2

        if x1==Constants.ZERO and x2==Constants.ZERO:
            diff = y2-y1
            if diff<Constants.ZERO:
                distance_points = Constants.ZERO
            else:
                distance_points = diff
        else:
            # Calculate the Euclidean distance between the two points
            distance_points = math.sqrt((x2 - x1)**Constants.TWO + (y2 - y1)**Constants.TWO)


        if pri=='p':
            print(distance_points, x1, y1, x2,y2)
        
        # Calculate the angle of the line joining the two points
        line_angle = math.atan2(y2 - y1, x2 - x1)  # Angle in radians

        cos_phi = abs(math.cos(line_angle))
        sin_phi = abs(math.sin(line_angle))
        projected_diagonal = (self.face_width * cos_phi + self.face_height * sin_phi) 
        
        # Normalize the distance
        normalized_distance = distance_points / projected_diagonal
       
        # Return results and intermediate values
        return  normalized_distance

    def get_mouth_features(self, points):
        # mouth features
        lip_corner_inner_h_distance = self.calculate_distance(points[78],points[308]) 
        lip_corner_outer_h_distance = self.calculate_distance(points[61],points[291])
        lip_inner_v_distance = self.calculate_distance(points[13],points[14])
        lip_outer_v_distance = self.calculate_distance(points[Constants.ZERO],points[17])
        mouth_mean_x, mouth_mean_y = self.mean_of_points(points, Constants.OUTER_MOUTH_IDXS)
        mouth_cen_chin_tip_dst = self.calculate_distance(points[175], (mouth_mean_x, mouth_mean_y))
        lip_upper_nose_tip_distance = self.calculate_distance(points[Constants.ZERO],points[4])
       
        lip_upper_roll_height = self.calculate_distance(points[13],points[Constants.ZERO])
        lip_lower_roll_height = self.calculate_distance(points[14],points[17])

        # lip corner vrtical motions
        left_out_lip_corner_v_dst = self.calculate_distance((Constants.ZERO, points[Constants.ZERO][Constants.ONE]), (Constants.ZERO, points[291][Constants.ONE]) ) 
        right_out_lip_corner_v_dst = self.calculate_distance((Constants.ZERO, points[Constants.ZERO][Constants.ONE]), (Constants.ZERO, points[61][Constants.ONE]) )

        mouth_out_mids_dst = self.calculate_distance(points[164], points[18])
        nose_tip_chin_distance = self.calculate_distance(points[4], points[199])

        # distances to calculate lip tightner
        lip_tight_d1 = self.calculate_distance(points[37], points[82])
        lip_tight_d2 = self.calculate_distance(points[267], points[312])
        lip_tight_d3 = self.calculate_distance(points[84], points[87])
        lip_tight_d4 = self.calculate_distance(points[314], points[317])

        # nose top outer lower lip
        lip_lower_out_nose_tip_distance = self.calculate_distance(points[Constants.ONE], points[Constants.FOUR])

        # lip roll
        lip_left_h = self.calculate_distance(points[267], points[314])  
        lip_right_h = self.calculate_distance(points[87], points[84])
        lip_height = (lip_left_h+lip_right_h)/Constants.TWO

        #lip corner depressor
        leftDownOutLipCornerVDistance = self.calculate_distance((Constants.ZERO,points[291][Constants.ONE]) , (Constants.ZERO,points[17][Constants.ONE]))
        rightDownOutLipCornerVDistance = self.calculate_distance((Constants.ZERO,points[61][Constants.ONE]) , (Constants.ZERO,points[17][Constants.ONE]))
        downOutLipCornerVDistance = (leftDownOutLipCornerVDistance+rightDownOutLipCornerVDistance)/Constants.TWO

        return [lip_corner_outer_h_distance,lip_outer_v_distance, lip_inner_v_distance, mouth_cen_chin_tip_dst,
                lip_upper_nose_tip_distance,lip_upper_roll_height,lip_lower_roll_height, left_out_lip_corner_v_dst,
                right_out_lip_corner_v_dst,mouth_out_mids_dst, nose_tip_chin_distance,lip_corner_inner_h_distance,
                lip_tight_d1, lip_tight_d2, lip_tight_d3, lip_tight_d4,
                lip_lower_out_nose_tip_distance,lip_height, downOutLipCornerVDistance]
    


    def get_nose_features(self, points):
        # au9 impl
        right_nose_bridge_tip_max_distance = self.calculate_distance( (Constants.ZERO,points[168][Constants.ONE]), (Constants.ZERO, points[49][Constants.ONE]))
        right_nose_bridge_tip_min_distance = self.calculate_distance( (Constants.ZERO,points[168][Constants.ONE]) ,(Constants.ZERO,points[209][Constants.ONE]))
        left_nose_bridge_tip_max_distance = self.calculate_distance( (Constants.ZERO,points[168][Constants.ONE]) ,(Constants.ZERO,points[279][Constants.ONE]))
        left_nose_bridge_tip_min_distance = self.calculate_distance( (Constants.ZERO,points[168][Constants.ONE]) ,(Constants.ZERO,points[429][Constants.ONE]))

        # au 11
        right_nasolabial_max_distance = self.calculate_distance(points[4], points[48])
        right_nasolabial_min_distance = self.calculate_distance(points[4], points[115])
        left_nasolabial_max_distance = self.calculate_distance(points[4], points[278])
        left_nasolabial_min_distance = self.calculate_distance(points[4], points[344])

        # au 6 - cheek
        left_inner_eye_mean_x, left_inner_eye_mean_y = self.mean_of_points(points, Constants.LEFT_INNER_EYE)
        right_inner_eye_mean_x, right_inner_eye_mean_y =  self.mean_of_points(points, Constants.RIGHT_INNER_EYE)

        left_cheek_raise_d1 = self.calculate_distance(points[340], (left_inner_eye_mean_x, left_inner_eye_mean_y))
        left_cheek_raise_d2 = self.calculate_distance(points[346], (left_inner_eye_mean_x, left_inner_eye_mean_y))
        left_cheek_raise_d3 = self.calculate_distance(points[347], (left_inner_eye_mean_x, left_inner_eye_mean_y))
        left_cheek_raise_d5 = self.calculate_distance(points[349], (left_inner_eye_mean_x, left_inner_eye_mean_y))
        
        right_cheek_raise_d1 = self.calculate_distance(points[111], (right_inner_eye_mean_x, right_inner_eye_mean_y))

        right_cheek_raise_d2 = self.calculate_distance(points[117], (right_inner_eye_mean_x, right_inner_eye_mean_y))
        
        right_cheek_raise_d3 = self.calculate_distance(points[118], (right_inner_eye_mean_x, right_inner_eye_mean_y))
        right_cheek_raise_d5 = self.calculate_distance(points[120], (right_inner_eye_mean_x, right_inner_eye_mean_y))

        left_cheek_raise_min_d1 = self.calculate_distance(points[261], (left_inner_eye_mean_x, left_inner_eye_mean_y))
        left_cheek_raise_min_d2 = self.calculate_distance(points[448], (left_inner_eye_mean_x, left_inner_eye_mean_y))
        left_cheek_raise_min_d3 = self.calculate_distance(points[449], (left_inner_eye_mean_x, left_inner_eye_mean_y))
        left_cheek_raise_min_d5 = self.calculate_distance(points[451], (left_inner_eye_mean_x, left_inner_eye_mean_y))
        
        right_cheek_raise_min_d1 = self.calculate_distance(points[31], (right_inner_eye_mean_x, right_inner_eye_mean_y))
        right_cheek_raise_min_d2 = self.calculate_distance(points[228], (right_inner_eye_mean_x, right_inner_eye_mean_y))
        right_cheek_raise_min_d3 = self.calculate_distance(points[229], (right_inner_eye_mean_x, right_inner_eye_mean_y))
        right_cheek_raise_min_d5 = self.calculate_distance(points[231], (right_inner_eye_mean_x, right_inner_eye_mean_y))


        
        left_cheek_raise_dst = (left_cheek_raise_d1+ left_cheek_raise_d2+ left_cheek_raise_d3+  left_cheek_raise_d5)/Constants.FOUR
        right_cheek_raise_dst = (right_cheek_raise_d1+ right_cheek_raise_d2+ right_cheek_raise_d3+  right_cheek_raise_d5)/Constants.FOUR
        left_cheek_raise_min_dst = (left_cheek_raise_min_d1+ left_cheek_raise_min_d2+ left_cheek_raise_min_d3+ left_cheek_raise_min_d5)/Constants.FOUR
        right_cheek_raise_min_dst = (right_cheek_raise_min_d1+ right_cheek_raise_min_d2+ right_cheek_raise_min_d3+ right_cheek_raise_min_d5)/Constants.FOUR

        return [# au6, au9 and au11
                right_nose_bridge_tip_max_distance, right_nose_bridge_tip_min_distance,
                left_nose_bridge_tip_max_distance, left_nose_bridge_tip_min_distance,

                right_nasolabial_max_distance, right_nasolabial_min_distance,
                left_nasolabial_max_distance, left_nasolabial_min_distance,

                # cheek
                left_cheek_raise_dst, right_cheek_raise_dst,
                left_cheek_raise_min_dst, right_cheek_raise_min_dst
                ]



    def get_mouth_extreme_values(self, crafted_features):
        lip_max_stretch_h = (Constants.HALF) * crafted_features[Constants.FACE_WIDTH_IDX]
        lip_max_stretch_v = (Constants.POINT_SEVEN_FIVE) * crafted_features[Constants.LIP_OUTER_V_DST_IDX]
        lip_corner_depress_v = (Constants.ONE) * crafted_features[Constants.LIP_OUTER_V_DST_IDX]
        lip_frown_pull_max_v = crafted_features[Constants.LIP_OUTER_V_DST_IDX]

        # lip part, jaw drop, mouth stretch
        lip_part_max = (Constants.ONE_THIRD) * crafted_features[Constants.MOUTH_CEN_CHIN_DST_IDX]
        jaw_drop_max = (Constants.TWO_THIRD) * crafted_features[Constants.MOUTH_CEN_CHIN_DST_IDX]
        mouth_stretch_v_max = crafted_features[Constants.MOUTH_CEN_CHIN_DST_IDX]
        lip_close_jaw_drop_max = (Constants.ONE_POINT_ONE) * crafted_features[Constants.NOSE_TIP_CHIN_DST_IDX]
        

        # upper lip raiser
        lip_raise_max = (Constants.HALF)*crafted_features[Constants.UPPER_LIP_NOSE_TIP_DST_IDX]

        # lip puckerer, funnel and tightner
        lip_fpt_min_width = (Constants.HALF) * crafted_features[Constants.LIP_OUTER_H_DST_IDX]
        lip_fpt_max_width = (Constants.TWO_THIRD) * crafted_features[Constants.LIP_OUTER_H_DST_IDX]  #- 33% of lip_h to 50% of lip h
        lip_fp_max_height = (1.5) * crafted_features[Constants.MOUTH_OUT_V_IDX] #- more than this and it'll be pucker, intensity based on lip width - funnerl, puckerer, tightner
        lip_tight_min_height = (Constants.POINT_EIGHT) * crafted_features[Constants.MOUTH_OUT_V_IDX]
        lip_tight_max_height = (Constants.ONE_TWO_FIVE) * crafted_features[Constants.MOUTH_OUT_V_IDX]

        # lip roll - not used kept to not mess indices
        upper_lip_roll_max = crafted_features[Constants.LIP_INNER_V_DST_IDX] # lip v inner 0.33 to 05
        lower_lip_roll_max = crafted_features[Constants.LIP_INNER_V_DST_IDX] # lip v inner 0.33 to 0.5

        # lip press and lip roll
        lip_press_max = crafted_features[Constants.LIP_OUTER_V_DST_IDX]
        lip_press_min = Constants.THREE* crafted_features[Constants.LIP_INNER_V_DST_IDX]

        
        return [lip_max_stretch_h, lip_max_stretch_v, lip_frown_pull_max_v, lip_part_max, jaw_drop_max, mouth_stretch_v_max, lip_raise_max,
                lip_fpt_min_width,lip_fpt_max_width, lip_fp_max_height, lip_tight_max_height,lip_tight_min_height, upper_lip_roll_max, lower_lip_roll_max, lip_press_max,lip_press_min,
                lip_close_jaw_drop_max,lip_corner_depress_v]

    def get_eye_extreme_values(self, crafted_features):
        # eye features
        # upper lib raiser and lip tight features
        left_eye_wide_max = (Constants.POINT_TWO_FIVE) * crafted_features[Constants.LEFT_EYE_WIDE_IDX-33]
        right_eye_wide_max = (Constants.POINT_TWO_FIVE) * crafted_features[Constants.RIGHT_EYE_WIDE_IDX-33]

        left_max_lid_tight_dist = (Constants.POINT_SEVEN_FIVE) * crafted_features[Constants.LEFT_EYE_INNER_DST_IDX-33]
        right_max_lid_tight_dist = (Constants.POINT_SEVEN_FIVE) * crafted_features[Constants.RIGHT_EYE_INNER_DST_IDX-33]

        # inner brow raiser and outer brow raiser
        left_inner_brow_raiser_max = crafted_features[Constants.EYE_WIDTH_INDEX-33]
        right_inner_brow_raiser_max = crafted_features[Constants.EYE_WIDTH_INDEX-33]

        left_brow_lower_max = Constants.POINT_FOUR * crafted_features[Constants.EYE_WIDTH_INDEX-33]
        right_brow_lower_max = Constants.POINT_FOUR * crafted_features[Constants.EYE_WIDTH_INDEX-33]

        left_outer_brow_raiser_max =crafted_features[Constants.EYE_WIDTH_INDEX-33]
        right_outer_brow_raiser_max = crafted_features[Constants.EYE_WIDTH_INDEX-33]

        left_eye_droop_max = (Constants.TWO_THIRD) * crafted_features[Constants.LEFT_EYE_INNER_DST_IDX-33]
        right_eye_droop_max = (Constants.TWO_THIRD) * crafted_features[Constants.RIGHT_EYE_INNER_DST_IDX-33]

        left_eye_slit_max = (Constants.POINT_TWO_FIVE) * crafted_features[Constants.LEFT_EYE_INNER_DST_IDX-33]
        right_eye_silt_max = (Constants.POINT_TWO_FIVE) * crafted_features[Constants.RIGHT_EYE_INNER_DST_IDX-33]

        left_eye_squint_max = (Constants.POINT_TWO_FIVE) * crafted_features[Constants.LEFT_EYE_INNER_DST_IDX-33]
        right_eye_squint_max = (Constants.POINT_TWO_FIVE) * crafted_features[Constants.LEFT_EYE_INNER_DST_IDX-33]

        return [ left_eye_wide_max, right_eye_wide_max, left_max_lid_tight_dist,
                right_max_lid_tight_dist, left_inner_brow_raiser_max, right_inner_brow_raiser_max, left_brow_lower_max, right_brow_lower_max,
                left_outer_brow_raiser_max, right_outer_brow_raiser_max, left_eye_droop_max, right_eye_droop_max, left_eye_slit_max, right_eye_silt_max,
                left_eye_squint_max, right_eye_squint_max]
        

    def mean_of_points(self, points,idx_lst):
        mean_x = Constants.ZERO
        mean_y = Constants.ZERO
        for idx in idx_lst:
            mean_x += points[idx][Constants.ZERO]
            mean_y += points[idx][Constants.ONE]
        mean_x/=len(idx_lst)
        mean_y/=len(idx_lst)
        return mean_x, mean_y


    def get_eye_features(self, points):
        # max dst eye wide
        left_eye_wide = self.calculate_distance(points[386], points[257])
        right_eye_wide = self.calculate_distance(points[159], points[27])
        
        # eye close dst
        left_eye_inner_v_distance = self.calculate_distance(points[386], points[374])
        right_eye_inner_v_distance = self.calculate_distance(points[159], points[145])
        
        # eye width
        left_eye_width = self.calculate_distance(points[362], points[263])
        right_eye_width = self.calculate_distance(points[133], points[33])

        eye_width_avg = (left_eye_width + right_eye_width)/(int(bool(left_eye_width)) + int(bool(right_eye_width))+ Constants.EPSILON)

        # brow lowerer
        left_eye_lower_brow_height = self.calculate_distance(points[285], points[362])
        right_eye_lower_brow_height = self.calculate_distance(points[55], points[133])

        left_inner_brow_raiser = self.calculate_distance(points[362], points[285])
        right_inner_brow_raiser = self.calculate_distance(points[133], points[55])
        left_outer_brow_raiser = self.calculate_distance(points[263], points[276])
        right_outer_brow_raiser = self.calculate_distance(points[33], points[46])

        # added dsts
        left_brow_center_dst = self.calculate_distance(points[285], points[8])
        right_brow_center_dst = self.calculate_distance(points[55], points[8])
        
        return [left_eye_wide, right_eye_wide, left_eye_inner_v_distance, right_eye_inner_v_distance, 
                eye_width_avg, left_eye_lower_brow_height, right_eye_lower_brow_height, left_inner_brow_raiser,
                right_inner_brow_raiser, left_outer_brow_raiser, right_outer_brow_raiser, 
                left_brow_center_dst, right_brow_center_dst]


    def get_face_features(self, mesh_array):
        # face width, face height
        self.face_width = max(mesh_array[:,Constants.ZERO])-min(mesh_array[:,Constants.ZERO])
        self.face_height = max(mesh_array[:,Constants.ONE])-min(mesh_array[:,Constants.ONE])
        self.face_diagonal = math.sqrt(self.face_width**Constants.TWO + self.face_height**Constants.TWO)  # Precomputed
        self.diagonal_angle = math.atan2(self.face_height, self.face_width)


    def get_crafted_features(self, mesh_array):

        self.get_face_features(mesh_array)
        # make crafted features
        crafted_features = []
        crafted_features.extend(self.get_mouth_features(mesh_array))
        crafted_features.extend([self.face_width/self.face_width, self.face_height/self.face_height])
        crafted_features.extend(self.get_nose_features(mesh_array))
        crafted_features.extend(self.get_eye_features(mesh_array))
        return crafted_features
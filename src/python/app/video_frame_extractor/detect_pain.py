"""
File: detect_emotion.py
Description:
Created on: 30/09/24
Author: 
E-mail:
"""

from src.python.app.constants.constants import Constants

class DetectPain:
    def __init__(self):
        pass

    def detect_pain(self, au_vals, blendshape_detection):
        # Initialize emotion flags
        pain = Constants.ZERO

     

        # # pain
        # if au_vals[Constants.AU4_KEY]:
        #     if au_vals[Constants.AU6_KEY]:
        #         if au_vals[Constants.AU7_KEY]:
        #             pain = 1
        #             if au_vals[Constants.AU41_KEY] or au_vals[Constants.AU42_KEY] or au_vals[Constants.AU43_KEY]:
        #                 pain = 2
        #             if au_vals[Constants.AU25_KEY] or au_vals[Constants.AU26_KEY] or au_vals[Constants.AU27_KEY]:  
        #                 pain = 3
        #                 if au_vals[Constants.AU41_KEY] or au_vals[Constants.AU42_KEY] or au_vals[Constants.AU43_KEY]:
        #                     pain = 7
        #             if au_vals[Constants.AU23_KEY] or au_vals[Constants.AU24_KEY]:
        #                 pain = 4
        #             if au_vals[Constants.AU9_KEY] or au_vals[Constants.AU11_KEY]:
        #                 pain = 5
        #             if au_vals[Constants.AU12_KEY] or au_vals[Constants.AU20_KEY]:
        #                 pain = 6
        #     elif au_vals[Constants.AU43_KEY]:
        #         if au_vals[Constants.AU20_KEY] or au_vals[Constants.AU12_KEY]:
        #             pain = 19
        #         if au_vals[Constants.AU23_KEY] or au_vals[Constants.AU24_KEY]:
        #             pain = 20

        #     elif au_vals[Constants.AU9_KEY]:
        #         pain = 8
        #         if au_vals[Constants.AU11_KEY]:
        #             pain = 9
        #             if au_vals[Constants.AU23_KEY] or au_vals[Constants.AU24_KEY]:
        #                 pain = 10
        #             if au_vals[Constants.AU12_KEY] or au_vals[Constants.AU20_KEY]:
        #                 pain = 11
        #         if au_vals[Constants.AU17_KEY]:
        #             pain = 12
        #         if au_vals[Constants.AU20_KEY]:
        #             if au_vals[Constants.AU24_KEY]:
        #                 pain = 14
            
        #     elif au_vals[Constants.AU15_KEY]:
        #         if au_vals[Constants.AU17_KEY]:
        #             pain = 13
        # elif au_vals[Constants.AU1_KEY] or au_vals[Constants.AU2_KEY]:
        #     if au_vals[Constants.AU5_KEY]:
        #         if au_vals[Constants.AU20_KEY]:
        #             pain = 15
        #         if au_vals[Constants.AU25_KEY] or au_vals[Constants.AU26_KEY] or au_vals[Constants.AU27_KEY]:
        #             pain = 16
        #         if au_vals[Constants.AU23_KEY] or au_vals[Constants.AU24_KEY]:
        #             pain = 21

        # elif au_vals[Constants.AU43_KEY]:
        #     if au_vals[Constants.AU20_KEY] or au_vals[Constants.AU12_KEY]:
        #         pain = 17
        #     if au_vals[Constants.AU23_KEY] or au_vals[Constants.AU24_KEY]:
        #         pain = 18
            
        # else:
        #     pain = 0


        # pain
        if au_vals[Constants.AU4_KEY]:
            pain = au_vals[Constants.AU4_KEY]
            if au_vals[Constants.AU6_KEY]:
                pain += au_vals[Constants.AU6_KEY]
                if au_vals[Constants.AU7_KEY]:
                    pain += au_vals[Constants.AU7_KEY]

                    

                    if au_vals[Constants.AU25_KEY] or au_vals[Constants.AU26_KEY] or au_vals[Constants.AU27_KEY]:  
                        pain += max(au_vals[Constants.AU25_KEY], au_vals[Constants.AU26_KEY], au_vals[Constants.AU27_KEY])

                        if au_vals[Constants.AU42_KEY] or au_vals[Constants.AU43_KEY]:
                            pain += max( au_vals[Constants.AU42_KEY], au_vals[Constants.AU43_KEY])


                    elif au_vals[Constants.AU42_KEY] or au_vals[Constants.AU43_KEY]:
                        pain += max(au_vals[Constants.AU42_KEY], au_vals[Constants.AU43_KEY])

                    if au_vals[Constants.AU23_KEY] or au_vals[Constants.AU24_KEY]:
                        pain += max(au_vals[Constants.AU23_KEY], au_vals[Constants.AU24_KEY])
                    if au_vals[Constants.AU9_KEY] or au_vals[Constants.AU11_KEY]:
                        pain += max(au_vals[Constants.AU9_KEY], au_vals[Constants.AU11_KEY])
                    if au_vals[Constants.AU12_KEY] or au_vals[Constants.AU20_KEY]:
                        pain += max(au_vals[Constants.AU12_KEY], au_vals[Constants.AU20_KEY])
                elif au_vals[Constants.AU9_KEY]:
                    if au_vals[Constants.AU10_KEY]:
                        pain += au_vals[Constants.AU9_KEY] + au_vals[Constants.AU10_KEY]

            elif au_vals[Constants.AU43_KEY]:
                if au_vals[Constants.AU20_KEY] or au_vals[Constants.AU12_KEY]:
                    pain += max(au_vals[Constants.AU20_KEY], au_vals[Constants.AU12_KEY])
                if au_vals[Constants.AU23_KEY] or au_vals[Constants.AU24_KEY]:
                    pain += max(au_vals[Constants.AU23_KEY], au_vals[Constants.AU24_KEY])
                if au_vals[Constants.AU11_KEY]:
                    pain += au_vals[Constants.AU11_KEY]

            elif au_vals[Constants.AU9_KEY]:
                pain = au_vals[Constants.AU9_KEY]
                if au_vals[Constants.AU11_KEY]:
                    pain = max(au_vals[Constants.AU9_KEY], au_vals[Constants.AU11_KEY])
                    if au_vals[Constants.AU23_KEY] or au_vals[Constants.AU24_KEY]:
                        pain += max(au_vals[Constants.AU23_KEY], au_vals[Constants.AU24_KEY])

                    if au_vals[Constants.AU12_KEY] or au_vals[Constants.AU20_KEY]:
                        pain += max(au_vals[Constants.AU12_KEY], au_vals[Constants.AU20_KEY])
                    
                    if au_vals[Constants.AU10_KEY]:
                        pain += au_vals[Constants.AU10_KEY]

                if au_vals[Constants.AU17_KEY]:
                    pain += au_vals[Constants.AU17_KEY]
                    if au_vals[Constants.AU24_KEY]:
                        pain += au_vals[Constants.AU24_KEY]
                if au_vals[Constants.AU20_KEY]:
                    pain += au_vals[Constants.AU20_KEY]
                    if au_vals[Constants.AU24_KEY]:
                        pain += au_vals[Constants.AU24_KEY]
            
            elif au_vals[Constants.AU15_KEY]:
                if au_vals[Constants.AU17_KEY]:
                    pain += au_vals[Constants.AU15_KEY] + au_vals[Constants.AU17_KEY]

        elif au_vals[Constants.AU1_KEY] or au_vals[Constants.AU2_KEY]:
            if au_vals[Constants.AU5_KEY]:
                pain = max(au_vals[Constants.AU1_KEY], au_vals[Constants.AU2_KEY]) + au_vals[Constants.AU5_KEY]
                if au_vals[Constants.AU20_KEY]:
                    pain += au_vals[Constants.AU20_KEY]
                if au_vals[Constants.AU25_KEY] or au_vals[Constants.AU26_KEY] or au_vals[Constants.AU27_KEY]:
                    pain += max(au_vals[Constants.AU25_KEY], au_vals[Constants.AU26_KEY] or au_vals[Constants.AU27_KEY])
                if au_vals[Constants.AU23_KEY] or au_vals[Constants.AU24_KEY]:
                    pain += max(au_vals[Constants.AU23_KEY], au_vals[Constants.AU24_KEY])

        elif au_vals[Constants.AU43_KEY]==5:
            pain = au_vals[Constants.AU43_KEY]
            if au_vals[Constants.AU20_KEY] or au_vals[Constants.AU12_KEY]:
                pain += max(au_vals[Constants.AU20_KEY], au_vals[Constants.AU12_KEY])
            else:
                pain = 0

            if au_vals[Constants.AU23_KEY] or au_vals[Constants.AU24_KEY]:
                pain += max(au_vals[Constants.AU23_KEY], au_vals[Constants.AU24_KEY])
            else:
                pain = 0

            if au_vals[Constants.AU11_KEY]:
                pain += au_vals[Constants.AU11_KEY]
            
        else:
            pain = 0

        max_depth = 40

        return pain
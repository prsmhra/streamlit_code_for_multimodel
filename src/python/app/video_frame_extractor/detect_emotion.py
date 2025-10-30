"""
File: detect_emotion.py
Description:
Created on: 30/09/24
Author: 
E-mail:
"""

from src.python.app.constants.constants import Constants

class DetectEmotion:
    def __init__(self):
        pass

    def detect_emotion(self, au_vals, blendshape_detection):
        # Initialize emotion flags
        sadness = Constants.ZERO
        surprise = Constants.ZERO
        fear = Constants.ZERO
        anger = Constants.ZERO
        disgust = Constants.ZERO
        contempt = Constants.ZERO
        joy = Constants.ZERO


        # joy
        if (blendshape_detection[Constants.MOUTH_SMILE_LEFT_IDX] and blendshape_detection[Constants.MOUTH_SMILE_RIGHT_IDX]):

            if au_vals[Constants.AU12_KEY]:
                joy = Constants.ONE
                if au_vals[Constants.AU6_KEY]:
                    joy = Constants.ONE
                if au_vals[Constants.AU25_KEY]:
                    joy = Constants.ONE
                if au_vals[Constants.AU6_KEY] and au_vals[Constants.AU25_KEY]:
                    joy = Constants.ONE
                if au_vals[Constants.AU4_KEY] or au_vals[Constants.AU5_KEY]:
                    joy = Constants.ZERO
            elif au_vals[Constants.AU10_KEY] and (au_vals[Constants.AU2_KEY] or au_vals[Constants.AU1_KEY]):
                joy = Constants.ZERO
            else:
                joy = Constants.ZERO
        elif (blendshape_detection[Constants.MOUTH_SMILE_LEFT_IDX] or blendshape_detection[Constants.MOUTH_SMILE_RIGHT_IDX])\
            or (blendshape_detection[Constants.MOUTH_STRETCH_LEFT_IDX] or blendshape_detection[Constants.MOUTH_STRETCH_RIGHT_IDX]):
            # contempt
            if au_vals[Constants.AU12_KEY]:
                contempt = Constants.ONE

                if au_vals[Constants.AU14_KEY]:
                    contempt = Constants.ONE
                    if au_vals[Constants.AU6_KEY]:
                        contempt = Constants.ONE
                if au_vals[Constants.AU6_KEY]:
                    contempt = Constants.ONE
                if au_vals[Constants.AU4_KEY] or au_vals[Constants.AU5_KEY]:
                    contempt = Constants.ZERO
            else:
                contempt = Constants.ZERO

            # elif au_vals[Constants.AU9_KEY]:
            if au_vals[Constants.AU9_KEY]:
                # contempt = Constants.ONE
                if au_vals[Constants.AU6_KEY]:
                    contempt = Constants.ONE
                if au_vals[Constants.AU4_KEY] or au_vals[Constants.AU5_KEY]:
                    contempt = Constants.ZERO
            
            else:
                contempt = Constants.ZERO
        else:
            joy = Constants.ZERO
            contempt = Constants.ZERO

        # surprise
        if au_vals[Constants.AU5_KEY] and au_vals[Constants.AU25_KEY] and au_vals[Constants.AU26_KEY]:
            if au_vals[Constants.AU1_KEY] or au_vals[Constants.AU2_KEY]:
                surprise = Constants.ONE
            if au_vals[Constants.AU6_KEY]:
                surprise = Constants.ONE
            if au_vals[Constants.AU12_KEY]:
                surprise = Constants.ONE
        else:
            surprise = Constants.ZERO

        # fear 
        if au_vals[Constants.AU5_KEY] and au_vals[Constants.AU25_KEY] and au_vals[Constants.AU26_KEY]:
            if au_vals[Constants.AU11_KEY]:
                fear = Constants.ONE
            if au_vals[Constants.AU20_KEY]:
                fear = Constants.ONE
        elif (au_vals[Constants.AU1_KEY] or au_vals[Constants.AU2_KEY]) and au_vals[Constants.AU5_KEY]:
            fear =Constants.ONE
        else:
            fear = Constants.ZERO


        # sadness
        if au_vals[Constants.AU15_KEY]:
            sadness = Constants.ONE
            if au_vals[Constants.AU1_KEY]:
                sadness = Constants.ONE
                if au_vals[Constants.AU4_KEY]:
                    sadness = Constants.ONE
                if au_vals[Constants.AU11_KEY]:
                    sadness = Constants.ONE
                    if au_vals[Constants.AU17_KEY]:
                        sadness = Constants.ONE
                if au_vals[Constants.AU17_KEY]:
                    sadness = Constants.ONE
                    if au_vals[Constants.AU44_KEY]:
                        sadness = Constants.ONE
                if (au_vals[Constants.AU43_KEY] and au_vals[Constants.AU44_KEY]):
                    sadness = Constants.ONE
            if au_vals[Constants.AU12_KEY]>0:
                sadness = Constants.ZERO
        elif au_vals[Constants.AU17_KEY] and au_vals[Constants.AU24_KEY]:  # added
            sadness = Constants.ONE
        else:
            sadness = Constants.ZERO

        # disgust
        if au_vals[Constants.AU9_KEY]:
            if au_vals[Constants.AU15_KEY]:
                disgust = Constants.ONE
            elif au_vals[Constants.AU4_KEY]:
                if (au_vals[Constants.AU7_KEY] and au_vals[Constants.AU17_KEY]):
                    disgust = Constants.ONE
                if au_vals[Constants.AU25_KEY] and au_vals[Constants.AU6_KEY]:
                        disgust = Constants.ONE
                if au_vals[Constants.AU15_KEY] and au_vals[Constants.AU25_KEY]:
                    disgust = Constants.ONE
                if au_vals[Constants.AU15_KEY] and au_vals[Constants.AU7_KEY] and au_vals[Constants.AU43_KEY]:
                    disgust = Constants.ONE
        elif (au_vals[Constants.AU4_KEY] and au_vals[Constants.AU11_KEY]):
            disgust = Constants.ONE
        elif (au_vals[Constants.AU10_KEY] and au_vals[Constants.AU22_KEY]\
              and au_vals[Constants.AU25_KEY] and not au_vals[Constants.AU12_KEY]):
            disgust = Constants.ONE
        else:
            disgust = Constants.ZERO

        # anger
        if au_vals[Constants.AU4_KEY]:
            if (au_vals[Constants.AU5_KEY] and au_vals[Constants.AU23_KEY]):
                anger = Constants.ONE
            elif (au_vals[Constants.AU6_KEY] and au_vals[Constants.AU23_KEY]):
                if au_vals[Constants.AU12_KEY] or au_vals[Constants.AU14_KEY] or au_vals[Constants.AU20_KEY]:
                    anger = Constants.ZERO
                else:
                    anger = Constants.ONE
            elif au_vals[Constants.AU2_KEY]:
                if au_vals[Constants.AU9_KEY]:
                    if au_vals[Constants.AU16_KEY] and au_vals[Constants.AU25_KEY]:
                        anger = Constants.ONE
                    if au_vals[Constants.AU23_KEY]:
                        anger = Constants.ONE
                if au_vals[Constants.AU23_KEY]:
                    anger = Constants.ONE
            elif au_vals[Constants.AU9_KEY]:
                if au_vals[Constants.AU25_KEY]:
                    anger = Constants.ONE
                    if au_vals[Constants.AU16_KEY]:
                        anger = Constants.ONE
                if au_vals[Constants.AU17_KEY]:
                    anger = Constants.ONE
        else:
            anger = Constants.ZERO


          
        return sadness, surprise, fear, anger, disgust, contempt, joy
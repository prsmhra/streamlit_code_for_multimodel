import cv2
import os
import csv
from src.python.app.constants.constants import Constants
from src.python.app.video_frame_extractor.au_detection import AUIntensity
from src.python.app.video_frame_extractor.queueExecution import QueueExecution
from src.python.app.video_frame_extractor.face_detector import FaceMeshDetector
from src.python.app.utils.draw_meshpoints import draw_mesh_points
from src.python.app.video_frame_extractor.image_crop_align import ImageCropAlign

class Infer:
    def __init__(self, path):
        self.path = path
        self.facemesh_detector = FaceMeshDetector()
        self.au_detetcion = AUIntensity()
        self.queue_execution = QueueExecution()
        self.image_crop_align = ImageCropAlign()
        self.emotion_blendshape_data = Constants.EMOTION_GRAPH
        self.emotion_dict = dict()


    def inference(self):
        video_files = self.path
        csv_path =f"{Constants.VISION_OUT_DIR}/{video_files.split(os.sep)[-Constants.ONE].split(Constants.DOT)[Constants.ZERO]}{Constants.DOT}{Constants.CSV_KEY}"
        os.makedirs(Constants.VISION_OUT_DIR, exist_ok=True)
        with open(csv_path, Constants.WRITE_MODE) as f:
            f.write(Constants.COMMA.join(Constants.CSV_HEADER_KEY))
            f.write(Constants.NEWLINE)
        
        with open(csv_path, Constants.APPEND_MODE) as f:
            writer = csv.DictWriter(f, fieldnames=Constants.CSV_HEADER_KEY)
            frame_count = Constants.ONE
            cap = cv2.VideoCapture(video_files)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                # Process the frame (e.g., display it)
                try:
                    landmarks, blendshapes, blendshape_dict, rotations = self.facemesh_detector.generate_blendshapes(
                        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                except Exception as e:
                    print(e)
                else:
                    if len(landmarks) == Constants.ZERO or len(blendshapes) == Constants.ZERO:
                        au_list = Constants.AU_DICT
                        blend_shapes = dict()
                        emotion_data = dict()
                        self.queue_execution.reset_queue()
                    mesh_frame = draw_mesh_points(landmarks, frame)
                    res_dct = self.queue_execution.processQueue(frame, mesh_frame, frame, landmarks,
                                                                            blendshapes, blendshape_dict, rotations)
                    
                    au_list = res_dct[Constants.AU_INTENSITY]
                    blend_shapes = res_dct[Constants.BLENDSHAPE_VALS]
                    emotion_data = res_dct[Constants.EMOTION_DETECTION]
                    final_result = {**au_list, **blend_shapes, **emotion_data}
                    final_result[Constants.FRAME_KEY] = frame_count
                    data_row = {key: final_result.get(key, Constants.ZERO) for key in Constants.CSV_HEADER_KEY}
                    writer.writerow(data_row)
                frame_count += Constants.ONE
            cap.release()
            cv2.destroyAllWindows()
        return csv_path
        

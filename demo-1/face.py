from abc import ABC, abstractmethod
import numpy as np
import logging
logging.getLogger('ultralytics').setLevel(logging.WARNING)
from ultralytics import YOLO
import supervision as sv
import face_recognition
from insightface.app import FaceAnalysis

### ABSTRACT CLASSES ###
class FaceDetection(ABC):
    @abstractmethod
    def detect(self, image):  # return a tuple of boxes [(left, top, right, bottom)] and conf_scores
        pass

class FaceRecognition(ABC):
    @abstractmethod
    def embed(self, image):
        pass

class Tracker(ABC):
    @abstractmethod
    def update(self, boxes, conf_scores):
        pass

### CHILD CLASSES ###
class YOLODetection(FaceDetection):
    def __init__(self, model):
        self.model = YOLO(model)

    def detect(self, image):
        results = self.model(image, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        conf_scores = results.boxes.conf.cpu().numpy()
        boxes = [tuple(map(float, box)) for box in boxes]
        return boxes, conf_scores

class DLIBRecognition(FaceRecognition):
    def embed(self, image):
        return face_recognition.face_encodings(image, [(0, image.shape[1], image.shape[0], 0)])[0]

class ByteTracker(Tracker):
    def __init__(self):
        self.tracker = sv.ByteTrack() # track_buffer, # max_time_lost

    def update(self, boxes, conf_scores): 
        boxes = np.asarray(boxes)
        conf_scores = np.asarray(conf_scores)
        if len(boxes) == 0: 
            return np.array([]), np.array([])
        detections = sv.Detections(
            xyxy=boxes,
            confidence=conf_scores,
        )
        tracked_detections = self.tracker.update_with_detections(detections)
        return tracked_detections.tracker_id, tracked_detections.confidence

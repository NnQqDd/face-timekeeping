from abc import ABC, abstractmethod
import numpy as np
from ultralytics import YOLO
import supervision as sv
class Tracker(ABC):
    @abstractmethod
    def update(self, boxes, conf_scores):
        pass

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

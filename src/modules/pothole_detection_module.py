from ultralytics import YOLO

from src.modules.base_module import BaseModule
from src.pothole_detection.pothole import Pothole


class PotholeDetectionModule(BaseModule):
    def __init__(self,
                 source_module: BaseModule,
                 model_weights: str,
                 detection_threshold: float = 0.5,
                 ):
        super().__init__()
        self.source_module = source_module
        self.model_weights = model_weights
        self.detection_threshold = detection_threshold
        self.model = YOLO(self.model_weights, task='detect')
        self.pothole_registry: list[Pothole] = []

    def perform(self):
        while not self._stop_event.is_set():
            frame = self.source_module.value
            if frame is not None:
                prediction_result = self.model.predict(frame, conf=self.detection_threshold, verbose=False)[0]
                self.pothole_registry = [
                    Pothole(xyxy=xyxy.numpy(), conf=conf)
                    for xyxy, conf in zip(prediction_result.boxes.xyxy, prediction_result.boxes.conf)
                ]
                self.value = self.pothole_registry

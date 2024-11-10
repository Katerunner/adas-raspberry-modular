from typing import Any

from ultralytics import YOLO

from src.modules.base_module import BaseModule
from src.object_tracking.naive_object_tracker import NaiveObjectTracker
from src.traffic_sign_detection.traffic_sign_registry import TrafficSignRegistry


class TrafficSignDetectionModule(BaseModule):
    def __init__(self,
                 source_module: BaseModule,
                 model_weights: str,
                 detection_threshold: float = 0.7,
                 registry_max_size: int = 8,
                 registry_min_occurrences: int = 5,
                 casting_lifetime: int = 3,
                 registry_lifetime: int = 5,
                 object_tracker: Any = None,
                 tracker_max_objects=None,
                 tracker_confidence_threshold=0.8,
                 tracker_feature_threshold=0.95,
                 tracker_position_threshold=0.95,
                 tracker_lifespan=3

                 ):
        """Initialize the TrafficSignDetectionModule with a source module and model weights."""
        super().__init__()
        self.source_module = source_module
        self.model_weights = model_weights
        self.detection_threshold = detection_threshold
        self.model = YOLO(self.model_weights, task='detect')

        self.traffic_sign_registry = TrafficSignRegistry(
            max_size=registry_max_size,
            min_occurrences=registry_min_occurrences,
            casting_lifetime=casting_lifetime,
            registry_lifetime=registry_lifetime
        )

        self.object_tracker = object_tracker or NaiveObjectTracker(
            max_objects=tracker_max_objects,
            confidence_threshold=tracker_confidence_threshold,
            feature_threshold=tracker_feature_threshold,
            position_threshold=tracker_position_threshold,
            lifespan=tracker_lifespan
        )

    def perform(self):
        """Perform traffic sign detection on frames from the source module."""
        while not self._stop_event.is_set():
            frame = self.source_module.value
            if frame is not None:
                prediction_result = self.model.predict(frame, conf=self.detection_threshold, verbose=False)[0]
                ids = self.object_tracker.process_yolo_result(prediction_result=prediction_result)
                self.traffic_sign_registry.from_yolo_results(prediction_result=prediction_result, ids=ids)
                self.value = self.traffic_sign_registry

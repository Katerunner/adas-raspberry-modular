from typing import Any

from ultralytics import YOLO

from src.modules.base_module import BaseModule
from src.moving_object_detection.moving_object_registry import MovingObjectRegistry
from src.object_tracking.naive_object_tracker import NaiveObjectTracker
from src.traffic_light_detection.traffic_light_registry import TrafficLightRegistry


class GeneralObjectDetectionModule(BaseModule):
    def __init__(self,
                 source_module: BaseModule,
                 model_weights: str,
                 detection_threshold: float = 0.7,
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

        self.moving_object_registry = MovingObjectRegistry(max_objects=20)
        self.traffic_light_registry = TrafficLightRegistry(max_lifetime=1)

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
                self.moving_object_registry.from_yolo_result(prediction_result=prediction_result, ids=ids)
                self.traffic_light_registry.from_yolo_result(prediction_result=prediction_result)

                self.value = {
                    "moving_object_registry": self.moving_object_registry,
                    "traffic_light_registry": self.traffic_light_registry
                }

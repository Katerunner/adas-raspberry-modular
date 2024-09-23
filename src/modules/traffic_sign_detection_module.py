from ultralytics import YOLO
from src.modules.base_module import BaseModule


class TrafficSignDetectionModule(BaseModule):
    def __init__(self, source_module: BaseModule, model_weights: str):
        """Initialize the TrafficSignDetectionModule with a source module and model weights."""
        super().__init__()
        self.source_module = source_module
        self.model_weights = model_weights
        self.model = YOLO(self.model_weights, task='detect')

    def perform(self):
        """Perform traffic sign detection on frames from the source module."""
        while not self._stop_event.is_set():
            frame = self.source_module.value
            if frame is not None:
                self.value = self.model(frame, verbose=False)[0].plot()

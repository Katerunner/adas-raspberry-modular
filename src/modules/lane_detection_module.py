import cv2
from ultralytics import YOLO

from src.lane_detection.lane_processor_corrector import LaneProcessorCorrector
from src.lane_detection.lane_registry import LaneRegistry
from src.modules.base_module import BaseModule
from src.lane_detection.lane_processor import LaneProcessor
from src.lane_detection.lane_curve_estimator import LaneCurveEstimator

DEFAULT_LANE_COLORS = {
    0: (255, 0, 0),  # LL: Blue
    1: (0, 255, 0),  # LC: Green
    2: (0, 0, 255),  # RC: Red
    3: (255, 255, 0)  # RR: Cyan
}


class LaneDetectionModule(BaseModule):
    def __init__(
            self,
            source_module: BaseModule,
            model_weights: str,
            confidence_threshold: float = 0.2,
            use_weights: bool = True,
            poly_degree: int = 2,
            ransac_min_samples: int = 2,
            ransac_loss: str = "squared_error",
            n_points: int = 10,
            memory_size: int = 100,
            decay: float = 2.0,
            lane_curve_estimator: LaneCurveEstimator = None,
            lane_processor_corrector: LaneProcessorCorrector = None
    ) -> None:
        super().__init__()
        self.lane_registry = LaneRegistry()
        self.source_module = source_module
        self.model_weights = model_weights
        self.confidence_threshold = confidence_threshold
        self.use_weights = use_weights
        self.model = YOLO(self.model_weights, task='detect')

        # Use an already initialized LaneCurveEstimator if provided, otherwise create a new one
        if lane_curve_estimator is None:
            self.lane_curve_estimator = LaneCurveEstimator(
                image_shape=(1, 1),
                poly_degree=poly_degree,
                ransac_min_samples=ransac_min_samples,
                ransac_loss=ransac_loss,
                n_points=n_points,
                memory_size=memory_size,
                decay=decay
            )
        else:
            self.lane_curve_estimator = lane_curve_estimator
        self.lane_processor_corrector = lane_processor_corrector

    def perform(self):
        """
        Perform the lane detection and correction process.

        Detects lanes using the YOLO model, estimates lane curves, and applies corrections using the
        LaneProcessorCorrector if provided.

        :raises Exception: If an error occurs during the lane detection process.
        """
        while not self._stop_event.is_set():
            frame = self.source_module.value
            if frame is not None:
                lane_results = self.model.predict(frame, verbose=False)

                lane_processor = LaneProcessor.from_yolo_result(
                    yolo_result=lane_results,
                    frame_shape=frame.shape[:2],
                    confidence_threshold=self.confidence_threshold,
                    lane_curve_estimator=self.lane_curve_estimator
                )

                lane_processor.estimate_lane_curves(use_weights=self.use_weights)
                if self.lane_processor_corrector is not None:
                    lane_processor = self.lane_processor_corrector.correct(lane_processor)

                self.lane_registry.update_from_lane_processor(lane_processor)
                self.value = self.lane_registry

    def draw_lanes(self, frame):
        traffic_lane_registry = self.value
        if traffic_lane_registry is None:
            return

        for lane_id in traffic_lane_registry.lane_labels:
            # lane_cls = traffic_lane_registry.lane_labels[lane_id]
            lane_obj = traffic_lane_registry.lanes[lane_id]
            if lane_obj and len(lane_obj.estimated_points) > 0:
                cv2.polylines(
                    frame,
                    [lane_obj.estimated_points],
                    isClosed=False,
                    color=DEFAULT_LANE_COLORS[lane_id],
                    thickness=2
                )
        return frame

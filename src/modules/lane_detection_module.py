from ultralytics import YOLO
from src.modules.base_module import BaseModule
from src.lane_detection.lane_processor import LaneProcessor
from src.lane_detection.lane_curve_estimator import LaneCurveEstimator


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
            lane_curve_estimator: LaneCurveEstimator = None
    ):
        """
        Initialize the LaneDetectionModule.

        :param source_module: The module providing the source frames.
        :type source_module: BaseModule
        :param model_weights: Path to the YOLO model weights.
        :type model_weights: str
        :param confidence_threshold: Minimum confidence threshold for lane detection, defaults to 0.2.
        :type confidence_threshold: float, optional
        :param use_weights: Whether to use confidences as weights for lane curve estimation, defaults to True.
        :type use_weights: bool, optional
        :param poly_degree: Degree of the polynomial for curve estimation, defaults to 2.
        :type poly_degree: int, optional
        :param ransac_min_samples: Minimum samples required for RANSAC fitting, defaults to 2.
        :type ransac_min_samples: int, optional
        :param ransac_loss: Loss function for RANSAC regression, defaults to 'squared_error'.
        :type ransac_loss: str, optional
        :param n_points: Number of predicted points for lane curve estimation, defaults to 10.
        :type n_points: int, optional
        :param memory_size: Maximum memory size for storing lane points, defaults to 100.
        :type memory_size: int, optional
        :param decay: Decay factor for weights, defaults to 2.0.
        :type decay: float, optional
        :param lane_curve_estimator: Pre-initialized LaneCurveEstimator, optional.
        :type lane_curve_estimator: LaneCurveEstimator, optional
        """
        super().__init__()
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

    def perform(self):
        """
        Perform the lane detection and curve estimation process.

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
                self.value = lane_processor

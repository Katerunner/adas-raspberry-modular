from concurrent.futures import ThreadPoolExecutor
from src.lane_detection.lane import Lane
from src.lane_detection.lane_curve_estimator import LaneCurveEstimator


class LaneProcessor:
    lane_labels = {0: "LL", 1: "LC", 2: "RC", 3: "RR"}

    def __init__(self, lanes: list[Lane] = None, lane_curve_estimator: LaneCurveEstimator = None):
        if lanes is None:
            lanes = [None, None, None, None]
        elif len(lanes) != 4:
            raise ValueError("LaneProcessor must be initialized with exactly 4 lanes.")

        if lane_curve_estimator is None:
            raise ValueError("LaneProcessor requires a pre-initialized LaneCurveEstimator instance.")

        self.lanes = lanes
        self.lane_curve_estimator = lane_curve_estimator

    @staticmethod
    def _process_single_class(lane_class,
                              confidence,
                              xywhn,
                              frame_shape,
                              confidence_threshold,
                              lane_points,
                              lane_confs):
        """Process a single lane class and directly append points and confidences to shared lists."""
        if confidence >= confidence_threshold:
            frame_height, frame_width = frame_shape
            cx = xywhn[0].item() * frame_width
            cy = xywhn[1].item() * frame_height
            lane_points[lane_class].append((int(cx), int(cy)))
            lane_confs[lane_class].append(confidence)

    @classmethod
    def from_yolo_result(cls, yolo_result, frame_shape, confidence_threshold=0.5, lane_curve_estimator=None):
        """Initialize LaneProcessor from YOLO results."""
        lane_boxes = yolo_result[0].boxes
        lane_classes = lane_boxes.cls
        lane_confidences = lane_boxes.conf
        lane_xywhns = lane_boxes.xywhn

        lane_points = {0: [], 1: [], 2: [], 3: []}
        lane_confs = {0: [], 1: [], 2: [], 3: []}

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(
                    cls._process_single_class,
                    int(lane_classes[i].item()),
                    lane_confidences[i].item(),
                    lane_xywhns[i],
                    frame_shape,
                    confidence_threshold,
                    lane_points,
                    lane_confs
                )
                for i in range(len(lane_classes))
            ]

        for future in futures:
            future.result()

        lanes = [
            Lane(points=lane_points[lane_class], confs=lane_confs[lane_class])
            if lane_points[lane_class] else None
            for lane_class in range(4)
        ]

        return cls(lanes=lanes, lane_curve_estimator=lane_curve_estimator)

    def estimate_lane_curves(self, use_weights=False):
        """Estimate lane curves in parallel and populate each lane's estimated points."""

        def _estimate_curve_for_lane(lane_class, lane):
            if lane is None or len(lane.points) < 3:
                return None

            lane_type = self.lane_labels[lane_class]
            weights = lane.confs if use_weights else None

            # Call the updated `predict_lane_points` method with lane_type
            _estimated_points = self.lane_curve_estimator.predict_lane_points(
                lane_type,
                lane.points,
                new_weights=weights
            )
            return _estimated_points

        if not self.lanes or len(self.lanes) != 4:
            raise ValueError("LaneProcessor expects exactly 4 lanes to estimate curves.")

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(_estimate_curve_for_lane, lane_class, lane) if lane is not None else None
                for lane_class, lane in enumerate(self.lanes)
            ]

        for i, future in enumerate(futures):
            if future is not None:  # Check if the lane existed and was processed
                estimated_points = future.result()
                if estimated_points is not None:
                    # Assign estimated points to the lane
                    # noinspection PyUnresolvedReferences
                    self.lanes[i].estimated_points = estimated_points

    def __repr__(self):
        return f"LaneProcessor(lanes={self.lanes}, lane_curve_estimator={self.lane_curve_estimator})"

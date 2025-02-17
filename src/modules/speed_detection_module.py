import cv2
import numpy as np
from src.modules.base_module import BaseModule
from src.modules.general_object_detection_module import GeneralObjectDetectionModule
from src.modules.image_reading_module import ImageReadingModule
from src.speed_detection.shift_processor import ShiftProcessor


class SpeedDetectionModule(BaseModule):
    def __init__(
            self,
            image_reading_module: ImageReadingModule,
            object_detection_module: GeneralObjectDetectionModule,
            # ROI definitions (normalized: 0-1)
            horiz_roi=(0.4, 0.6, 0.0, 1.0),  # (x_start, x_end, y_start, y_end) for horizontal shift
            radial_roi=(0.0, 1.0, 0.0, 1.0),  # for vertical (radial) shift (full frame)
            # Kalman filter parameters (tuned for noisy data but fast smoothing)
            kalman_process_noise=1e-3,
            kalman_measurement_noise=1e-3,
            # Weighting factors for radial displacement
            inner_weight=0.75, middle_weight=1.0, outer_weight=1.25,
            # Angular weighting (left/right of center)
            left_weight=1.0, right_weight=1.0
    ):
        super().__init__()

        self.object_detection_module = object_detection_module
        self.image_reading_module = image_reading_module

        self.shift_processor = ShiftProcessor(
            horiz_roi=horiz_roi,
            radial_roi=radial_roi,
            kalman_process_noise=kalman_process_noise,
            kalman_measurement_noise=kalman_measurement_noise,
            inner_weight=inner_weight, middle_weight=middle_weight, outer_weight=outer_weight,
            left_weight=left_weight, right_weight=right_weight
        )

    @staticmethod
    def _draw_dashed_line(img, pt1, pt2, color, thickness=1, dash_length=10, gap_length=5):
        pt1 = np.array(pt1, dtype=float)
        pt2 = np.array(pt2, dtype=float)
        line_length = np.linalg.norm(pt2 - pt1)
        if line_length == 0:
            return
        num_dashes = int(line_length // (dash_length + gap_length))
        if num_dashes < 1:
            num_dashes = 1
        direction = (pt2 - pt1) / line_length
        for i in range(num_dashes + 1):
            start = pt1 + (dash_length + gap_length) * i * direction
            end = start + dash_length * direction
            # Clamp the end point so it doesn't exceed pt2.
            if np.linalg.norm(end - pt1) > line_length:
                end = pt2
            cv2.line(img, tuple(start.astype(int)), tuple(end.astype(int)), color, thickness)

    def _draw_dashed_rectangle(self, img, top_left, bottom_right, color, thickness=1, dash_length=10, gap_length=5):
        x1, y1 = top_left
        x2, y2 = bottom_right
        self._draw_dashed_line(img, (x1, y1), (x2, y1), color, thickness, dash_length, gap_length)  # top edge
        self._draw_dashed_line(img, (x1, y2), (x2, y2), color, thickness, dash_length, gap_length)  # bottom edge
        self._draw_dashed_line(img, (x1, y1), (x1, y2), color, thickness, dash_length, gap_length)  # left edge
        self._draw_dashed_line(img, (x2, y1), (x2, y2), color, thickness, dash_length, gap_length)  # right edge

    @staticmethod
    def _draw_dashed_circle(img, center, radius, color, thickness=1, dash_length=10, gap_length=5):
        if radius <= 0:
            return
        # Convert dash and gap lengths into angular increments.
        dash_angle = dash_length / radius
        gap_angle = gap_length / radius
        angle = 0
        while angle < 2 * np.pi:
            start_angle = angle
            end_angle = min(angle + dash_angle, 2 * np.pi)
            pt1 = (int(center[0] + radius * np.cos(start_angle)),
                   int(center[1] + radius * np.sin(start_angle)))
            pt2 = (int(center[0] + radius * np.cos(end_angle)),
                   int(center[1] + radius * np.sin(end_angle)))
            cv2.line(img, pt1, pt2, color, thickness)
            angle += dash_angle + gap_angle

    def draw_rois(self, frame: np.ndarray):
        if frame is None:
            return frame

        # Get image dimensions.
        h, w = frame.shape[:2]

        # --- Draw the Horizontal ROI as a dashed rectangle ---
        x_start, x_end, y_start, y_end = self.shift_processor._compute_roi(frame, self.shift_processor.horiz_roi)
        self._draw_dashed_rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 255), thickness=1,
                                    dash_length=10, gap_length=5)

        # --- Draw the Radial ROI as concentric dashed circles ---
        cx, cy, ring1, ring2 = self.shift_processor._compute_concentric_rings(w, h)
        self._draw_dashed_circle(frame, (cx, cy), ring1, (0, 255, 255), thickness=1, dash_length=10, gap_length=5)
        self._draw_dashed_circle(frame, (cx, cy), ring2, (0, 255, 255), thickness=1, dash_length=10, gap_length=5)

        # --- Display the Shift Values (if available) in yellow ---
        if self.value is not None:
            shift_x, shift_y = self.value
            text = f"H: {shift_x:.2f} px/s, V: {shift_y:.2f} px/s"
            cv2.putText(frame, text, (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        return frame

    def perform(self):
        while not self._stop_event.is_set():
            if self.image_reading_module.value is not None:
                input_image = self.image_reading_module.value.copy()

                # Mask out moving objects to avoid affecting optical flow.
                if self.object_detection_module.value:
                    for obj in self.object_detection_module.value['moving_object_registry'].registry:
                        x1, y1, x2, y2 = obj.xyxy.astype(int)
                        input_image[y1:y2, x1:x2] = 0

                shifts = self.shift_processor.process(input_image.copy())
                self.value = shifts

import cv2
import numpy as np
import time


class ShiftProcessor:
    def __init__(self,
                 # ROI definitions (normalized: 0-1)
                 horiz_roi=(0.4, 0.6, 0.0, 1.0),  # (x_start, x_end, y_start, y_end) for horizontal shift
                 radial_roi=(0.0, 1.0, 0.0, 1.0),  # for vertical (radial) shift (full frame)
                 # Kalman filter parameters (tuned for noisy data but fast smoothing)
                 kalman_process_noise=1e-3,
                 kalman_measurement_noise=1e-3,
                 # Weighting factors for radial displacement
                 inner_weight=0.75, middle_weight=1.0, outer_weight=1.25,
                 # Angular weighting (left/right of center)
                 left_weight=1.1, right_weight=0.9):
        """
        Initialize the ShiftProcessor.

        Parameters:
          horiz_roi: tuple of normalized ROI coordinates for horizontal shift (x_start, x_end, y_start, y_end).
          radial_roi: tuple of normalized ROI coordinates for radial (vertical) shift.
          kalman_process_noise: Process noise covariance multiplier.
          kalman_measurement_noise: Measurement noise covariance multiplier.
          inner_weight, middle_weight, outer_weight: Weights for keypoints in inner, middle, and outer rings.
          left_weight, right_weight: Angular weights for keypoints on the left/right side of the image center.
        """
        # SIFT detector for keypoint extraction
        self.sift = cv2.SIFT_create()

        # Store ROI definitions
        self.horiz_roi = horiz_roi
        self.radial_roi = radial_roi

        # Weighting parameters
        self.inner_weight = inner_weight
        self.middle_weight = middle_weight
        self.outer_weight = outer_weight
        self.left_weight = left_weight
        self.right_weight = right_weight

        # Kalman filter setup
        self.kalman_process_noise = kalman_process_noise
        self.kalman_measurement_noise = kalman_measurement_noise
        self._init_kalman_filter()

        # Previous frames and time (for optical flow)
        self.prev_gray_h = None  # for horizontal ROI
        self.prev_gray_v = None  # for full-frame (radial) ROI
        self.prev_time = None

    def _init_kalman_filter(self):
        self.kalman = cv2.KalmanFilter(2, 2)
        self.kalman.transitionMatrix = np.eye(2, dtype=np.float32)
        self.kalman.measurementMatrix = np.eye(2, dtype=np.float32)
        self.kalman.processNoiseCov = np.eye(2, dtype=np.float32) * self.kalman_process_noise
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * self.kalman_measurement_noise
        self.kalman.errorCovPost = np.eye(2, dtype=np.float32)
        self.kalman.statePost = np.zeros((2, 1), dtype=np.float32)

    def _compute_roi(self, frame, roi_params):
        """Return integer ROI coordinates given normalized parameters."""
        h, w = frame.shape[:2]
        x_start = int(roi_params[0] * w)
        x_end = int(roi_params[1] * w)
        y_start = int(roi_params[2] * h)
        y_end = int(roi_params[3] * h)
        return x_start, x_end, y_start, y_end

    def _draw_roi(self, frame, roi_params, color=(255, 0, 0), thickness=2):
        x_start, x_end, y_start, y_end = self._compute_roi(frame, roi_params)
        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), color, thickness)

    def _compute_concentric_rings(self, w, h):
        """Compute the center and ring boundaries based on the image dimensions."""
        cx, cy = w // 2, h // 2
        # Maximum distance from center to a corner:
        d1 = np.linalg.norm(np.array([cx, cy]))
        d2 = np.linalg.norm(np.array([w - cx, cy]))
        d3 = np.linalg.norm(np.array([cx, h - cy]))
        d4 = np.linalg.norm(np.array([w - cx, h - cy]))
        R_max = max(d1, d2, d3, d4)
        ring1 = int(R_max / 3)
        ring2 = int(2 * R_max / 3)
        return cx, cy, ring1, ring2

    def _compute_horizontal_shift(self, prev_gray, curr_gray, dt):
        """Compute horizontal shift using keypoints in the horizontal ROI."""
        keypoints = self.sift.detect(prev_gray, None)
        shift_x = None
        if keypoints:
            pts_prev = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)
            pts_next, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts_prev, None)
            if pts_next is not None:
                good_prev = pts_prev[status.flatten() == 1]
                good_next = pts_next[status.flatten() == 1]
                if len(good_prev) > 0:
                    disp = good_next - good_prev
                    median_disp = np.median(disp, axis=0)
                    median_disp = np.squeeze(median_disp)
                    # Use only the x-component
                    if median_disp.ndim == 0:
                        shift_x = median_disp / dt
                    else:
                        shift_x = median_disp[0] / dt
        return shift_x

    def _compute_radial_shift(self, prev_gray, curr_gray, dt, w, h):
        """Compute radial (vertical) shift using keypoints from the full frame."""
        keypoints = self.sift.detect(prev_gray, None)
        shift_y = None
        if keypoints:
            pts_prev = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)
            pts_next, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts_prev, None)
            if pts_next is not None:
                good_prev = pts_prev[status.flatten() == 1].reshape(-1, 2)
                good_next = pts_next[status.flatten() == 1].reshape(-1, 2)
                if len(good_prev) > 0:
                    d = good_next - good_prev  # displacement vectors
                    cx, cy, ring1, ring2 = self._compute_concentric_rings(w, h)
                    vectors = good_prev - np.array([cx, cy])
                    norms = np.linalg.norm(vectors, axis=1)
                    # Avoid division by zero.
                    norms[norms == 0] = 1
                    u = vectors / norms[:, np.newaxis]
                    radial_disp = np.sum(d * u, axis=1)
                    # Radial weighting based on distance from center.
                    radial_weights = np.where(norms < ring1, self.inner_weight,
                                              np.where(norms < ring2, self.middle_weight, self.outer_weight))
                    # Angular weighting: keypoints left of center get one multiplier, right get another.
                    dx = good_prev[:, 0] - cx
                    angular_weights = np.where(dx < 0, self.left_weight, self.right_weight)
                    combined_weights = radial_weights * angular_weights
                    weighted_radial_disp = radial_disp * combined_weights
                    shift_y = np.median(weighted_radial_disp) / dt
        return shift_y

    def process(self, frame):
        """
        Process the input frame and return a tuple (horizontal_shift, vertical_shift) in px/s.
        The frame is also annotated with visual ROIs and rings.
        """
        h, w = frame.shape[:2]
        current_time = time.time()

        # --- Compute and draw ROIs ---
        # Horizontal ROI for horizontal shift:
        x_start_h, x_end_h, y_start_h, y_end_h = self._compute_roi(frame, self.horiz_roi)
        cv2.rectangle(frame, (x_start_h, y_start_h), (x_end_h, y_end_h), (255, 0, 0), 2)

        # For radial (vertical) shift we use the full frame;
        # draw concentric rings for visualization.
        cx, cy, ring1, ring2 = self._compute_concentric_rings(w, h)
        cv2.circle(frame, (cx, cy), ring1, (0, 255, 255), 2)
        cv2.circle(frame, (cx, cy), ring2, (0, 255, 255), 2)

        # --- Prepare grayscale ROIs ---
        roi_h = frame[y_start_h:y_end_h, x_start_h:x_end_h]
        roi_h_gray = cv2.cvtColor(roi_h, cv2.COLOR_BGR2GRAY)
        # For the radial (vertical) shift, use the full frame.
        roi_v_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- Initialize previous images if needed ---
        if self.prev_gray_h is None or self.prev_gray_v is None:
            self.prev_gray_h = roi_h_gray
            self.prev_gray_v = roi_v_gray
            self.prev_time = current_time
            # On first frame, no shift data is available.
            return (0.0, 0.0)

        dt = current_time - self.prev_time
        if dt == 0:
            dt = 1e-3

        # --- Compute shifts ---
        shift_x = self._compute_horizontal_shift(self.prev_gray_h, roi_h_gray, dt)
        if shift_x is None:
            shift_x = self.kalman.statePost[0, 0]
        shift_y = self._compute_radial_shift(self.prev_gray_v, roi_v_gray, dt, w, h)
        if shift_y is None:
            shift_y = self.kalman.statePost[1, 0]

        # --- Update Kalman filter ---
        measurement = np.array([[np.float32(shift_x)],
                                [np.float32(shift_y)]])
        self.kalman.correct(measurement)
        filtered = self.kalman.predict()
        final_shift_x = filtered[0, 0]
        final_shift_y = filtered[1, 0]

        # --- Update previous frames/time for the next iteration ---
        self.prev_gray_h = roi_h_gray
        self.prev_gray_v = roi_v_gray
        self.prev_time = current_time

        return (final_shift_x, final_shift_y)
